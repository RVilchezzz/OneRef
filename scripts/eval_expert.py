"""Evaluate the visual expert alone on a split (dev only while developing).

For every image I run the expert (query vs ``random_templates[0]``) and record the raw peak patch
distance and the peak location. Masks are read ONLY here, to score the expert, never as an input.

Reported per source and overall:
- AUROC of the peak distance, normal vs defective (threshold-free: 0.5 = chance, 1.0 = perfect);
- balanced accuracy of the current ``expert_flag_threshold`` (what the aggregator would use);
- peak-in-mask rate on defective images (does the hottest point fall on the real defect?).

    python scripts/eval_expert.py --split dev                     # DINOv3, DINOv2 fallback
    python scripts/eval_expert.py --split dev --model facebook/dinov2-base --limit 50

Writes ``outputs/expert_eval/<run>/per_image.jsonl`` and ``summary.json`` (a new folder per run,
never overwritten) plus a few heatmaps for the slides.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspector.agents.visual_expert import VisualExpert, anomaly_map, grid_cell  # noqa: E402
from inspector.config.settings import get_settings  # noqa: E402
from inspector.schemas.mmad import load_mmad  # noqa: E402


def auroc(scores_neg: list[float], scores_pos: list[float]) -> float:
    """Mann-Whitney AUROC: probability that a defective image scores above a normal one."""
    if not scores_neg or not scores_pos:
        return float("nan")
    allv = np.concatenate([scores_neg, scores_pos])
    ranks = allv.argsort().argsort().astype(float) + 1
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    r_pos = ranks[len(scores_neg):].sum()
    n_p, n_n = len(scores_pos), len(scores_neg)
    return float((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def load_mask(root: Path, sample_key: str, entry: dict, size: tuple[int, int]) -> np.ndarray | None:
    """Union of the image's mask file(s), resized to the query size; None if unavailable."""
    if not entry.get("mask_path"):
        return None
    base = root / (sample_key[: -len(entry["image_path"])] + entry["mask_path"])
    paths = [base] if base.is_file() else sorted(base.glob("*.png")) if base.is_dir() else []
    if not paths:
        return None
    m = np.zeros(size[::-1], dtype=bool)
    for p in paths:
        m |= np.asarray(Image.open(p).convert("L").resize(size, Image.NEAREST)) > 0
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the visual expert alone on a split")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="override the expert model id")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--gallery", type=int, default=12, help="how many heatmaps to keep for the slides")
    args = ap.parse_args()
    if args.split == "test":
        print("The expert is calibrated on dev; scoring on test only happens with the frozen config.", file=sys.stderr)

    s = get_settings()
    if args.model:
        s.expert_model = args.model
    ids = set(json.loads((s.splits_dir / f"{args.split}_ids.json").read_text(encoding="utf-8")))
    raw = json.loads(Path(s.mmad_json).read_text(encoding="utf-8"))
    samples = [x for x in load_mmad(s.mmad_json) if x.image_id in ids][: args.limit]

    run = args.run_name or f"{args.split}_{s.expert_model.split('/')[-1]}_{datetime.now():%Y%m%d-%H%M}"
    out = ROOT / "outputs" / "expert_eval" / run
    out.mkdir(parents=True, exist_ok=False)
    expert = VisualExpert(s, out)
    expert.load()
    print(f"expert: {expert.model_id} on {expert.device}, input {expert.size}px, patch {expert.patch}")

    rows, t0 = [], time.time()
    with open(out / "per_image.jsonl", "w", encoding="utf-8") as f:
        for i, x in enumerate(samples):
            q = Image.open(s.mmad_root / x.image_path).convert("RGB")
            r = Image.open(s.mmad_root / x.reference).convert("RGB")
            dist = anomaly_map(expert._features(q), expert._features(r))
            g = expert.size // expert.patch
            up = np.asarray(Image.fromarray(dist.reshape(g, g).astype(np.float32), mode="F").resize(q.size, Image.BICUBIC))
            py, px = np.unravel_index(int(up.argmax()), up.shape)
            mask = load_mask(s.mmad_root, x.image_path, raw[x.image_path], q.size) if x.is_defective else None
            row = {
                "image_id": x.image_id, "source": x.source, "category": x.category, "is_defective": x.is_defective,
                "peak": float(dist.max()), "mean": float(dist.mean()), "region": grid_cell(int(py), int(px), q.size[1], q.size[0]),
                "peak_in_mask": None if mask is None else bool(mask[py, px]),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            if i < args.gallery:  # heatmap + crop, same artefacts the pipeline produces
                expert.out_dir = out / "gallery"
                expert.out_dir.mkdir(exist_ok=True)
                expert.compute(q, r, x.image_id)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(samples)}  {(time.time() - t0) / (i + 1):.2f} s/img")
    expert.unload()

    def summarize(rs: list[dict]) -> dict:
        neg = [r["peak"] for r in rs if not r["is_defective"]]
        pos = [r["peak"] for r in rs if r["is_defective"]]
        thr_score = lambda p: (p - s.anomaly_norm_low) / (s.anomaly_norm_high - s.anomaly_norm_low)  # noqa: E731
        flag = lambda r: thr_score(r["peak"]) >= s.expert_flag_threshold  # noqa: E731
        tnr = np.mean([not flag(r) for r in rs if not r["is_defective"]]) if neg else float("nan")
        tpr = np.mean([flag(r) for r in rs if r["is_defective"]]) if pos else float("nan")
        hits = [r["peak_in_mask"] for r in rs if r["peak_in_mask"] is not None]
        return {
            "n_images": len(rs), "n_normal": len(neg), "n_defective": len(pos),
            "auroc_peak": auroc(neg, pos),
            "balanced_acc_at_threshold": float(np.nanmean([tnr, tpr])),
            "peak_in_mask_rate": float(np.mean(hits)) if hits else float("nan"), "n_masks": len(hits),
            "peak_normal_median": float(np.median(neg)) if neg else float("nan"),
            "peak_defective_median": float(np.median(pos)) if pos else float("nan"),
        }

    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    summary = {
        "split": args.split, "model": expert.model_id, "input_px": expert.size, "seconds_per_image": (time.time() - t0) / max(1, len(rows)),
        "calibration": {"norm_low": s.anomaly_norm_low, "norm_high": s.anomaly_norm_high, "flag_threshold": s.expert_flag_threshold},
        "overall": summarize(rows), "by_source": {k: summarize(v) for k, v in sorted(by_src.items())},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
