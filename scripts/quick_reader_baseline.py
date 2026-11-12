"""Quick local reader run on a small stratified dev sample, with and without the expert's hint.

This is a smoke test of the core idea on my 6 GB GPU (Qwen3-VL-4B in 4-bit through
``transformers``), not the real B1: the real readers are the 8B models served with vLLM on Colab.
It answers the same questions twice:
- ``noexpert``: part + reference + question (a single-model baseline);
- ``expert``: the same, plus one sentence with the visual expert's most anomalous 3x3 region.

    python scripts/quick_reader_baseline.py --per-source 12 \
        --expert-run outputs/expert_eval/dev_dinov3-vitb16_v1

Writes ``outputs/quick_reader/<run>/answers.jsonl`` and ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspector.config.settings import get_settings  # noqa: E402
from inspector.schemas.mmad import load_mmad  # noqa: E402
from inspector.utils import metrics  # noqa: E402

SYSTEM = ("You are an industrial quality inspector. Image 1 is the part to inspect. Image 2 is a normal "
          "(defect-free) reference of the same product. Answer the multiple-choice question with the "
          "letter of one option only.")


def sample_images(samples, per_source: int, seed: int):
    """Same number of images per source, half normal and half defective when possible."""
    rng, picked = random.Random(seed), []
    by = defaultdict(list)
    for s in samples:
        by[(s.source, s.is_defective)].append(s)
    for src in sorted({s.source for s in samples}):
        for d in (False, True):
            g = sorted(by[(src, d)], key=lambda s: s.image_id)
            rng.shuffle(g)
            picked += g[: per_source // 2]
    return picked


def load_img(path: Path, side: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--per-source", type=int, default=12)
    ap.add_argument("--side", type=int, default=448, help="max image side in pixels")
    ap.add_argument("--expert-run", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bits", type=int, choices=[4, 16], default=4, help="4 = nf4 (6 GB GPU / T4), 16 = bf16 (L4)")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    s = get_settings()
    dev_ids = set(json.loads((s.splits_dir / "dev_ids.json").read_text()))
    samples = sample_images([x for x in load_mmad(s.mmad_json) if x.image_id in dev_ids], args.per_source, args.seed)
    region = {}
    for line in open(args.expert_run / "per_image.jsonl"):
        r = json.loads(line)
        region[r["image_id"]] = r["region"]
    n_q = sum(len(x.questions) for x in samples)
    print(f"{len(samples)} images, {n_q} questions x 2 conditions")

    proc = AutoProcessor.from_pretrained(args.model)
    quant = (dict(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"))
             if args.bits == 4 else dict(dtype=torch.bfloat16))
    model = AutoModelForImageTextToText.from_pretrained(args.model, device_map="cuda", **quant).eval()

    run = f"{args.model.split('/')[-1]}_{'4bit' if args.bits == 4 else 'bf16'}_{datetime.now():%Y%m%d-%H%M}"
    out = ROOT / "outputs" / "quick_reader" / run
    out.mkdir(parents=True)
    rows = {"noexpert": [], "expert": []}
    t0, done = time.time(), 0
    with open(out / "answers.jsonl", "w") as f:
        for x in samples:
            q_img, r_img = load_img(s.mmad_root / x.image_path, args.side), load_img(s.mmad_root / x.reference, args.side)
            for q in x.questions:
                opts = "\n".join(f"{k}. {v}" for k, v in sorted(q.options.items()))
                for cond in ("noexpert", "expert"):
                    hint = (f"A visual comparison with the reference found that the region that differs most is: "
                            f"{region[x.image_id]}.\n\n") if cond == "expert" else ""
                    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                            {"role": "user", "content": [{"type": "image", "image": q_img}, {"type": "image", "image": r_img},
                                                         {"type": "text", "text": f"{hint}Question: {q.text}\nOptions:\n{opts}\nAnswer with the letter only."}]}]
                    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                    text = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                    m = re.search(r"\b([A-D])\b", text.strip().upper())
                    pred = m.group(1) if m and m.group(1) in q.options else None  # invalid = error, stays in the denominator
                    row = {"image_id": x.image_id, "question_id": q.id, "subtask": q.subtask, "correct": pred == q.answer,
                           "is_defective": x.is_defective, "source": x.source}
                    rows[cond].append(row)
                    f.write(json.dumps({**row, "subtask": q.subtask.value, "cond": cond, "pred": pred, "raw": text}) + "\n")
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{n_q} questions, {(time.time() - t0) / done:.1f} s per question (both conditions)")

    summary = {"model": args.model + (" (4-bit nf4)" if args.bits == 4 else " (bf16)"), "gpu": torch.cuda.get_device_name(0), "n_images": len(samples), "n_questions": n_q, "seconds_per_question_pair": (time.time() - t0) / max(1, done)}
    for cond, rs in rows.items():
        lo, hi = metrics.bootstrap_ci(rs)
        summary[cond] = {"mean": metrics.mean_accuracy(rs), "ci95": [lo, hi],
                         "per_subtask": {k.value: v for k, v in metrics.per_subtask_accuracy(rs).items()},
                         "by_source": {src: metrics.mean_accuracy([r for r in rs if r["source"] == src]) for src in sorted({r["source"] for r in rs})}}
    a, b, p = metrics.mcnemar_exact(rows["expert"], rows["noexpert"])
    summary["mcnemar_expert_vs_noexpert"] = {"expert_only_right": a, "noexpert_only_right": b, "p_value": p}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
