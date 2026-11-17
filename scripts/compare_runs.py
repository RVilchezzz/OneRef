"""Compare finished runs question by question (dev only).

For each run: accuracy per subtask (discrimination = balanced accuracy), mean of the subtasks with its
image-level bootstrap 95 % CI, FPR / missed-defect rate, and the mean per source.
Against the first run: exact McNemar on the paired questions.
For every pair of runs: complementarity, i.e. how often one is right when the other is wrong, the
agreement rate and the "oracle" accuracy (right if either is right). Readers worth combining are the
ones that are wrong on *different* questions, not just the most accurate ones.

The answer of a run is read from its last cached pass (``7_aggregator`` by default).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/compare_runs.py \
        hint=outputs/dev_ps20_qwen4b_hint_full nohint=outputs/dev_ps20_qwen4b_hint_none \
        --out results/compare/2026-11-16_hint
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from inspector.config.settings import get_settings
from inspector.schemas.mmad import SubTask, load_mmad
from inspector.schemas.state import InspectionState
from inspector.utils import metrics


def load_run(run_dir: Path, step: str, by_id: dict) -> list[dict]:
    states = [InspectionState.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted((run_dir / step).glob("*.json"))]
    if not states:
        raise SystemExit(f"no cached states in {run_dir / step}")
    return metrics.build_rows([by_id[s.image_id] for s in states], states)


def summary(rows: list[dict]) -> dict:
    acc = metrics.per_subtask_accuracy(rows)
    lo, hi = metrics.bootstrap_ci(rows)
    fpr, miss = metrics.fpr_and_miss([r for r in rows if r["subtask"] == SubTask.ANOMALY_DISCRIMINATION])
    per_source = defaultdict(list)
    for r in rows:
        per_source[r["source"]].append(r)
    return {
        "n_images": len({r["image_id"] for r in rows}), "n_questions": len(rows),
        "per_subtask": {k.value: v for k, v in acc.items()},
        "mean": metrics.mean_accuracy(rows), "ci95": [lo, hi], "fpr": fpr, "miss": miss,
        "mean_per_source": {s: metrics.mean_accuracy(rs) for s, rs in sorted(per_source.items())},
    }


def pair(rows_a: list[dict], rows_b: list[dict]) -> dict:
    ca = {r["question_id"]: r["correct"] for r in rows_a}
    cb = {r["question_id"]: r["correct"] for r in rows_b}
    ids = sorted(ca.keys() & cb.keys())
    both = sum(ca[i] and cb[i] for i in ids)
    a_only = sum(ca[i] and not cb[i] for i in ids)
    b_only = sum(cb[i] and not ca[i] for i in ids)
    _, _, p = metrics.mcnemar_exact(rows_a, rows_b)
    return {"paired_questions": len(ids), "both_right": both, "only_first": a_only, "only_second": b_only,
            "both_wrong": len(ids) - both - a_only - b_only,
            "oracle_acc": (both + a_only + b_only) / len(ids) if ids else float("nan"),
            "mcnemar_p": p}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def to_markdown(runs: dict[str, dict], pairs: dict[str, dict]) -> str:
    names = list(runs)
    subtasks = [s.value for s in SubTask if any(s.value in r["per_subtask"] for r in runs.values())]
    out = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    out.append("| Images / questions | " + " | ".join(f"{r['n_images']} / {r['n_questions']}" for r in runs.values()) + " |")
    for s in subtasks:
        out.append(f"| {s} | " + " | ".join(pct(r["per_subtask"].get(s, float("nan"))) for r in runs.values()) + " |")
    out.append("| **Mean of subtasks** [95 % CI] | " + " | ".join(
        f"**{pct(r['mean'])}** [{pct(r['ci95'][0])}, {pct(r['ci95'][1])}]" for r in runs.values()) + " |")
    out.append("| FPR (good parts called defective) | " + " | ".join(pct(r["fpr"]) for r in runs.values()) + " |")
    out.append("| Missed defects | " + " | ".join(pct(r["miss"]) for r in runs.values()) + " |")
    sources = sorted({s for r in runs.values() for s in r["mean_per_source"]})
    for s in sources:
        out.append(f"| Mean, {s} | " + " | ".join(pct(r["mean_per_source"].get(s, float("nan"))) for r in runs.values()) + " |")
    out += ["", "| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |",
            "|---|---|---|---|---|---|---|---|"]
    for k, v in pairs.items():
        out.append(f"| {k} | {v['paired_questions']} | {v['both_right']} | {v['only_first']} | {v['only_second']} | "
                   f"{v['both_wrong']} | {pct(v['oracle_acc'])} | {v['mcnemar_p']:.3g} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--step", default="7_aggregator", help="cached pass whose final answers are scored")
    ap.add_argument("--out", default=None, help="folder for summary.json and summary.md")
    args = ap.parse_args()

    by_id = {s.image_id: s for s in load_mmad(get_settings().mmad_json)}
    rows = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        rows[label] = load_run(Path(path), args.step, by_id)
    runs = {k: summary(v) for k, v in rows.items()}
    pairs = {f"{a} vs {b}": pair(rows[a], rows[b]) for a, b in itertools.combinations(rows, 2)}
    md = to_markdown(runs, pairs)
    print(md)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"runs": runs, "pairs": pairs, "args": vars(args)}, indent=2), encoding="utf-8")
        (out / "summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
