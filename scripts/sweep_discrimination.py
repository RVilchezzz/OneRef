"""Calibrate the expert's vote on anomaly discrimination from a cached run (dev only, no GPU).

Grid over ``expert_flag_threshold`` x ``expert_weight_discrimination`` (and the readers that vote):
re-applies ``Aggregator`` in memory and reports, per setting, the balanced accuracy of discrimination,
FPR, missed defects and the mean of the 7 subtasks. The choice is made on dev and frozen for test.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/sweep_discrimination.py outputs/dev_full_S \
        --out results/compare/2026-11-17_discrimination
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspector.agents.aggregator import Aggregator
from inspector.config.settings import get_settings
from inspector.schemas.mmad import SubTask, load_mmad
from inspector.schemas.state import InspectionState
from inspector.utils import metrics

from reaggregate import last_pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--from-pass", default=None)
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0")
    ap.add_argument("--weights", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--readers", default="reader_a,reader_b;reader_a;reader_b", help="';'-separated voter sets")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    src = args.run_dir / args.from_pass if args.from_pass else last_pass(args.run_dir)
    raw = [p.read_text(encoding="utf-8") for p in sorted(src.glob("*.json"))]
    base = get_settings()
    by_id = {s.image_id: s for s in load_mmad(base.mmad_json)}
    print(f"{len(raw)} states from {src}")

    res = []
    for readers in args.readers.split(";"):
        for w in map(float, args.weights.split(",")):
            for t in map(float, args.thresholds.split(",")):
                if w == 0 and t != float(args.thresholds.split(",")[0]):
                    continue  # the threshold is irrelevant without the expert vote
                agg = Aggregator(base.model_copy(update={"aggregate_readers": readers.split(","),
                                                         "expert_weight_discrimination": w,
                                                         "expert_flag_threshold": t}))
                states = [agg._run(InspectionState.model_validate_json(r), {}) for r in raw]
                rows = metrics.build_rows([by_id[s.image_id] for s in states], states)
                disc = [r for r in rows if r["subtask"] == SubTask.ANOMALY_DISCRIMINATION]
                fpr, miss = metrics.fpr_and_miss(disc)
                res.append({"readers": readers, "expert_weight": w, "threshold": t if w else None,
                            "disc_bal_acc": metrics.balanced_accuracy(disc), "fpr": fpr, "miss": miss,
                            "mean7": metrics.mean_accuracy(rows)})

    res.sort(key=lambda r: -r["mean7"])
    lines = ["| Readers | Expert weight | Threshold | Discrimination (bal.) | FPR | Missed | Mean of 7 |",
             "|---|---|---|---|---|---|---|"]
    for r in res:
        th = "-" if r["threshold"] is None else f"{r['threshold']:.2f}"
        lines.append(f"| {r['readers']} | {r['expert_weight']:g} | {th} | {100*r['disc_bal_acc']:.1f} | "
                     f"{100*r['fpr']:.1f} | {100*r['miss']:.1f} | {100*r['mean7']:.1f} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "sweep.json").write_text(json.dumps({"from": str(src), "results": res}, indent=2), encoding="utf-8")
        (args.out / "sweep.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
