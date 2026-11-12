"""CLI entry point.

    python -m inspector.main --split dev  --limit 50 --run-name dev01
    python -m inspector.main --split test --frozen-config --run-name final     # run ONCE
    python -m inspector.main --split dev --limit 5 --mock                       # GPU-free smoke test
    python -m inspector.main --split dev --per-source 4 --run-name quick         # few images of every source

Needs ``src`` on PYTHONPATH (``pip install -e .`` or ``set PYTHONPATH=src``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inspector.config.settings import get_settings
from inspector.orchestrator.graph import build_steps
from inspector.orchestrator.pass_runner import PassRunner
from inspector.schemas.mmad import load_mmad
from inspector.schemas.state import InspectionState
from inspector.utils import metrics
from inspector.utils.splits import make_split, sample_per_source, save_split
from inspector.utils.telemetry import Telemetry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OneRef multi-agent inspector (strict MMAD 1-shot)")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images of the split")
    ap.add_argument("--per-source", type=int, default=None, help="use only K images per source (half normal, half defective)")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--mock", action="store_true", help="GPU-free mock agents (smoke test)")
    ap.add_argument("--frozen-config", action="store_true", help="required for --split test: config is frozen")
    args = ap.parse_args(argv)

    if args.split == "test" and not args.frozen_config:
        print("Refusing to run the test split without --frozen-config (tune on dev only).", file=sys.stderr)
        return 2

    s = get_settings()
    samples = load_mmad(s.mmad_json)
    dev, test = make_split(samples, s.dev_fraction, s.seed)
    save_split(dev, test, s.splits_dir)
    chosen = dev if args.split == "dev" else test
    if args.per_source:
        chosen = sample_per_source(chosen, args.per_source, s.seed)
    chosen = chosen[: args.limit]

    run_dir = Path(s.outputs_dir) / args.run_name
    tel = Telemetry(run_dir / "telemetry.jsonl", run=args.run_name, use_logfire=s.use_logfire)
    states = [InspectionState.from_sample(x, str(s.mmad_root)) for x in chosen]
    states = PassRunner(run_dir, tel).run(build_steps(s, run_dir, mock=args.mock), states)

    rows = metrics.build_rows(chosen, states)
    acc = metrics.per_subtask_accuracy(rows)
    for k, v in acc.items():
        print(f"{k.value:26s} {v:6.1%}")
    lo, hi = metrics.bootstrap_ci(rows)
    print(f"{'MEAN (7 subtasks)':26s} {metrics.mean_accuracy(rows):6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  n_img={len(states)}")
    tel.write({"event": "final_metrics", "mean": metrics.mean_accuracy(rows), "per_subtask": {k.value: v for k, v in acc.items()}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
