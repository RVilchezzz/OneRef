"""Re-aggregate a finished run with other settings, without the GPU (ablations from one run).

The last cached pass before the aggregator (e.g. ``6b_judge``) already holds every answer: readers
A, B, C, the judge and the expert. This script applies ``Aggregator`` again with each variant's
settings and writes ``<out>/<variant>/7_aggregator/*.json``, so ``scripts/compare_runs.py`` can score
the variants like ordinary runs. A variant is a JSON object of ``Settings`` fields; every other field
comes from the environment, as in a normal run.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/reaggregate.py outputs/dev_full_S \
        --variant 'AB={"aggregate_readers": ["reader_a", "reader_b"], "judge_mode": "off"}' \
        --variant 'ABC={"judge_mode": "off"}' \
        --variant 'AB+judge={"aggregate_readers": ["reader_a", "reader_b"]}' \
        --out outputs/reagg/dev_full_S
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspector.agents.aggregator import Aggregator
from inspector.config.settings import Settings
from inspector.schemas.state import InspectionState


def last_pass(run_dir: Path) -> Path:
    passes = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name[0].isdigit() and p.name != "7_aggregator")
    if not passes:
        raise SystemExit(f"no cached pass in {run_dir}")
    return passes[-1]  # names sort in workflow order: 1_expert < ... < 6_reader_b_reopen < 6b_judge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--variant", action="append", required=True, help="name=JSON of Settings overrides")
    ap.add_argument("--from-pass", default=None, help="cached pass to start from (default: the last one)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src = args.run_dir / args.from_pass if args.from_pass else last_pass(args.run_dir)
    files = sorted(src.glob("*.json"))
    print(f"{len(files)} states from {src}")
    for spec in args.variant:
        name, raw = spec.split("=", 1)
        overrides = json.loads(raw)
        agg = Aggregator(Settings(**overrides))
        dst = args.out / name / "7_aggregator"
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            st = agg._run(InspectionState.model_validate_json(f.read_text(encoding="utf-8")), {})
            (dst / f.name).write_text(st.model_dump_json(), encoding="utf-8")
        (args.out / name / "variant.json").write_text(json.dumps({"from": str(src), "overrides": overrides}, indent=2), encoding="utf-8")
        print(f"{name}: {len(files)} images -> {dst}")


if __name__ == "__main__":
    main()
