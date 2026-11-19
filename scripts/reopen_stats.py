"""What the verifier's reopens did, per reader (dev only).

Compares each reader's answers before the critic (``3_reader_b`` by default, which holds A and B)
with its answers after its reopen pass (``5_reader_a_reopen`` / ``6_reader_b_reopen``), on the
questions the critic sent back to that reader. Counts: reopened, changed, fixed (wrong -> right),
broken (right -> wrong), recovered (invalid -> valid), plus the number of images with at least one reopen.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/reopen_stats.py gemma=outputs/dev_full_S \
        rules=outputs/dev_full_S_rules --out results/compare/2026-11-17_verifier
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspector.schemas.state import InspectionState

AFTER = {"reader_a": "5_reader_a_reopen", "reader_b": "6_reader_b_reopen"}


def load(step_dir: Path) -> dict[str, InspectionState]:
    return {p.name: InspectionState.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(step_dir.glob("*.json"))}


def store(st: InspectionState, reader: str):
    return st.reader_a_responses if reader == "reader_a" else st.reader_b_responses


def stats(run_dir: Path, before_step: str) -> dict:
    before = load(run_dir / before_step)
    critic = load(run_dir / "4_critic")
    out: dict = {"images": len(critic),
                 "images_flagged": sum(1 for c in critic.values() if c.critic_verdict and (c.critic_verdict.reopen_questions or any(c.critic_verdict.reopen_by_reader.values())))}
    for reader, step in AFTER.items():
        after = load(run_dir / step)
        c = dict(reopened=0, changed=0, fixed=0, broken=0, recovered=0, lost=0, wrong_to_wrong=0)
        for name, st in critic.items():
            v = st.critic_verdict
            if v is None or name not in after or name not in before:
                continue
            qids = v.reopen_by_reader.get(reader, []) if v.reopen_by_reader else v.reopen_questions
            truth = {q.id: q.answer for q in st.questions}
            for qid in qids:
                b, a = store(before[name], reader).get(qid), store(after[name], reader).get(qid)
                c["reopened"] += 1
                pb, pa = (b.selected_option if b else None), (a.selected_option if a else None)
                if pb != pa:
                    c["changed"] += 1
                ok_b, ok_a = pb == truth.get(qid), pa == truth.get(qid)
                c["fixed"] += (not ok_b) and ok_a
                c["broken"] += ok_b and not ok_a
                c["recovered"] += pb is None and pa is not None
                c["lost"] += pb is not None and pa is None
                c["wrong_to_wrong"] += (not ok_b) and (not ok_a) and pb != pa
        out[reader] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--before-step", default="3_reader_b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        res[label] = stats(Path(path), args.before_step)
    lines = ["| Run | Reader | Images flagged | Reopened | Changed | Fixed | Broken | Recovered (invalid->valid) | Lost |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, r in res.items():
        for reader in AFTER:
            c = r[reader]
            lines.append(f"| {label} | {reader} | {r['images_flagged']}/{r['images']} | {c['reopened']} | {c['changed']} | "
                         f"{c['fixed']} | {c['broken']} | {c['recovered']} | {c['lost']} |")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "reopen_stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (out / "reopen_stats.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
