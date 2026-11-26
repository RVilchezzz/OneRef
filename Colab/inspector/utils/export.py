"""Partial results of a run: every voter's answer per question (CSV) and the scores so far (JSON)."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from inspector.schemas.mmad import Sample
from inspector.schemas.state import InspectionState
from inspector.utils import metrics

READER_STEPS = {"reader_a": "2_reader_a", "reader_b": "3_reader_b"}


def _letter(outputs: dict, qid: str) -> str | None:
    out = outputs.get(qid)
    return out.selected_option if out else None


@lru_cache(maxsize=None)  # written once per image, re-used by every later tier (slow on Drive)
def _cached(run_dir: Path, step: str, image_id: str) -> InspectionState | None:
    """The state cached right after one pass (a reader's first pass = before any reopen)."""
    p = run_dir / step / (image_id.replace("/", "__") + ".json")
    return InspectionState.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def export_tier(run_dir: Path, samples: list[Sample], states: list[InspectionState]) -> dict:
    """Write ``<run_dir>/tiers/tier_<n images>/{votes.csv, metrics.json}`` and return the metrics."""
    out = Path(run_dir) / "tiers" / f"tier_{len(states):04d}"
    out.mkdir(parents=True, exist_ok=True)
    by_id = {s.image_id: s for s in samples}
    table = []
    for st in states:
        sample, ev = by_id[st.image_id], st.visual_evidence
        first = {r: _cached(Path(run_dir), step, st.image_id) for r, step in READER_STEPS.items()}
        for q in sample.questions:
            fa = st.final_answers.get(q.id)
            row = {"image_id": st.image_id, "source": sample.source, "category": sample.category,
                   "is_defective": sample.is_defective, "question_id": q.id, "subtask": q.subtask.value,
                   "truth": q.answer, "expert_score": ev.confidence_score if ev else None,
                   "expert_region": ev.candidate_region if ev else None}
            for r in READER_STEPS:
                row[f"{r}_first"] = _letter(getattr(first[r], f"{r}_responses"), q.id) if first[r] else None
                row[r] = _letter(getattr(st, f"{r}_responses"), q.id)
            row["judge"] = _letter(st.judge_answers, q.id)
            row["final"] = fa.selected_option if fa else None
            row["vote_weights"] = json.dumps(fa.votes) if fa else None
            table.append(row)
    with (out / "votes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    rows = metrics.build_rows([by_id[st.image_id] for st in states], states)  # scored on the final answers
    by_question = {t["question_id"]: t for t in table}
    voters = {"final": rows}
    for r in READER_STEPS:  # same questions, scored on one reader's answer alone
        voters[r] = [{**x, "correct": by_question[x["question_id"]][r] == by_question[x["question_id"]]["truth"]}
                     for x in rows]
    judged = [t for t in table if t["judge"]]
    result = {
        "images": len(states), "questions": len(table),
        "mean": {v: metrics.mean_accuracy(r) for v, r in voters.items()},
        "ci95_final": list(metrics.bootstrap_ci(rows)),
        "per_subtask": {v: {k.value: a for k, a in metrics.per_subtask_accuracy(r).items()} for v, r in voters.items()},
        "judge": {"answered": len(judged), "correct": sum(t["judge"] == t["truth"] for t in judged)},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
