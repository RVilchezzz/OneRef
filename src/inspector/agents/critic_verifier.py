"""Critic / verifier (service agent). Rule-based, no model, so it costs no GPU pass.

Checks, in order:
1. Schema/format: a reader answer is missing (invalid after retry) -> ``is_valid=False``, reopen it.
2. Consistency across questions of the same image (per reader):
   a. discrimination says "No defect" but the expert sees a strong anomaly (score >= threshold);
   b. a localization answer names a 3x3 cell different from the expert's region while the expert is confident;
   c. (optional, off by default) discrimination says "No" yet defect-localization questions exist.
      Disabled by default: question presence reveals the label in MMAD (strict protocol).
3. Disagreement between readers is NOT a conflict by itself: the aggregator votes.

The reopen is granted at most ``Settings.max_reopen`` times per image (MAST: no unbounded loops);
``state.reopen_count`` is the guard.
"""
from __future__ import annotations

from typing import Any

from inspector.agents.base import BaseAgent
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import CriticOutput, ReaderOutput
from inspector.schemas.mmad import Question, SubTask
from inspector.schemas.state import InspectionState

_CELLS = [
    "top left", "top center", "top right", "middle left", "center",
    "middle right", "bottom left", "bottom center", "bottom right",
]


def says_no_defect(q: Question, letter: str) -> bool:
    return q.options.get(letter, "").strip().lower().rstrip(".") in {"no", "no defect"}


def says_yes_defect(q: Question, letter: str) -> bool:
    return q.options.get(letter, "").strip().lower().rstrip(".") in {"yes", "defect"}


def cell_in_text(text: str) -> str | None:
    """Return the 3x3 cell named in ``text`` ("top-right part" -> "top right"), if any."""
    t = text.lower().replace("-", " ").replace("_", " ")
    for c in sorted(_CELLS, key=len, reverse=True):  # longest first so "top center" wins over "center"
        if c in t:
            return c
    return None


class CriticVerifier(BaseAgent):
    name = "critic"

    def __init__(self, settings: Settings):
        self.s = settings

    def check_reader(self, state: InspectionState, answers: dict[str, ReaderOutput | None], who: str) -> tuple[bool, list[str], list[str]]:
        """Return (valid, reopen_ids, reasons) for one reader."""
        valid, reopen, reasons = True, [], []
        ev = state.visual_evidence
        strong = ev is not None and ev.confidence_score >= self.s.expert_flag_threshold
        has_loc = any(q.subtask == SubTask.DEFECT_LOCALIZATION for q in state.questions)
        for q in state.questions:
            a = answers.get(q.id)
            if a is None:
                valid = False
                reopen.append(q.id)
                reasons.append(f"{who}: no valid answer for {q.id}")
                continue
            if q.subtask == SubTask.ANOMALY_DISCRIMINATION and says_no_defect(q, a.selected_option):
                if strong:
                    reopen.append(q.id)
                    reasons.append(f"{who}: said no defect but the visual expert found a strong difference in the {ev.candidate_region}.")
                elif self.s.critic_use_question_structure and has_loc:
                    reopen.append(q.id)
                    reasons.append(f"{who}: said no defect but a defect-localization question exists.")
            if q.subtask == SubTask.DEFECT_LOCALIZATION and strong:
                cell = cell_in_text(q.options[a.selected_option])
                if cell and cell != ev.candidate_region:
                    reopen.append(q.id)
                    reasons.append(f"{who}: chose '{cell}' but the expert's strongest region is '{ev.candidate_region}'.")
        return valid, reopen, reasons

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        valid, reopen, reasons = True, [], []
        for who, ans in (("reader_a", state.reader_a_responses), ("reader_b", state.reader_b_responses)):
            v, r, why = self.check_reader(state, ans, who)
            valid &= v
            reopen += r
            reasons += why
        reopen = sorted(set(reopen))
        allowed = state.reopen_count < self.s.max_reopen
        if reopen and allowed:
            state.reopen_count += 1  # the single reopen is granted here
        verdict = CriticOutput(
            is_valid=valid,
            conflict_detected=bool(reopen),
            reopen_questions=reopen if allowed else [],
            critique_reason=" ".join(reasons),
        )
        state.critic_verdict = verdict
        rec["verdict"] = verdict.model_dump()
        return state
