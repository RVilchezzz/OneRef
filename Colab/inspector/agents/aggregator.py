"""Aggregator: weighted vote per question.

* Each reader votes for its option with weight ``Settings.weights[agent][subtask]`` (default 1.0),
  fixed on the dev split.
* On anomaly discrimination the visual expert votes too (defect if score >= threshold); weight and
  threshold can be set per MMAD source (``expert_weight_by_source``, ``expert_threshold_by_source``).
* Optional third reader (``reader_c``) votes like the others; ``Settings.aggregate_readers`` picks
  which readers count (ablations re-aggregated from cache, no GPU).
* Optional conflict judge: ``judge_mode="override"`` makes its answer final where it answered,
  ``"vote"`` adds it as one vote of ``judge_weight``, ``"off"`` ignores it.
* Ties: higher summed confidence, then alphabetical letter (deterministic).
* No valid vote at all -> ``selected_option=None`` (scored as an error).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from inspector.agents.base import BaseAgent
from inspector.agents.critic_verifier import says_no_defect, says_yes_defect
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import FinalAnswer, ReaderOutput
from inspector.schemas.mmad import Question, SubTask
from inspector.schemas.state import InspectionState


class Aggregator(BaseAgent):
    name = "aggregator"

    def __init__(self, settings: Settings):
        self.s = settings

    def _w(self, agent: str, subtask: SubTask) -> float:
        return self.s.weights.get(agent, {}).get(subtask.value, self.s.default_weight)

    def tally(self, state: InspectionState, q: Question) -> tuple[dict[str, float], dict[str, float], ReaderOutput | None]:
        """Weighted votes and summed confidences per option for one question, plus the judge's answer (if any)."""
        ev = state.visual_evidence
        votes: dict[str, float] = defaultdict(float)
        conf: dict[str, float] = defaultdict(float)
        stores = {"reader_a": state.reader_a_responses, "reader_b": state.reader_b_responses,
                  "reader_c": state.reader_c_responses}
        for agent in self.s.aggregate_readers:
            a = stores[agent].get(q.id)
            if a is not None:
                votes[a.selected_option] += self._w(agent, q.subtask)
                conf[a.selected_option] += a.confidence
        if ev is not None and q.subtask == SubTask.ANOMALY_DISCRIMINATION:
            w_exp = self.s.expert_weight_by_source.get(state.source, self.s.expert_weight_discrimination)
            thr = self.s.expert_threshold_by_source.get(state.source, self.s.expert_flag_threshold)
            flag = ev.confidence_score >= thr
            for letter in q.options:
                if w_exp and ((flag and says_yes_defect(q, letter)) or (not flag and says_no_defect(q, letter))):
                    votes[letter] += w_exp
        j = state.judge_answers.get(q.id) if self.s.judge_mode != "off" else None
        if j is not None and self.s.judge_mode == "vote":
            votes[j.selected_option] += self.s.judge_weight
            conf[j.selected_option] += j.confidence
        return votes, conf, j

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        for q in state.questions:
            votes, conf, j = self.tally(state, q)
            best = min(votes, key=lambda k: (-votes[k], -conf[k], k)) if votes else None
            if j is not None and self.s.judge_mode == "override":
                best = j.selected_option
            state.final_answers[q.id] = FinalAnswer(question_id=q.id, selected_option=best, votes=dict(votes))
        rec["final"] = {k: v.selected_option for k, v in state.final_answers.items()}
        return state
