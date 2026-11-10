"""Aggregator: weighted vote per question.

* Each reader votes for its option with weight ``Settings.weights[agent][subtask]`` (default 1.0),
  fixed on the dev split.
* On anomaly discrimination the visual expert votes too (defect if score >= threshold).
* Ties: higher summed confidence, then alphabetical letter (deterministic).
* No valid vote at all -> ``selected_option=None`` (scored as an error).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from inspector.agents.base import BaseAgent
from inspector.agents.critic_verifier import says_no_defect, says_yes_defect
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import FinalAnswer
from inspector.schemas.mmad import SubTask
from inspector.schemas.state import InspectionState


class Aggregator(BaseAgent):
    name = "aggregator"

    def __init__(self, settings: Settings):
        self.s = settings

    def _w(self, agent: str, subtask: SubTask) -> float:
        return self.s.weights.get(agent, {}).get(subtask.value, self.s.default_weight)

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        ev = state.visual_evidence
        for q in state.questions:
            votes: dict[str, float] = defaultdict(float)
            conf: dict[str, float] = defaultdict(float)
            for agent, store in (("reader_a", state.reader_a_responses), ("reader_b", state.reader_b_responses)):
                a = store.get(q.id)
                if a is not None:
                    votes[a.selected_option] += self._w(agent, q.subtask)
                    conf[a.selected_option] += a.confidence
            if ev is not None and q.subtask == SubTask.ANOMALY_DISCRIMINATION:
                flag = ev.confidence_score >= self.s.expert_flag_threshold
                for letter in q.options:
                    if (flag and says_yes_defect(q, letter)) or (not flag and says_no_defect(q, letter)):
                        votes[letter] += self.s.expert_weight_discrimination
            best = min(votes, key=lambda k: (-votes[k], -conf[k], k)) if votes else None
            state.final_answers[q.id] = FinalAnswer(question_id=q.id, selected_option=best, votes=dict(votes))
        rec["final"] = {k: v.selected_option for k, v in state.final_answers.items()}
        return state
