"""LLM aggregator (optional, ``Settings.aggregator_model``): a text-only model settles contested questions.

The weighted vote (``Aggregator.tally``) is computed first. Where every voter (readers, expert, judge)
agrees, that option is final with no LLM call. Where they disagree, the LLM sees the question, the
candidate options, each source's answer and rationale under anonymous labels, the expert's flag on
discrimination, the judge's answer and the weighted tally, and must pick one candidate
(``output_validator`` -> ``ModelRetry``). It never sees the images. If it fails, the vote stands.
The judge is an input here: ``judge_mode="override"`` is not applied (the LLM has the last word).
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.exceptions import (ModelContextWindowExceededError, ModelRequestRejectedError, ModelServerError,
                                     UnexpectedModelBehavior)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from inspector.agents.aggregator import Aggregator
from inspector.agents.conflict_judge import JudgeDeps
from inspector.agents.llm import LLMAgent, response_texts
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import FinalAnswer, ReaderAnswer, ReaderOutput
from inspector.schemas.mmad import Question, SubTask
from inspector.schemas.state import InspectionState

AGGREGATOR_PROMPT = (
    "You are the lead of an industrial quality-control team and write the final answer to a "
    "multiple-choice question about one part. You cannot see the images. Several sources gave "
    "different answers: weigh their arguments, their confidence and the weighted tally, and choose "
    "the most likely correct option among the candidates only. Answer with the letter, a confidence "
    "between 0 and 1, and one short sentence of rationale."
)


class LLMAggregator(LLMAgent):
    name = "aggregator"

    def __init__(self, settings: Settings, model: Model | None = None):
        super().__init__(settings, settings.aggregator_model or "", model)
        self.rule = Aggregator(settings)

    def build(self, model: Model) -> Agent[JudgeDeps, ReaderAnswer]:
        agent = Agent(
            model,
            name="Aggregator",
            system_prompt=AGGREGATOR_PROMPT,
            output_type=PromptedOutput([ReaderAnswer]),
            deps_type=JudgeDeps,
            retries={"output": self.s.json_retries},
            model_settings=ModelSettings(temperature=0.0, max_tokens=self.s.max_new_tokens,
                                         **({"extra_body": self.s.aggregator_extra_body} if self.s.aggregator_extra_body else {})),
        )

        @agent.output_validator
        def is_a_candidate(ctx: RunContext[JudgeDeps], out: ReaderAnswer) -> ReaderAnswer:
            if out.selected_option not in ctx.deps.candidates:
                raise ModelRetry(f"'{out.selected_option}' is not a candidate. Choose one of {sorted(ctx.deps.candidates)}.")
            return out

        return agent

    @staticmethod
    def candidates(votes: dict[str, float], j: ReaderOutput | None) -> set[str]:
        return set(votes) | ({j.selected_option} if j else set())

    def contested(self, state: InspectionState) -> list[tuple[Question, dict[str, float], ReaderOutput | None]]:
        """Questions whose voters (readers, expert, judge) do not all give the same option."""
        out = []
        for q in state.questions:
            votes, _, j = self.rule.tally(state, q)
            if len(self.candidates(votes, j)) >= 2:
                out.append((q, votes, j))
        return out

    def applies(self, state: InspectionState) -> bool:
        return bool(self.contested(state))

    def prompt(self, state: InspectionState, q: Question, votes: dict[str, float], j: ReaderOutput | None) -> str:
        cands = sorted(self.candidates(votes, j))
        by_reader = {"reader_a": state.reader_a_responses, "reader_b": state.reader_b_responses,
                     "reader_c": state.reader_c_responses}
        sources = []
        for r in self.s.aggregate_readers:
            a = by_reader[r].get(q.id)
            if a is not None:
                sources.append(f"- Inspector {len(sources) + 1}: {a.selected_option} (confidence {a.confidence:.2f}): {a.brief_rationale}")
        ev = state.visual_evidence
        if ev is not None and q.subtask == SubTask.ANOMALY_DISCRIMINATION:
            sources.append(f"- {ev.as_text()}")
        if j is not None:
            sources.append(f"- Senior inspector (looked at the images): {j.selected_option} (confidence {j.confidence:.2f}): {j.brief_rationale}")
        opts = "\n".join(f"{k}. {q.options[k]}" for k in cands)
        tally = ", ".join(f"{k}={votes[k]:g}" for k in sorted(votes))
        return (f"Question: {q.text}\nCandidate options:\n{opts}\n\nSources:\n" + "\n".join(sources)
                + f"\n\nWeighted tally: {tally}")

    async def decide(self, state: InspectionState, q: Question, votes: dict[str, float], j: ReaderOutput | None,
                     rec: dict[str, Any]) -> ReaderOutput | None:
        deps = JudgeDeps(question=q, candidates=self.candidates(votes, j))
        call: dict[str, Any] = {"question_id": q.id, "prompt": self.prompt(state, q, votes, j)}
        rec.setdefault("calls", []).append(call)
        with capture_run_messages() as messages:
            try:
                result = await self.agent.run(call["prompt"], deps=deps)
            except (UnexpectedModelBehavior, ModelRequestRejectedError, ModelContextWindowExceededError, ModelServerError) as e:
                call.update(raw=response_texts(messages), error=repr(e))
                return None  # the weighted vote stands
        usage = result.usage
        call.update(raw=response_texts(messages), tokens_in=usage.input_tokens, tokens_out=usage.output_tokens)
        return ReaderOutput(question_id=q.id, **result.output.model_dump())

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        todo = self.contested(state)

        async def decide_all() -> list[ReaderOutput | None]:
            return await asyncio.gather(*(self.decide(state, q, v, j, rec) for q, v, j in todo))

        llm = {q.id: out for (q, _, _), out in zip(todo, self.run_async(decide_all()))} if todo else {}
        for q in state.questions:
            votes, conf, _ = self.rule.tally(state, q)
            best = min(votes, key=lambda k: (-votes[k], -conf[k], k)) if votes else None
            if llm.get(q.id) is not None:
                best = llm[q.id].selected_option
            state.final_answers[q.id] = FinalAnswer(question_id=q.id, selected_option=best, votes=dict(votes))
        rec["final"] = {k: v.selected_option for k, v in state.final_answers.items()}
        rec["llm_decided"] = [k for k, v in llm.items() if v is not None]
        return state
