"""Conflict judge (optional, ``Settings.judge_model``): a model of another family settles disagreements.

It runs after the reopen passes, on the readers' *final* answers, and only on the questions whose
voters (``Settings.aggregate_readers``) disagree or where a reader gave no valid answer. It sees the
same images as the readers, the question, and only the candidate options the readers proposed, each
with its rationale, under anonymous labels ("Inspector 1", "Inspector 2"...), so the judge cannot
prefer a model by name. It must pick one of those candidates (``output_validator`` -> ``ModelRetry``).

Why choose among candidates instead of re-asking: a pass that re-reads the question without new
information rarely changes the answer (reopen in the Nov 11 run: 4 questions, 0 corrected); a
judge from a different family that compares the arguments is a second, independent opinion.
How the aggregator uses its answer: ``Settings.judge_mode`` ("override" | "vote" | "off").
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.exceptions import (ModelContextWindowExceededError, ModelRequestRejectedError, ModelServerError,
                                     UnexpectedModelBehavior)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from inspector.agents.llm import LLMAgent, response_texts
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import ReaderAnswer, ReaderOutput
from inspector.schemas.mmad import Question
from inspector.schemas.state import InspectionState

JUDGE_PROMPT = (
    "You are the senior inspector of an industrial quality-control team. Image 1 is the part to "
    "inspect. Image 2 is a normal (defect-free) reference of the same product. Image 3, if present, "
    "is a zoomed crop of the region that differs most from the reference. Several inspectors gave "
    "different answers to a multiple-choice question about image 1. Look at the images yourself, "
    "weigh their arguments, and choose the correct option among the candidates only. Answer with "
    "the letter, a confidence between 0 and 1, and one short sentence of rationale."
)


@dataclass
class JudgeDeps:
    question: Question
    candidates: set[str]


def stores(state: InspectionState) -> dict[str, dict[str, ReaderOutput | None]]:
    return {"reader_a": state.reader_a_responses, "reader_b": state.reader_b_responses,
            "reader_c": state.reader_c_responses}


def conflicts(state: InspectionState, readers: list[str]) -> list[tuple[Question, list[ReaderOutput]]]:
    """Questions to judge: the valid answers of ``readers`` propose >= 2 distinct options.
    Returns each question with those answers (a single valid answer leaves nothing to choose)."""
    out = []
    by_reader = stores(state)
    for q in state.questions:
        answers = [by_reader[r].get(q.id) for r in readers if r in by_reader]
        answered = [a for a in answers if a is not None and a.selected_option in q.options]
        if not answered:
            continue
        distinct = {a.selected_option for a in answered}
        if len(distinct) >= 2:
            out.append((q, answered))
    return out


def build_prompt(q: Question, answered: list[ReaderOutput], evidence_text: str | None) -> str:
    cands = sorted({a.selected_option for a in answered})
    opts = "\n".join(f"{k}. {q.options[k]}" for k in cands)
    lines = [evidence_text] if evidence_text else []
    lines.append(f"Question: {q.text}\nCandidate options:\n{opts}")
    lines.append("Inspectors' answers:")
    for i, a in enumerate(answered, 1):
        lines.append(f"- Inspector {i}: {a.selected_option} (confidence {a.confidence:.2f}): {a.brief_rationale}")
    return "\n\n".join(lines)


class ConflictJudge(LLMAgent):
    name = "judge"

    def __init__(self, settings: Settings, model: Model | None = None):
        super().__init__(settings, settings.judge_model or "", model)

    def build(self, model: Model) -> Agent[JudgeDeps, ReaderAnswer]:
        agent = Agent(
            model,
            name="Judge",
            system_prompt=JUDGE_PROMPT,
            output_type=PromptedOutput([ReaderAnswer]),
            deps_type=JudgeDeps,
            retries={"output": self.s.json_retries},
            model_settings=ModelSettings(temperature=0.0, max_tokens=self.s.max_new_tokens,
                                         **({"extra_body": self.s.judge_extra_body} if self.s.judge_extra_body else {})),
        )

        @agent.output_validator
        def is_a_candidate(ctx: RunContext[JudgeDeps], out: ReaderAnswer) -> ReaderAnswer:
            if out.selected_option not in ctx.deps.candidates:
                raise ModelRetry(f"'{out.selected_option}' is not a candidate. Choose one of {sorted(ctx.deps.candidates)}.")
            return out

        return agent

    def applies(self, state: InspectionState) -> bool:
        return bool(conflicts(state, self.s.aggregate_readers))

    def image_parts(self, state: InspectionState) -> list:
        parts = super().image_parts(state)
        return parts if self.s.hint_on(state.source) else parts[:2]

    async def judge(self, q: Question, answered: list[ReaderOutput], state: InspectionState, images: list,
                    rec: dict[str, Any]) -> ReaderOutput | None:
        ev = state.visual_evidence if self.s.hint_on(state.source) else None
        prompt = build_prompt(q, answered, ev.as_text() if ev else None)
        deps = JudgeDeps(question=q, candidates={a.selected_option for a in answered})
        call: dict[str, Any] = {"question_id": q.id, "prompt": prompt}
        rec.setdefault("calls", []).append(call)
        with capture_run_messages() as messages:
            try:
                result = await self.agent.run([prompt, *images], deps=deps)
            except (UnexpectedModelBehavior, ModelRequestRejectedError, ModelContextWindowExceededError, ModelServerError) as e:
                call.update(raw=response_texts(messages), error=repr(e))
                return None  # no judgement: the readers' vote stands
        usage = result.usage
        call.update(raw=response_texts(messages), tokens_in=usage.input_tokens, tokens_out=usage.output_tokens)
        return ReaderOutput(question_id=q.id, **result.output.model_dump())

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        todo = conflicts(state, self.s.aggregate_readers)
        images = self.image_parts(state)

        async def judge_all() -> list[ReaderOutput | None]:
            return await asyncio.gather(*(self.judge(q, ans, state, images, rec) for q, ans in todo))

        for (q, _), out in zip(todo, self.run_async(judge_all())):
            state.judge_answers[q.id] = out
        rec["answers"] = {q.id: (state.judge_answers[q.id].selected_option if state.judge_answers.get(q.id) else None) for q, _ in todo}
        return state
