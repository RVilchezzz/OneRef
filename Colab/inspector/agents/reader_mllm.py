"""MLLM reader agent (Reader A = Qwen3-VL-8B, Reader B = InternVL3-8B), pydantic-ai style.

One class, two instances. Each question is one ``agent.run([prompt, query, reference, crop])``:
the model answers in JSON (``PromptedOutput``), pydantic-ai validates it against ``ReaderAnswer``
and an ``output_validator`` rejects options the question does not offer (``ModelRetry``).
After ``json_retries`` failed retries the answer is ``None`` (counts as an error downstream).

``reopen=True`` makes this a *reopen pass*: only the questions flagged by the critic are re-asked,
with the critic's reason added to the prompt.
"""
from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from inspector.agents.llm import LLMAgent, response_texts
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import ReaderAnswer, ReaderOutput
from inspector.schemas.mmad import Question
from inspector.schemas.state import InspectionState

SYSTEM_PROMPT = (
    "You are an industrial quality inspector. Image 1 is the part to inspect. Image 2 is a normal "
    "(defect-free) reference of the same product. Image 3, if present, is a zoomed crop of the region "
    "that differs most from the reference. Answer the multiple-choice question with the letter of one "
    "of the given options, a confidence between 0 and 1, and one short sentence of rationale."
)


def build_prompt(q: Question, evidence_text: str | None, critique: str | None = None) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(q.options.items()))
    parts = []
    if evidence_text:
        parts.append(evidence_text)
    if critique:
        parts.append(f"A reviewer flagged a possible inconsistency in your previous answers: {critique} Reconsider carefully.")
    parts.append(f"Question: {q.text}\nOptions:\n{opts}")
    return "\n\n".join(parts)


class ReaderMLLM(LLMAgent):
    def __init__(self, settings: Settings, model_name: str, name: str, reopen: bool = False, model: Model | None = None):
        super().__init__(settings, model_name, model)
        self.name = name
        self.reopen = reopen

    def build(self, model: Model) -> Agent[Question, ReaderAnswer]:
        agent = Agent(
            model,
            name=self.name,
            system_prompt=SYSTEM_PROMPT,
            output_type=PromptedOutput([ReaderAnswer]),
            deps_type=Question,
            retries={"output": self.s.json_retries},
            model_settings=ModelSettings(temperature=0.0, max_tokens=self.s.max_new_tokens),
        )

        @agent.output_validator
        def option_is_offered(ctx: RunContext[Question], out: ReaderAnswer) -> ReaderAnswer:
            if out.selected_option not in ctx.deps.options:
                raise ModelRetry(f"'{out.selected_option}' is not an option. Choose one of {sorted(ctx.deps.options)}.")
            return out

        return agent

    # ---- which dict of the state this reader owns ---------------------------
    def _store(self, state: InspectionState) -> dict[str, ReaderOutput | None]:
        return state.reader_a_responses if "reader_a" in self.name else state.reader_b_responses

    def applies(self, state: InspectionState) -> bool:
        if not self.reopen:
            return True
        return bool(state.critic_verdict and state.critic_verdict.reopen_questions)

    # ---- agent ---------------------------------------------------------------
    async def ask(self, q: Question, state: InspectionState, critique: str | None, images: list, rec: dict[str, Any]) -> ReaderOutput | None:
        ev = state.visual_evidence
        prompt = build_prompt(q, ev.as_text() if ev else None, critique)
        call: dict[str, Any] = {"question_id": q.id, "prompt": prompt}
        rec.setdefault("calls", []).append(call)
        with capture_run_messages() as messages:
            try:
                result = await self.agent.run([prompt, *images], deps=q)
            except UnexpectedModelBehavior as e:  # invalid after retries => counted as an error
                call.update(raw=response_texts(messages), error=repr(e))
                return None
        usage = result.usage
        call.update(raw=response_texts(messages), tokens_in=usage.input_tokens, tokens_out=usage.output_tokens)
        rec["tokens_in"] = rec.get("tokens_in", 0) + usage.input_tokens
        rec["tokens_out"] = rec.get("tokens_out", 0) + usage.output_tokens
        return ReaderOutput(question_id=q.id, **result.output.model_dump())

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        store = self._store(state)
        if self.reopen:
            ids = set(state.critic_verdict.reopen_questions) if state.critic_verdict else set()
            critique = state.critic_verdict.critique_reason if state.critic_verdict else None
            todo = [q for q in state.questions if q.id in ids]
        else:
            critique, todo = None, state.questions
        images = self.image_parts(state)  # read once per image, shared by its questions

        async def ask_all() -> list[ReaderOutput | None]:  # concurrent => vLLM batches the questions
            return await asyncio.gather(*(self.ask(q, state, critique, images, rec) for q in todo))

        for q, out in zip(todo, self.run_async(ask_all())):
            store[q.id] = out
        rec["answers"] = {q.id: (store[q.id].selected_option if store.get(q.id) else None) for q in todo}
        return state
