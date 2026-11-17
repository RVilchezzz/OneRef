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
from collections import Counter
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.exceptions import ModelContextWindowExceededError, ModelRequestRejectedError, UnexpectedModelBehavior
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
        role = self.name.removesuffix("_reopen")
        return {"reader_a": state.reader_a_responses, "reader_b": state.reader_b_responses,
                "reader_c": state.reader_c_responses}[role]

    def _reopen_for_me(self, state: InspectionState) -> tuple[set[str], str | None]:
        """Questions this reader must re-answer and the reasons that concern it (targeted reopen)."""
        v = state.critic_verdict
        if v is None:
            return set(), None
        role = self.name.removesuffix("_reopen")
        if v.reopen_by_reader:
            return set(v.reopen_by_reader.get(role, [])), v.reason_by_reader.get(role) or v.critique_reason
        return set(v.reopen_questions), v.critique_reason  # verdict cached before targeted reopen

    def applies(self, state: InspectionState) -> bool:
        if not self.reopen:
            return True
        return bool(self._reopen_for_me(state)[0])

    # ---- agent ---------------------------------------------------------------
    async def ask_once(self, q: Question, prompt: str, images: list, rec: dict[str, Any],
                       settings: ModelSettings | None = None) -> ReaderAnswer | None:
        call: dict[str, Any] = {"question_id": q.id, "prompt": prompt}
        if settings:
            call["sampling"] = dict(settings)
        rec.setdefault("calls", []).append(call)
        with capture_run_messages() as messages:
            try:
                result = await self.agent.run([prompt, *images], deps=q, model_settings=settings)
            except (UnexpectedModelBehavior, ModelRequestRejectedError, ModelContextWindowExceededError) as e:  # invalid after retries or request refused => counted as an error
                call.update(raw=response_texts(messages), error=repr(e))
                return None
        usage = result.usage
        call.update(raw=response_texts(messages), tokens_in=usage.input_tokens, tokens_out=usage.output_tokens)
        rec["tokens_in"] = rec.get("tokens_in", 0) + usage.input_tokens
        rec["tokens_out"] = rec.get("tokens_out", 0) + usage.output_tokens
        return result.output

    async def ask(self, q: Question, state: InspectionState, critique: str | None, images: list, rec: dict[str, Any]) -> ReaderOutput | None:
        ev = state.visual_evidence if self.s.hint_on(state.source) else None
        prompt = build_prompt(q, ev.as_text() if ev else None, critique)
        k = self.s.reader_samples
        if k == 1:  # greedy (temperature 0), the setting of every system except B3
            out = await self.ask_once(q, prompt, images, rec)
            return ReaderOutput(question_id=q.id, **out.model_dump()) if out else None
        # B3: k sampled answers (seeds 0..k-1), majority vote; tie -> higher summed confidence, then letter
        outs = [await self.ask_once(q, prompt, images, rec, ModelSettings(temperature=self.s.reader_temperature, seed=i,
                                                                            max_tokens=self.s.max_new_tokens))
                for i in range(k)]
        valid = [o for o in outs if o is not None]
        if not valid:
            return None
        count = Counter(o.selected_option for o in valid)
        conf = Counter()
        for o in valid:
            conf[o.selected_option] += o.confidence
        best = min(count, key=lambda x: (-count[x], -conf[x], x))
        first = next(o for o in valid if o.selected_option == best)
        return ReaderOutput(question_id=q.id, selected_option=best, confidence=conf[best] / count[best],
                            brief_rationale=f"vote {count[best]}/{k}: {first.brief_rationale}")

    def image_parts(self, state: InspectionState) -> list:
        parts = super().image_parts(state)
        return parts if self.s.hint_on(state.source) else parts[:2]  # no hint => no expert crop either

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        store = self._store(state)
        if self.reopen:
            ids, critique = self._reopen_for_me(state)
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
