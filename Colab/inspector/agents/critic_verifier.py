"""Critic / verifier (service agent): deterministic rules + an optional LLM critic (pydantic-ai).

Rules (always on, no model), in order:
1. Schema/format: a reader answer is missing (invalid after retry) -> ``is_valid=False``, reopen it.
2. Consistency across questions of the same image (per reader):
   a. discrimination says "No defect" but the expert sees a strong anomaly (score >= threshold);
   b. a localization answer names a 3x3 cell different from the expert's region while the expert is confident;
   c. (optional, off by default) discrimination says "No" yet defect-localization questions exist.
      Disabled by default: question presence reveals the label in MMAD (strict protocol).
3. Disagreement between readers is NOT a conflict by itself: the aggregator votes.

LLM critic (``use_llm_critic``, Qwen3-VL-8B with a critic prompt): sees the images, the expert
evidence and both readers' answers; the rules are exposed to it as the ``run_consistency_rules`` tool
(PasteurAIze tool pattern). Its reopen list is merged with the rules' (union). If it fails, the
rules-only verdict stands. ``use_llm_critic=False`` is the "rules-only critic" ablation (no GPU pass).

The reopen is granted at most ``Settings.max_reopen`` times per image (MAST: no unbounded loops);
``state.reopen_count`` is the guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from inspector.agents.llm import LLMAgent, response_texts
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import CriticOutput, ReaderOutput
from inspector.schemas.mmad import Question, SubTask
from inspector.schemas.state import InspectionState

_CELLS = [
    "top left", "top center", "top right", "middle left", "center",
    "middle right", "bottom left", "bottom center", "bottom right",
]

CRITIC_PROMPT = (
    "You are the quality-control reviewer of an industrial inspection team. Image 1 is the inspected "
    "part, image 2 a normal reference, image 3 (if present) the region a visual expert found most "
    "different. Two inspectors answered multiple-choice questions about image 1. First call "
    "`run_consistency_rules` to get the automatic checks. Then decide which questions must be "
    "re-asked because an answer contradicts the images, the expert evidence, or another answer of "
    "the same inspector. Only list question ids that appear in the input. Do not re-ask a question "
    "just because the two inspectors disagree."
)


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


@dataclass
class CriticDeps:
    findings: list[str]        # rule findings, returned by the tool
    question_ids: set[str]     # valid ids for the output validator


class CriticVerifier(LLMAgent):
    name = "critic"

    def __init__(self, settings: Settings, model: Model | None = None):
        super().__init__(settings, settings.critic_model, model)

    # ---- LLM critic ------------------------------------------------------------
    def enabled(self) -> bool:
        return self.s.use_llm_critic

    def build(self, model: Model) -> Agent[CriticDeps, CriticOutput]:
        agent = Agent(
            model,
            name="Critic",
            system_prompt=CRITIC_PROMPT,
            output_type=PromptedOutput([CriticOutput]),
            deps_type=CriticDeps,
            retries={"output": self.s.json_retries},
            model_settings=ModelSettings(temperature=0.0, max_tokens=self.s.max_new_tokens),
        )

        @agent.tool
        def run_consistency_rules(ctx: RunContext[CriticDeps]) -> str:
            """Deterministic checks: missing answers, answers contradicting the visual expert."""
            return "\n".join(ctx.deps.findings) or "No rule violation found."

        @agent.output_validator
        def ids_exist(ctx: RunContext[CriticDeps], out: CriticOutput) -> CriticOutput:
            unknown = set(out.reopen_questions) - ctx.deps.question_ids
            if unknown:
                raise ModelRetry(f"Unknown question ids {sorted(unknown)}. Valid ids: {sorted(ctx.deps.question_ids)}.")
            return out

        return agent

    @staticmethod
    def describe(state: InspectionState) -> str:
        """Text input of the LLM critic: expert evidence + both readers' answers per question."""
        lines = [state.visual_evidence.as_text()] if state.visual_evidence else []
        for q in state.questions:
            opts = "; ".join(f"{k}. {v}" for k, v in sorted(q.options.items()))
            lines.append(f"[{q.id}] {q.text} ({opts})")
            for who, store in (("inspector A", state.reader_a_responses), ("inspector B", state.reader_b_responses)):
                a = store.get(q.id)
                lines.append(f"  {who}: " + (f"{a.selected_option} (conf {a.confidence:.2f}) {a.brief_rationale}" if a else "no valid answer"))
        return "\n".join(lines)

    # ---- rules -------------------------------------------------------------------
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

        if self.agent is not None:
            deps = CriticDeps(findings=list(reasons), question_ids={q.id for q in state.questions})
            prompt = self.describe(state)
            rec["prompt"] = prompt
            with capture_run_messages() as messages:
                try:
                    res = self.run_async(self.agent.run([prompt, *self.image_parts(state)], deps=deps))
                    usage = res.usage
                    rec.update(llm=res.output.model_dump(), tokens_in=usage.input_tokens, tokens_out=usage.output_tokens)
                    reopen += res.output.reopen_questions
                    if res.output.reopen_questions and res.output.critique_reason:
                        reasons.append(f"reviewer: {res.output.critique_reason}")
                except UnexpectedModelBehavior as e:  # LLM critic failed: the rules-only verdict stands
                    rec["llm_error"] = repr(e)
            rec["raw"] = response_texts(messages)

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
