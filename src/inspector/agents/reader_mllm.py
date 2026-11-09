"""MLLM reader agent (Reader A = Qwen3-VL-8B, Reader B = InternVL3-8B).

One class, two instances. Input per question: query image, normal reference, expert crop,
expert text hint, question + options. Output: JSON forced by the prompt, parsed tolerantly and
validated with Pydantic (``ReaderOutput``). One retry on malformed output; after that the answer is
``None`` (counts as an error downstream, never silently dropped).

``reopen=True`` makes this a *reopen pass*: only the questions flagged by the critic are re-asked,
with the critic's reason added to the prompt.

The generation backend is isolated in ``_generate`` so it can be swapped (vLLM) or mocked in tests.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from inspector.agents.base import BaseAgent
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import ReaderOutput
from inspector.schemas.mmad import Question
from inspector.schemas.state import InspectionState

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an industrial quality inspector. Image 1 is the part to inspect. Image 2 is a normal "
    "(defect-free) reference of the same product. Image 3, if present, is a zoomed crop of the region "
    "that differs most from the reference. Answer the multiple-choice question."
)


def build_prompt(q: Question, evidence_text: str | None, critique: str | None = None) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(q.options.items()))
    parts = [SYSTEM_PROMPT]
    if evidence_text:
        parts.append(evidence_text)
    if critique:
        parts.append(f"A reviewer flagged a possible inconsistency in your previous answers: {critique} Reconsider carefully.")
    parts.append(f"Question: {q.text}\nOptions:\n{opts}")
    parts.append(
        'Reply with ONLY a JSON object: {"selected_option": "<letter>", "confidence": <0..1>, '
        '"brief_rationale": "<one short sentence>"}'
    )
    return "\n\n".join(parts)


def parse_reader_json(text: str, q: Question) -> ReaderOutput | None:
    """Tolerant parser: first JSON object in ``text``; returns None if invalid / option not offered."""
    for m in _JSON_RE.finditer(text):
        try:
            data = json.loads(m.group(0))
            data["question_id"] = q.id
            data["selected_option"] = str(data.get("selected_option", "")).strip().upper()[:1]
            out = ReaderOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError, AttributeError, TypeError):
            continue
        if out.selected_option in q.options:
            return out
    return None


class ReaderMLLM(BaseAgent):
    def __init__(self, settings: Settings, model_id: str, name: str, reopen: bool = False):
        self.s = settings
        self.model_id = model_id
        self.name = name
        self.reopen = reopen
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None

    # ---- which dict of the state this reader owns ---------------------------
    def _store(self, state: InspectionState) -> dict[str, ReaderOutput | None]:
        return state.reader_a_responses if "reader_a" in self.name else state.reader_b_responses

    def applies(self, state: InspectionState) -> bool:
        if not self.reopen:
            return True
        return bool(state.critic_verdict and state.critic_verdict.reopen_questions)

    # ---- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        if self.s.reader_backend == "vllm":
            raise NotImplementedError("vLLM backend is a planned extension; use reader_backend='hf' for now.")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        device = self.s.device if (self.s.device != "cuda" or torch.cuda.is_available()) else "cpu"
        kwargs: dict[str, Any] = {"device_map": device if device != "cpu" else None}
        if self.s.load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, self.s.dtype))
        else:
            kwargs["torch_dtype"] = getattr(torch, self.s.dtype) if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs).eval()

    def unload(self) -> None:
        self.model = None
        self.processor = None
        super().unload()

    # ---- generation backend (HF transformers) --------------------------------
    def _generate(self, image_paths: list[str], prompt: str) -> tuple[str, int, int]:
        """Return (text, tokens_in, tokens_out)."""
        content = [{"type": "image", "path": p} for p in image_paths] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.s.max_new_tokens, do_sample=False)
        new = out[0, n_in:]
        return self.processor.decode(new, skip_special_tokens=True), n_in, int(new.shape[0])

    # ---- agent ---------------------------------------------------------------
    def ask(self, q: Question, state: InspectionState, critique: str | None, rec: dict[str, Any]) -> ReaderOutput | None:
        ev = state.visual_evidence
        images = [state.query_image_path, state.reference_image_path] + ([ev.crop_path] if ev else [])
        prompt = build_prompt(q, ev.as_text() if ev else None, critique)
        calls = rec.setdefault("calls", [])
        for _ in range(1 + self.s.json_retries):
            text, n_in, n_out = self._generate(images, prompt)
            calls.append({"question_id": q.id, "prompt": prompt, "raw": text, "tokens_in": n_in, "tokens_out": n_out})
            rec["tokens_in"] = rec.get("tokens_in", 0) + n_in
            rec["tokens_out"] = rec.get("tokens_out", 0) + n_out
            out = parse_reader_json(text, q)
            if out:
                return out
        return None  # invalid after retry => counted as an error

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        store = self._store(state)
        if self.reopen:
            ids = set(state.critic_verdict.reopen_questions) if state.critic_verdict else set()
            critique = state.critic_verdict.critique_reason if state.critic_verdict else None
            todo = [q for q in state.questions if q.id in ids]
        else:
            critique, todo = None, state.questions
        for q in todo:
            store[q.id] = self.ask(q, state, critique, rec)
        rec["answers"] = {q.id: (store[q.id].selected_option if store.get(q.id) else None) for q in todo}
        return state
