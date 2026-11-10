"""Deterministic GPU-free stand-ins for smoke runs and tests. Never read ground truth.

Readers and critic are the REAL pydantic-ai agents (prompt, output schema, validators, tool),
driven by a ``FunctionModel`` instead of a served MLLM.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from inspector.agents.base import BaseAgent
from inspector.agents.critic_verifier import CriticVerifier
from inspector.agents.reader_mllm import ReaderMLLM
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import VisualExpertOutput
from inspector.schemas.state import InspectionState

_OPTION_RE = re.compile(r"^([A-D])\. ", re.MULTILINE)


def _h(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest(), 16)


def _user_text(messages: list[ModelMessage]) -> str:
    """Text of the first user prompt (the images are BinaryContent items, skipped)."""
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    return p.content if isinstance(p.content, str) else " ".join(c for c in p.content if isinstance(c, str))
    return ""


def reader_model(seed: str) -> FunctionModel:
    """Answers a pseudo-random option that the question offers."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = _user_text(messages)
        letters = sorted(set(_OPTION_RE.findall(text))) or ["A"]
        letter = letters[_h(seed, text) % len(letters)]
        return ModelResponse(parts=[TextPart(json.dumps({"selected_option": letter, "confidence": 0.5, "brief_rationale": "mock"}))])

    return FunctionModel(fn)


def critic_model() -> FunctionModel:
    """Calls ``run_consistency_rules`` first, then approves (no extra reopen)."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        called = any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts)
        if not called:
            return ModelResponse(parts=[ToolCallPart(tool_name="run_consistency_rules", args={})])
        out = {"is_valid": True, "conflict_detected": False, "reopen_questions": [], "critique_reason": "mock review"}
        return ModelResponse(parts=[TextPart(json.dumps(out))])

    return FunctionModel(fn)


class MockExpert(BaseAgent):
    name = "visual_expert"

    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        h = _h(state.image_id)
        cells = ["top left", "top center", "top right", "middle left", "center", "middle right", "bottom left", "bottom center", "bottom right"]
        state.visual_evidence = VisualExpertOutput(
            heatmap_path="mock_heatmap.png", candidate_region=cells[h % 9],  # type: ignore[arg-type]
            confidence_score=(h % 100) / 100, crop_path="mock_crop.png",
        )
        return state


class MockReader(ReaderMLLM):
    def __init__(self, settings: Settings, name: str, reopen: bool = False):
        super().__init__(settings, model_name="mock", name=name, reopen=reopen, model=reader_model(name))

    def image_parts(self, state: InspectionState) -> list:
        return []  # mock runs have no image files


class MockCritic(CriticVerifier):
    def __init__(self, settings: Settings):
        super().__init__(settings, model=critic_model())

    def image_parts(self, state: InspectionState) -> list:
        return []
