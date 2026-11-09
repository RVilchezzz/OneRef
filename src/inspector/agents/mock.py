"""Deterministic GPU-free stand-ins for smoke runs and tests. Never read ground truth."""
from __future__ import annotations

import hashlib
from typing import Any

from inspector.agents.base import BaseAgent
from inspector.agents.reader_mllm import ReaderMLLM
from inspector.config.settings import Settings
from inspector.schemas.agent_outputs import VisualExpertOutput
from inspector.schemas.state import InspectionState


def _h(*parts: str) -> int:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest(), 16)


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
    """Real prompt/parse/retry logic, fake generation: answers a pseudo-random valid option."""

    def __init__(self, settings: Settings, name: str, reopen: bool = False):
        super().__init__(settings, model_id="mock", name=name, reopen=reopen)

    def load(self) -> None:
        pass

    def _generate(self, image_paths: list[str], prompt: str) -> tuple[str, int, int]:
        letters = sorted(l for l in "ABCD" if f"\n{l}. " in prompt)
        letter = letters[_h(self.name, prompt) % len(letters)]
        return f'{{"selected_option": "{letter}", "confidence": 0.5, "brief_rationale": "mock"}}', len(prompt) // 4, 20
