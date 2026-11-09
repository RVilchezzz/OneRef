"""Base class for all agents.

Lifecycle (driven by ``orchestrator.pass_runner``): ``load()`` once per pass, ``run()`` for
every image of the split, ``unload()`` to free the GPU before the next pass. An agent never
keeps weights after ``unload()``, which is what keeps peak VRAM at "one model at a time".
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import Any

from inspector.schemas.state import InspectionState
from inspector.utils.telemetry import Telemetry


def free_gpu() -> None:
    """Release Python garbage and cached CUDA blocks (call after dropping model refs)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - torch is optional for CPU-only / mock runs
        pass


class BaseAgent(ABC):
    name: str = "agent"

    def load(self) -> None:
        """Load weights. Default: nothing to load (rule-based agents)."""

    def unload(self) -> None:
        """Drop weights and free VRAM. Subclasses must null their model refs first."""
        free_gpu()

    def applies(self, state: InspectionState) -> bool:
        """False => the runner skips this image for this pass (and may skip loading the model)."""
        return True

    def run(self, state: InspectionState, tel: Telemetry) -> InspectionState:
        """Run on one image inside a telemetry span; returns the updated state."""
        with tel.span(self.name, state.image_id) as rec:
            state = self._run(state, rec)
        return state

    @abstractmethod
    def _run(self, state: InspectionState, rec: dict[str, Any]) -> InspectionState:
        """Do the work; put prompts / raw outputs / token counts in ``rec``."""
