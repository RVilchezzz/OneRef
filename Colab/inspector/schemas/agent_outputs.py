"""Typed messages exchanged between agents. Every inter-agent payload is validated here."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from inspector.schemas.mmad import OptionLetter

# 3x3 grid cell names, row-major.
GridCell = Literal[
    "top left", "top center", "top right",
    "middle left", "center", "middle right",
    "bottom left", "bottom center", "bottom right",
]


class VisualExpertOutput(BaseModel):
    """Evidence from the DINOv3 expert: where the query differs from the reference."""

    heatmap_path: str
    candidate_region: GridCell
    confidence_score: float = Field(ge=0.0, le=1.0)  # normalized anomaly peak
    crop_path: str                                   # zoomed crop around the peak
    crop_base64: str | None = None                   # optional inline form of the crop

    def as_text(self) -> str:
        """Text injected into the readers' prompt."""
        level = "high" if self.confidence_score >= 0.66 else "medium" if self.confidence_score >= 0.33 else "low"
        return (
            f"Visual expert (automatic patch comparison with the reference): "
            f"most different region = {self.candidate_region}, "
            f"difference level = {level} ({self.confidence_score:.2f}). "
            "This is a hint, not ground truth."
        )


class ReaderAnswer(BaseModel):
    """What the MLLM itself returns (the pydantic-ai ``output_type`` of a reader)."""

    model_config = ConfigDict(extra="ignore")

    selected_option: OptionLetter
    confidence: float = Field(ge=0.0, le=1.0)
    brief_rationale: str = ""


class ReaderOutput(ReaderAnswer):
    """One MLLM answer to one question, tagged with the question it answers."""

    question_id: str


class CriticOutput(BaseModel):
    is_valid: bool                       # schema/format checks passed
    conflict_detected: bool              # logical inconsistency found
    reopen_questions: list[str] = Field(default_factory=list)  # question ids
    critique_reason: str = ""


class FinalAnswer(BaseModel):
    question_id: str
    selected_option: OptionLetter | None  # None = no valid answer (counts as an error)
    votes: dict[str, float] = Field(default_factory=dict)  # option -> weight


class FinalReport(BaseModel):
    """Per-image inspection report."""

    image_id: str
    answers: list[FinalAnswer]
    expert: VisualExpertOutput | None = None
    critic: CriticOutput | None = None
    reopened: bool = False
