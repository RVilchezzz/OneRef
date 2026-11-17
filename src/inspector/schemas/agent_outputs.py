"""Typed messages exchanged between agents. Every inter-agent payload is validated here."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def unwrap_schema_echo(cls, data: Any) -> Any:
        """Format normalization, content unchanged: some small models (InternVL3.5-4B GGUF in Ollama)
        echo the JSON-schema layout, ``{"properties": {"selected_option": ...}, "required": [...]}``.
        If the answer sits under ``properties`` and not at the top level, use it."""
        if isinstance(data, dict) and "selected_option" not in data:
            inner = data.get("properties")
            if isinstance(inner, dict) and "selected_option" in inner:
                return inner
        return data


class ReaderOutput(ReaderAnswer):
    """One MLLM answer to one question, tagged with the question it answers."""

    question_id: str


class CriticReview(BaseModel):
    """What the LLM critic itself returns (its pydantic-ai ``output_type``): the 4 fields it fills.
    Kept apart from ``CriticOutput`` so internal fields never enter the schema the model must follow."""

    model_config = ConfigDict(extra="ignore")

    is_valid: bool
    conflict_detected: bool
    reopen_questions: list[str] = Field(default_factory=list)
    critique_reason: str = ""


class CriticOutput(BaseModel):
    """The critic's verdict stored in the state (rules + LLM review)."""

    is_valid: bool                       # schema/format checks passed
    conflict_detected: bool              # logical inconsistency found
    reopen_questions: list[str] = Field(default_factory=list)  # question ids (union over readers)
    critique_reason: str = ""
    # Targeted reopen: each reader re-answers only its own flagged questions, with its own reasons.
    # Empty dict = verdict from an older cache -> readers fall back to reopen_questions.
    reopen_by_reader: dict[str, list[str]] = Field(default_factory=dict)
    reason_by_reader: dict[str, str] = Field(default_factory=dict)


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
