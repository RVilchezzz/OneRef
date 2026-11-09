"""Shared workflow state, one instance per image, passed through every pass."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from inspector.schemas.agent_outputs import CriticOutput, FinalAnswer, ReaderOutput, VisualExpertOutput
from inspector.schemas.mmad import Question, Sample


class InspectionState(BaseModel):
    image_id: str
    query_image_path: str
    reference_image_path: str            # == random_templates[0]
    questions: list[Question]

    visual_evidence: VisualExpertOutput | None = None
    reader_a_responses: dict[str, ReaderOutput | None] = Field(default_factory=dict)  # None = invalid
    reader_b_responses: dict[str, ReaderOutput | None] = Field(default_factory=dict)
    critic_verdict: CriticOutput | None = None
    final_answers: dict[str, FinalAnswer] = Field(default_factory=dict)
    telemetry_logs: list[dict[str, Any]] = Field(default_factory=list)

    reopen_count: int = 0                # guard: never exceeds settings.max_reopen

    @classmethod
    def from_sample(cls, s: Sample, image_root: str = "") -> "InspectionState":
        from pathlib import Path

        root = Path(image_root) if image_root else None
        resolve = (lambda p: str(root / p)) if root else (lambda p: p)
        return cls(
            image_id=s.image_id,
            query_image_path=resolve(s.image_path),
            reference_image_path=resolve(s.reference),
            questions=s.questions,
        )
