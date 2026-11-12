"""Pydantic models for ``mmad.json`` (MMAD benchmark, strict 1-shot protocol).

Strict protocol: the only reference image ever exposed is ``random_templates[0]``.
``similar_templates`` is deliberately NOT modelled so it cannot leak into a prompt.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OptionLetter = Literal["A", "B", "C", "D"]


class SubTask(str, Enum):
    """The 7 MMAD subtasks (Object Structure/Details are merged into object_analysis)."""

    ANOMALY_DISCRIMINATION = "anomaly_discrimination"
    DEFECT_CLASSIFICATION = "defect_classification"
    DEFECT_LOCALIZATION = "defect_localization"
    DEFECT_DESCRIPTION = "defect_description"
    DEFECT_ANALYSIS = "defect_analysis"
    OBJECT_CLASSIFICATION = "object_classification"
    OBJECT_ANALYSIS = "object_analysis"


# Raw MMAD "type" strings -> SubTask (matched case-insensitively, spaces/underscores ignored).
_RAW_TYPES: dict[str, SubTask] = {
    "anomalydetection": SubTask.ANOMALY_DISCRIMINATION,  # the name used in mmad.json
    "anomalydiscrimination": SubTask.ANOMALY_DISCRIMINATION,
    "defectclassification": SubTask.DEFECT_CLASSIFICATION,
    "defectlocalization": SubTask.DEFECT_LOCALIZATION,
    "defectdescription": SubTask.DEFECT_DESCRIPTION,
    "defectanalysis": SubTask.DEFECT_ANALYSIS,
    "objectclassification": SubTask.OBJECT_CLASSIFICATION,
    "objectanalysis": SubTask.OBJECT_ANALYSIS,
    "objectstructure": SubTask.OBJECT_ANALYSIS,
    "objectdetails": SubTask.OBJECT_ANALYSIS,
}


def parse_subtask(raw: str) -> SubTask:
    key = raw.replace(" ", "").replace("_", "").lower()
    try:
        return _RAW_TYPES[key]
    except KeyError as e:  # unknown type must fail loudly, never be silently scored
        raise ValueError(f"Unknown MMAD subtask type: {raw!r}") from e


class Question(BaseModel):
    """One multiple-choice question. ``options`` maps letter -> text (2 or 4 entries)."""

    model_config = ConfigDict(frozen=True)

    id: str                                   # "<image_id>#<index>"
    text: str
    options: dict[OptionLetter, str] = Field(min_length=2, max_length=4)
    answer: OptionLetter | None = None        # ground truth, used ONLY by metrics
    subtask: SubTask

    @model_validator(mode="after")
    def _answer_in_options(self) -> "Question":
        if self.answer is not None and self.answer not in self.options:
            raise ValueError(f"answer {self.answer!r} not in options {list(self.options)}")
        return self


class Sample(BaseModel):
    """One MMAD image with its questions and its single normal reference."""

    image_id: str
    image_path: str
    mask_path: str | None = None              # evaluation only, never given to agents
    source: str = "unknown"                   # MVTec / VisA / MVTec LOCO / GoodsAD
    category: str = "unknown"
    is_defective: bool = False
    templates: list[str] = Field(min_length=1)  # random_templates, order preserved
    questions: list[Question]

    @property
    def reference(self) -> str:
        """Strict 1-shot: always ``random_templates[0]``."""
        return self.templates[0]


def _parse_options(raw: dict | list) -> dict[str, str]:
    """MMAD stores options as {"A": "...", ...}; tolerate a plain list too."""
    if isinstance(raw, dict):
        return {str(k).strip().upper()[:1]: str(v) for k, v in raw.items()}
    return {"ABCD"[i]: str(v) for i, v in enumerate(raw)}


def load_mmad(path: str | Path, limit: int | None = None) -> list[Sample]:
    """Load ``mmad.json`` (dict keyed by image path) into validated ``Sample`` objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.items() if isinstance(data, dict) else ((d["image"], d) for d in data)
    samples: list[Sample] = []
    for image_path, entry in items:
        parts = Path(image_path).parts
        image_id = "/".join(parts[-4:]).rsplit(".", 1)[0]
        mask = entry.get("mask_path")
        questions = [
            Question(
                id=f"{image_id}#{i}",
                text=q["Question"],
                options=_parse_options(q["Options"]),
                answer=str(q["Answer"]).strip().upper()[:1] if q.get("Answer") else None,
                subtask=parse_subtask(q["type"]),
            )
            for i, q in enumerate(entry["conversation"])
        ]
        samples.append(
            Sample(
                image_id=image_id,
                image_path=image_path,
                mask_path=mask,
                source=parts[0] if parts else "unknown",
                category=parts[1] if len(parts) > 1 else "unknown",
                is_defective=bool(mask),
                templates=list(entry["random_templates"]),
                questions=questions,
            )
        )
        if limit and len(samples) >= limit:
            break
    return samples
