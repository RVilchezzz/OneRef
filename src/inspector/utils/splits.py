"""Stratified dev/test split (fixed seed). MMAD is test-only, so we carve out a dev set for prompts."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from inspector.schemas.mmad import Sample


def make_split(samples: list[Sample], dev_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Stratify by (source, category, is_defective); every stratum with >= 2 images gets >= 1 dev image."""
    strata: dict[tuple, list[Sample]] = defaultdict(list)
    for s in samples:
        strata[(s.source, s.category, s.is_defective)].append(s)
    rng = random.Random(seed)
    dev: list[Sample] = []
    test: list[Sample] = []
    for key in sorted(strata, key=str):
        group = sorted(strata[key], key=lambda s: s.image_id)
        rng.shuffle(group)
        k = round(len(group) * dev_fraction)
        k = max(1, k) if len(group) >= 2 else 0
        dev += group[:k]
        test += group[k:]
    return dev, test


def sample_per_source(samples: list[Sample], per_source: int, seed: int) -> list[Sample]:
    """Small balanced subset for quick runs: ``per_source`` images per source, half normal, half defective."""
    rng = random.Random(seed)
    by: dict[tuple, list[Sample]] = defaultdict(list)
    for s in samples:
        by[(s.source, s.is_defective)].append(s)
    picked: list[Sample] = []
    for src in sorted({s.source for s in samples}):
        for defective in (False, True):
            group = sorted(by[(src, defective)], key=lambda s: s.image_id)
            rng.shuffle(group)
            picked += group[: max(1, per_source // 2)]
    return picked


def save_split(dev: list[Sample], test: list[Sample], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev_ids.json").write_text(json.dumps([s.image_id for s in dev]), encoding="utf-8")
    (out_dir / "test_ids.json").write_text(json.dumps([s.image_id for s in test]), encoding="utf-8")
