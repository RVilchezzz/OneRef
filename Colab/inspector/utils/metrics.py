"""Evaluation: per-subtask accuracy, balanced accuracy, FPR / miss rate, McNemar, bootstrap CI.

A *row* is one scored question: ``{image_id, subtask, correct, is_defective, source}``.
Unanswered questions (``pred is None``) are rows with ``correct=False`` (they stay in the denominator).
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable

from scipy.stats import binomtest

from inspector.schemas.mmad import Sample, SubTask
from inspector.schemas.state import InspectionState

Row = dict


def build_rows(samples: Iterable[Sample], states: Iterable[InspectionState]) -> list[Row]:
    by_id = {s.image_id: s for s in samples}
    rows: list[Row] = []
    for st in states:
        sample = by_id[st.image_id]
        for q in sample.questions:
            fa = st.final_answers.get(q.id)
            pred = fa.selected_option if fa else None
            rows.append({
                "image_id": st.image_id, "question_id": q.id, "subtask": q.subtask,
                "correct": pred is not None and pred == q.answer,
                "is_defective": sample.is_defective, "source": sample.source,
            })
    return rows


def per_subtask_accuracy(rows: list[Row]) -> dict[SubTask, float]:
    """Accuracy per subtask; discrimination is *balanced* (mean of normal and defective accuracy)."""
    acc: dict[SubTask, float] = {}
    for st in SubTask:
        sub = [r for r in rows if r["subtask"] == st]
        if not sub:
            continue
        if st == SubTask.ANOMALY_DISCRIMINATION:
            acc[st] = balanced_accuracy(sub)
        else:
            acc[st] = sum(r["correct"] for r in sub) / len(sub)
    return acc


def mean_accuracy(rows: list[Row]) -> float:
    """Official metric: simple mean over the subtasks present."""
    acc = per_subtask_accuracy(rows)
    return sum(acc.values()) / len(acc) if acc else float("nan")


def balanced_accuracy(disc_rows: list[Row]) -> float:
    groups = [[r["correct"] for r in disc_rows if r["is_defective"] == d] for d in (False, True)]
    rates = [sum(g) / len(g) for g in groups if g]
    return sum(rates) / len(rates) if rates else float("nan")


def fpr_and_miss(disc_rows: list[Row]) -> tuple[float, float]:
    """(false-positive rate on normal parts, missed-defect rate on defective parts)."""
    normal = [r for r in disc_rows if not r["is_defective"]]
    bad = [r for r in disc_rows if r["is_defective"]]
    fpr = sum(not r["correct"] for r in normal) / len(normal) if normal else float("nan")
    miss = sum(not r["correct"] for r in bad) / len(bad) if bad else float("nan")
    return fpr, miss


def mcnemar_exact(rows_a: list[Row], rows_b: list[Row]) -> tuple[int, int, float]:
    """Exact (binomial) two-sided McNemar on paired questions -> (a_only, b_only, p_value)."""
    ca = {r["question_id"]: r["correct"] for r in rows_a}
    cb = {r["question_id"]: r["correct"] for r in rows_b}
    ids = ca.keys() & cb.keys()
    a_only = sum(ca[i] and not cb[i] for i in ids)
    b_only = sum(cb[i] and not ca[i] for i in ids)
    n = a_only + b_only
    return a_only, b_only, 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)


def bootstrap_ci(rows: list[Row], stat: Callable[[list[Row]], float] = mean_accuracy,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Image-level percentile bootstrap (resample images, not questions: questions are correlated)."""
    by_img: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_img[r["image_id"]].append(r)
    ids, rng = list(by_img), random.Random(seed)
    stats = sorted(
        stat([r for i in (rng.choice(ids) for _ in ids) for r in by_img[i]]) for _ in range(n_boot)
    )
    return stats[int(alpha / 2 * n_boot)], stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
