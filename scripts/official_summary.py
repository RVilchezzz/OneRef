"""Score runs with MMAD's official metric (``evaluation/examples/helper/summary.py``), next to ours.

The official script (jam-cc/MMAD, ``caculate_accuracy_mmad``) differs from ``inspector.utils.metrics``:

1. **Per-dataset mean.** DS-MVTec and MVTec-AD are merged into one "MVTec-AD" dataset. The script computes
   the accuracy per subtask inside each of the 4 datasets, averages the subtasks per dataset, and then
   averages the 4 datasets. Ours pools all questions before averaging the subtasks.
2. **Invalid answers are dropped.** A question whose answer is not a letter A–E leaves the denominator.
   Ours counts it as wrong. The official loop also calls ``list.remove`` while iterating, which skips the
   next question. That is reproduced here, so the numbers match the official script exactly.
3. **Normal vs defective** comes from ``'good' in image path``. Ours uses "has a mask". Both give the
   same labels on MMAD; the check below counts any mismatch.
4. Discrimination is balanced accuracy, per dataset, in both.

For each run this script writes ``answers_<label>.json`` in MMAD's answer format, so the official
script can be run on it unchanged, and a table with three numbers:
- the official metric;
- the official metric with invalid answers counted as wrong (our strict rule, per dataset);
- our pooled metric (as in ``compare_runs.py``).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/official_summary.py S=outputs/dev_full_S B1=... \
        --out results/compare/2026-11-18_official_check
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from inspector.config.settings import get_settings
from inspector.schemas.mmad import load_mmad
from inspector.schemas.state import InspectionState
from inspector.utils import metrics

LETTERS = {"A", "B", "C", "D", "E"}
OBJECT_ANALYSIS = {"Object Structure", "Object Details"}


def dataset_of(image: str) -> str:
    name = image.split("/")[0]
    return "MVTec-AD" if name in ("DS-MVTec", "MVTec-AD") else name


def export_answers(run_dir: Path, step: str, mmad: dict, key_of: dict) -> tuple[list[dict], list[InspectionState]]:
    """MMAD answer records (image, question, question_type, correct_answer, gpt_answer) of a run."""
    states = [InspectionState.model_validate_json(p.read_text(encoding="utf-8"))
              for p in sorted((run_dir / step).glob("*.json"))]
    if not states:
        raise SystemExit(f"no cached states in {run_dir / step}")
    out = []
    for st in states:
        key = key_of[st.image_id]
        for i, q in enumerate(mmad[key]["conversation"]):
            fa = st.final_answers.get(f"{st.image_id}#{i}")
            out.append({"image": key, "question": q["Question"], "question_type": q["type"],
                        "correct_answer": str(q["Answer"]).strip().upper()[:1],
                        "gpt_answer": fa.selected_option if fa else ""})
    return out, states


def official_metric(answers: list[dict], drop_invalid: bool = True, normal_flag: str = "good") -> dict:
    """Faithful re-implementation of ``caculate_accuracy_mmad`` (without pandas), plus a strict variant."""
    answers = list(answers)
    datasets, types = [], []
    for a in answers:
        d, t = dataset_of(a["image"]), a["question_type"]
        t = "Object Analysis" if t in OBJECT_ANALYSIS else t
        datasets += [d] if d not in datasets else []
        types += [t] if t not in types else []
    q = {d: {t: [0, 0] for t in types} for d in datasets}          # [total, correct]
    det = {d: {"normal": [0, 0], "abnormal": [0, 0]} for d in datasets}
    removed = 0
    for a in answers:  # iterating while removing, exactly like the official loop (skips the next item)
        d, t = dataset_of(a["image"]), a["question_type"]
        t = "Object Analysis" if t in OBJECT_ANALYSIS else t
        ok = a["correct_answer"] == a["gpt_answer"]
        if drop_invalid and (a["correct_answer"] not in LETTERS or a["gpt_answer"] not in LETTERS):
            answers.remove(a)
            removed += 1
            continue
        q[d][t][0] += 1
        q[d][t][1] += ok
        if t == "Anomaly Detection":
            g = det[d]["normal" if normal_flag in a["image"] else "abnormal"]
            g[0] += 1
            g[1] += ok
    table = {}
    for d in datasets:
        row = {t: 100 * q[d][t][1] / q[d][t][0] if q[d][t][0] else 0.0 for t in types}
        if "Anomaly Detection" in row:
            n, b = det[d]["normal"], det[d]["abnormal"]
            row["Anomaly Detection"] = 100 * ((n[1] / n[0] if n[0] else 0) + (b[1] / b[0] if b[0] else 0)) / 2
            row["Overkill"] = 100 * (1 - n[1] / n[0]) if n[0] else 0.0
            row["Miss"] = 100 * (1 - b[1] / b[0]) if b[0] else 0.0
        row["Average"] = sum(row[t] for t in types) / len(types)
        table[d] = row
    return {"per_dataset": table, "average": sum(r["Average"] for r in table.values()) / len(table),
            "removed": removed, "types": types}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--step", default="7_aggregator")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s = get_settings()
    mmad = json.loads(Path(s.mmad_json).read_text(encoding="utf-8"))
    key_of = {"/".join(Path(k).parts[-4:]).rsplit(".", 1)[0]: k for k in mmad}
    by_id = {x.image_id: x for x in load_mmad(s.mmad_json)}
    args.out.mkdir(parents=True, exist_ok=True)

    results, lines = {}, []
    for spec in args.runs:
        label, path = spec.split("=", 1)
        answers, states = export_answers(Path(path), args.step, mmad, key_of)
        (args.out / f"answers_{label}.json").write_text(json.dumps(answers), encoding="utf-8")
        off = official_metric(answers)
        strict = official_metric(answers, drop_invalid=False)
        ours = metrics.mean_accuracy(metrics.build_rows([by_id[x.image_id] for x in states], states))
        mismatch = sum((("good" in by_id[x.image_id].image_path) == by_id[x.image_id].is_defective) for x in states)
        invalid = sum(a["gpt_answer"] not in LETTERS for a in answers)
        results[label] = {"official": off, "official_invalid_as_wrong": strict, "ours_pooled": 100 * ours,
                          "n_questions": len(answers), "invalid_answers": invalid, "label_mismatch_images": mismatch}
        lines.append((label, off, strict, 100 * ours, len(answers), invalid, mismatch))

    md = ["| Run | Questions | Invalid | Official (drops invalid) | Official, invalid = wrong | Ours (pooled) | Label mismatches |",
          "|---|---|---|---|---|---|---|"]
    for label, off, strict, ours, n, inv, mm in lines:
        md.append(f"| {label} | {n} | {inv} | **{off['average']:.1f}** | {strict['average']:.1f} | {ours:.1f} | {mm} |")
    for label, off, strict, *_ in lines:
        cols = off["types"] + ["Average", "Overkill", "Miss"]
        md += ["", f"Official table, {label} (per dataset; DS-MVTec + MVTec-AD merged):", "",
               "| Dataset | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for d, row in off["per_dataset"].items():
            md.append(f"| {d} | " + " | ".join(f"{row.get(c, float('nan')):.1f}" for c in cols) + " |")
    text = "\n".join(md) + "\n"
    print(text)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
