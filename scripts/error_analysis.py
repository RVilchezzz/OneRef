"""Error analysis of a finished run: where each wrong answer was lost, and a 100-case sheet to label by hand.

Every wrong final answer is traced through the workflow with the cached passes:
- the answers of readers A and B **before** the verifier (``--pre-step``, default ``3_reader_b``);
- their answers after the reopen, the judge's answer and the final vote (``--final-step``).

Each failure gets an **automatic trace category**. It is a hint for the human label, not the label:

| Trace | Meaning | Suggested label |
|---|---|---|
| ``lost_at_reopen`` | a reader was right before the verifier and wrong after its reopen | MAST orchestration |
| ``lost_at_judge`` | after the reopens a voter was right, the judge overrode it with a wrong answer | MAST orchestration |
| ``lost_at_vote`` | a voter was right after the reopens, the vote picked another answer | MAST orchestration |
| ``nobody_right`` | no agent ever gave the right answer | perception or reasoning (or annotation) |
| ``invalid`` | no valid final answer (format errors after the retry) | MAST orchestration |

Outputs in ``--out``:
- ``traces.md`` / ``traces.json``: trace categories over **all** failures, per source and per subtask;
- ``sheet.csv``: ``--n`` failures (stratified by source × subtask, fixed seed), with the empty columns
  ``label`` (perception / reasoning / orchestration / annotation) and ``note`` to fill in;
- ``sheet.html``: the same cases with the query, the reference, the expert heatmap and crop, every answer.

Usage (dev, frozen S):
    PYTHONPATH=src .venv/bin/python scripts/error_analysis.py outputs/dev_full_S_judge \
        --final-dir outputs/reagg/dev_full_S_judge/override_C2/7_aggregator --out results/error_analysis/dev_S
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from inspector.config.settings import get_settings
from inspector.schemas.mmad import load_mmad
from inspector.schemas.state import InspectionState

SUGGESTED = {"lost_at_reopen": "orchestration", "lost_at_judge": "orchestration", "lost_at_vote": "orchestration",
             "invalid": "orchestration", "nobody_right": "perception/reasoning"}


def load_states(folder: Path) -> dict[str, InspectionState]:
    states = {}
    for p in sorted(folder.glob("*.json")):
        st = InspectionState.model_validate_json(p.read_text(encoding="utf-8"))
        states[st.image_id] = st
    if not states:
        raise SystemExit(f"no cached states in {folder}")
    return states


def letter(d: dict, qid: str) -> str:
    r = d.get(qid)
    return r.selected_option if r else ""


def trace(q, pre: InspectionState | None, fin: InspectionState) -> dict:
    truth = q.answer
    a0, b0 = (letter(pre.reader_a_responses, q.id), letter(pre.reader_b_responses, q.id)) if pre else ("", "")
    a1, b1 = letter(fin.reader_a_responses, q.id), letter(fin.reader_b_responses, q.id)
    c1, j = letter(fin.reader_c_responses, q.id), letter(fin.judge_answers, q.id)
    fa = fin.final_answers.get(q.id)
    final = fa.selected_option if fa and fa.selected_option else ""
    if not final:
        cat = "invalid"
    elif truth in (a1, b1, c1):
        cat = "lost_at_judge" if j and j == final else "lost_at_vote"
    elif truth in (a0, b0):
        cat = "lost_at_reopen"
    else:
        cat = "nobody_right"
    reopened = bool(fin.critic_verdict and q.id in fin.critic_verdict.reopen_questions)
    return {"truth": truth, "final": final, "a_pre": a0, "b_pre": b0, "a_post": a1, "b_post": b1, "c": c1,
            "judge": j, "reopened": reopened, "trace": cat,
            "votes": json.dumps(fa.votes if fa else {}), "rationale_a": rationale(fin.reader_a_responses, q.id),
            "rationale_b": rationale(fin.reader_b_responses, q.id), "rationale_judge": rationale(fin.judge_answers, q.id)}


def rationale(d: dict, qid: str) -> str:
    r = d.get(qid)
    return r.brief_rationale if r else ""


def mask_file(p: Path) -> str:
    """MVTec LOCO stores a folder of masks per image: show its first file."""
    if p.is_dir():
        files = sorted(x for x in p.iterdir() if x.is_file())
        return str(files[0]) if files else ""
    return str(p)


def stratified_pick(rows: list[dict], n: int, seed: int) -> list[dict]:
    """``n`` rows, proportional to (source, subtask), largest remainder; seeded shuffle inside each stratum."""
    if n >= len(rows):
        return rows
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[(r["source"], r["subtask"])].append(r)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / len(rows) for k in keys}
    take = {k: int(quota[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k] - take[k]), k))[: n - sum(take.values())]:
        take[k] += 1
    rng = random.Random(seed)
    out = []
    for k in keys:
        group = sorted(strata[k], key=lambda r: r["question_id"])
        rng.shuffle(group)
        out += group[: take[k]]
    return out


def table(counter_by: dict[str, Counter], cats: list[str]) -> list[str]:
    lines = ["| Group | Failures | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
    for g, c in counter_by.items():
        tot = sum(c.values())
        lines.append(f"| {g} | {tot} | " + " | ".join(f"{c[k]} ({100 * c[k] / tot:.0f} %)" for k in cats) + " |")
    return lines


def write_html(cases: list[dict], out: Path) -> None:
    def img(path: str, label: str) -> str:
        if not path:
            return ""
        rel = os.path.relpath(path, out)
        return (f'<figure><a href="{html.escape(rel)}"><img src="{html.escape(rel)}" loading="lazy"></a>'
                f"<figcaption>{label}</figcaption></figure>")

    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Error analysis</title><style>",
             "body{font-family:sans-serif;margin:1.5em;max-width:1300px}"
             ".case{border-top:2px solid #888;padding:1em 0}.imgs{display:flex;gap:8px;flex-wrap:wrap}"
             "figure{margin:0}img{height:220px;border:1px solid #ccc}figcaption{font-size:12px;color:#555}"
             "table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:left}"
             ".ok{background:#d9f2d9}.bad{background:#f8d7d7}.tag{font-weight:bold}",
             "</style></head><body><h1>Error analysis: labelling sheet</h1>",
             "<p>Label each case in <code>sheet.csv</code> (column <code>label</code>): perception / reasoning / "
             "orchestration / annotation. The trace is computed automatically and only suggests the label.</p>"]
    for i, c in enumerate(cases, 1):
        cell = lambda x: f"<td class='{'ok' if x and x == c['truth'] else 'bad' if x else ''}'>{x or '—'}</td>"
        opts = "<br>".join(f"<b>{k}</b>. {html.escape(v)}" for k, v in json.loads(c["options"]).items())
        parts.append(
            f"<div class='case'><h3>#{i} · {html.escape(c['source'])} · {html.escape(c['subtask'])} · "
            f"<span class='tag'>{c['trace']}</span> · {'defective' if c['is_defective'] else 'normal'} part · "
            f"expert {c['expert_score']}</h3>"
            f"<p><code>{html.escape(c['question_id'])}</code><br><b>{html.escape(c['question'])}</b><br>{opts}</p>"
            "<table><tr><th>Truth</th><th>A before</th><th>B before</th><th>A after</th><th>B after</th>"
            "<th>Judge</th><th>Final</th><th>Reopened</th></tr><tr>"
            f"<td>{c['truth']}</td>{cell(c['a_pre'])}{cell(c['b_pre'])}{cell(c['a_post'])}{cell(c['b_post'])}"
            f"{cell(c['judge'])}{cell(c['final'])}<td>{'yes' if c['reopened'] else 'no'}</td></tr></table>"
            f"<p><b>A:</b> {html.escape(c['rationale_a'])}<br><b>B:</b> {html.escape(c['rationale_b'])}<br>"
            f"<b>Judge:</b> {html.escape(c['rationale_judge'])}</p><div class='imgs'>"
            + img(c["query_path"], "query") + img(c["reference_path"], "reference (random_templates[0])")
            + img(c["heatmap_path"], "expert heatmap") + img(c["crop_path"], "crop shown to the readers")
            + img(c["mask_path"], "mask (evaluation only)") + "</div></div>")
    parts.append("</body></html>")
    (out / "sheet.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, help="run with the cached passes (e.g. outputs/test_S)")
    ap.add_argument("--pre-step", default="3_reader_b", help="pass with the answers before the verifier")
    ap.add_argument("--final-step", default="7_aggregator")
    ap.add_argument("--final-dir", type=Path, default=None, help="folder of final states (e.g. a re-aggregation)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    s = get_settings()
    by_id = {x.image_id: x for x in load_mmad(s.mmad_json)}
    root = Path(s.mmad_root)
    final = load_states(args.final_dir or args.run_dir / args.final_step)
    pre_dir = args.run_dir / args.pre_step
    pre = load_states(pre_dir) if pre_dir.is_dir() else {}

    fails, n_questions = [], 0
    for iid, fin in final.items():
        sample = by_id[iid]
        cat_dir = str(Path(sample.image_path).parent.parent.parent)  # mask_path is relative to the category folder
        for q in sample.questions:
            n_questions += 1
            t = trace(q, pre.get(iid), fin)
            if t["final"] == q.answer:
                continue
            ev = fin.visual_evidence
            fails.append({
                "question_id": q.id, "source": sample.source, "category": sample.category,
                "subtask": q.subtask.value, "is_defective": sample.is_defective, "question": q.text,
                "options": json.dumps(q.options, ensure_ascii=False), **t,
                "expert_score": f"{ev.confidence_score:.2f}" if ev else "", "expert_region": str(ev.candidate_region) if ev else "",
                "query_path": fin.query_image_path, "reference_path": fin.reference_image_path,
                "heatmap_path": ev.heatmap_path if ev else "", "crop_path": ev.crop_path if ev else "",
                "mask_path": mask_file(root / cat_dir / sample.mask_path) if sample.mask_path else "",
                "suggested_label": SUGGESTED[t["trace"]], "label": "", "note": "",
            })

    args.out.mkdir(parents=True, exist_ok=True)
    cats = ["lost_at_reopen", "lost_at_judge", "lost_at_vote", "invalid", "nobody_right"]
    groups: dict[str, Counter] = {"**all**": Counter(f["trace"] for f in fails)}
    for key in ("source", "subtask"):
        for g in sorted({f[key] for f in fails}):
            groups[g] = Counter(f["trace"] for f in fails if f[key] == g)
    recoverable = sum(f["trace"] != "nobody_right" for f in fails)
    md = [f"# Error traces: `{args.final_dir or args.run_dir}`", "",
          f"{len(fails)} wrong answers out of {n_questions} questions ({100 * len(fails) / n_questions:.1f} %).",
          f"In **{recoverable} ({100 * recoverable / len(fails):.0f} %)** of them, some agent had the right answer "
          "at some point: an upper bound on what better orchestration could recover.", "",
          *table(groups, cats), ""]
    (args.out / "traces.md").write_text("\n".join(md), encoding="utf-8")
    (args.out / "traces.json").write_text(json.dumps({g: dict(c) for g, c in groups.items()}, indent=2), encoding="utf-8")

    cases = stratified_pick(fails, args.n, args.seed)
    with (args.out / "sheet.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cases[0]))
        w.writeheader()
        w.writerows(cases)
    write_html(cases, args.out)
    print("\n".join(md))
    print(f"sheet: {len(cases)} cases -> {args.out}/sheet.csv, sheet.html")


if __name__ == "__main__":
    main()
