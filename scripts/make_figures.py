"""Figures for the final report and slides, from the JSON that ``compare_runs.py`` and ``error_analysis.py`` write.

No model calls and no GPU: it only reads result files, so it can be re-run at any time.

Figures (PNG, 200 dpi, sized for a 16:9 slide), referenced in ``docs/slides/final/slides_content.md``:
1. ``means.png``: mean of the 7 subtasks with its 95 % CI for the main systems. External references are dashed
   lines. With ``--test``, the dev bars are drawn in grey next to the test bars.
2. ``tradeoff.png``: false-positive rate vs missed-defect rate, one point per system.
3. ``sources.png``: heatmap of the mean per source × system.
4. ``ablations.png``: S and its ablations, with the McNemar p-value against S.
5. ``errors.png``: where the failures of S come from (trace categories), overall and per source.

Usage (dev; on test, pass ``--test results/test/frozen_2026-11-19/summary.json`` and the test traces):
    .venv/bin/python scripts/make_figures.py --dev results/compare/2026-11-19_dev_frozen/summary.json \
        --errors results/error_analysis/dev_S_frozen/traces.json --skip-source MVTec-AD \
        --out results/figures/dev
Needs matplotlib (``pip install -e .[figures]``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette of the slides (docs/slides/README.md).
NAVY, CREAM, RUST, GREY, LIGHT, PAPER = "#15253A", "#F6F4EF", "#B04A17", "#5B6673", "#C9C2B4", "#FFFDF8"
SOFT_RUST = "#E8925C"

LABELS = {
    "S": "S (full system)", "B1_qwen": "B1 Qwen3-VL-4B", "B2_internvl": "B2 InternVL3.5-4B",
    "B3_k3": "B3 Qwen × 3 (iso-compute)", "AB_noverif": "A + B vote, no verifier",
    "S_nojudge": "S without judge", "S_noexpertvote": "S without expert vote",
}
MAIN = ["B2_internvl", "B3_k3", "B1_qwen", "S"]
ABLATIONS = ["AB_noverif", "S_nojudge", "S_noexpertvote", "S"]
# External references, strict 1-shot MMAD (Jiang et al., ICLR 2025; leaderboard Oct 23, 2026).
EXTERNAL = [("Qwen3-VL-8B", 73.91), ("GPT-4o", 74.9), ("Human experts", 86.65)]
ERRORS = [("lost_at_judge", "Lost at the judge", RUST), ("lost_at_reopen", "Lost at the reopen", SOFT_RUST),
          ("nobody_right", "No agent ever right", NAVY)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GREY, "axes.labelcolor": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM, "text.color": NAVY,
})


def load(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def label(name: str, wrap: bool = False) -> str:
    text = LABELS.get(name, name)
    return text.replace(" ", "\n", 1) if wrap else text


def color(name: str) -> str:
    return RUST if name == "S" else NAVY


def p_vs_s(summary: dict, name: str) -> float | None:
    pair = summary["pairs"].get(f"S vs {name}")
    return pair["mcnemar_p"] if pair else None


def fmt_p(p: float | None) -> str:
    if p is None:
        return ""
    return "p < 0.001" if p < 1e-3 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}"


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}")


def fig_means(dev: dict, test: dict | None, systems: list[str], out: Path) -> None:
    main = test or dev
    systems = [s for s in systems if s in main["runs"]]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    width = 0.38 if test else 0.6
    for i, name in enumerate(systems):
        bars = [(dev, i - width / 2, LIGHT, "dev"), (test, i + width / 2, color(name), "test")] if test \
            else [(dev, i, color(name), "dev")]
        for summ, x, c, split in bars:
            r = summ["runs"].get(name)
            if r is None:
                continue
            m, (lo, hi) = 100 * r["mean"], (100 * v for v in r["ci95"])
            ax.bar(x, m, width, color=c, zorder=2)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=GREY, capsize=5, lw=1.5, zorder=3)
            ax.text(x, hi + 0.6, f"{m:.1f}", ha="center", va="bottom", fontsize=13,
                    fontweight="bold" if summ is main else "normal", color=NAVY)
        p = p_vs_s(main, name)
        if p is not None:
            ax.text(i + width / 2 if test else i, 51, fmt_p(p), ha="center", va="bottom", fontsize=10 if test else 11,
                    color=CREAM, zorder=4)
    for k, (txt, v) in enumerate(EXTERNAL):
        ax.axhline(v, ls="--", lw=1.2, color=GREY, zorder=1)
        below = k + 1 < len(EXTERNAL) and EXTERNAL[k + 1][1] - v < 2  # next line too close: write under this one
        ax.text(len(systems) - 0.5, v - 0.3 if below else v + 0.3, f"{txt} {v:.1f}", ha="right",
                va="top" if below else "bottom", fontsize=11, color=GREY)
    ax.set_xticks(range(len(systems)), [label(s, wrap=True) for s in systems])
    ax.set_ylim(50, 90)
    ax.set_ylabel("Mean of the 7 subtasks (%)")
    ax.grid(axis="y", color=LIGHT, lw=0.6, zorder=0)
    title = "Test (frozen config), dev in grey" if test else "Dev (835 images, frozen aggregation)"
    ax.set_title(f"{title} · bars: 95 % bootstrap CI · white text: McNemar vs S", fontsize=12, color=GREY, loc="left")
    save(fig, out, "means.png")


def fig_tradeoff(summ: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    pts = {n: (100 * r["fpr"], 100 * r["miss"]) for n, r in summ["runs"].items()
           if n != "S_noexpertvote"}  # identical to S once the judge is on; it would overprint the S point
    for name, (x, y) in pts.items():
        ax.scatter(x, y, s=140, color=color(name), zorder=3)
        # Put the label on the side away from the closest other point.
        others = [(ox, oy) for n, (ox, oy) in pts.items() if n != name]
        ox, oy = min(others, key=lambda o: (o[0] - x) ** 2 + (o[1] - y) ** 2) if others else (x - 1, y)
        xs = [px for px, _ in pts.values()]
        near_left = x - min(xs) < 0.15 * (max(xs) - min(xs))
        dx = 10 if near_left or ox < x else -10
        dy = -14 if oy > y else 8
        ax.annotate(label(name), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=12,
                    ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom")
    ax.margins(x=0.15, y=0.1)
    ax.set_xlabel("False-positive rate: good parts called defective (%)")
    ax.set_ylabel("Missed defects (%)")
    ax.grid(color=LIGHT, lw=0.6, zorder=0)
    ax.set_title("Anomaly discrimination: lower-left is better", fontsize=12, color=GREY, loc="left")
    save(fig, out, "tradeoff.png")


def fig_sources(summ: dict, systems: list[str], skip: set[str], out: Path) -> None:
    systems = [s for s in systems if s in summ["runs"]]
    sources = [s for s in summ["runs"][systems[0]]["mean_per_source"] if s not in skip]
    grid = [[100 * summ["runs"][n]["mean_per_source"][src] for n in systems] for src in sources]
    fig, ax = plt.subplots(figsize=(11, 1.1 * len(sources) + 1.6))
    im = ax.imshow(grid, cmap="Oranges", vmin=50, vmax=85, aspect="auto")
    for i, row in enumerate(grid):
        best = max(row)
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=14,
                    fontweight="bold" if v == best else "normal", color=NAVY if v < 75 else CREAM)
    ax.set_xticks(range(len(systems)), [label(s, wrap=True) for s in systems])
    ax.set_yticks(range(len(sources)), sources)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean of subtasks (%)")
    ax.set_title("Mean per source (best of each row in bold)", fontsize=12, color=GREY, loc="left")
    save(fig, out, "sources.png")


def fig_ablations(summ: dict, out: Path) -> None:
    names = [n for n in ABLATIONS if n in summ["runs"]]
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(names) + 1.4))
    for i, name in enumerate(names):
        r = summ["runs"][name]
        m, (lo, hi) = 100 * r["mean"], (100 * v for v in r["ci95"])
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", ms=11, color=color(name), capsize=5, lw=1.5)
        ax.text(hi + 0.25, i, f"{m:.1f}   {fmt_p(p_vs_s(summ, name))}", va="center", fontsize=12)
    ax.set_yticks(range(len(names)), [label(n) for n in names])
    ax.invert_yaxis()
    lo_all = min(100 * summ["runs"][n]["ci95"][0] for n in names)
    hi_all = max(100 * summ["runs"][n]["ci95"][1] for n in names)
    ax.set_xlim(lo_all - 1, hi_all + 4)
    ax.set_xlabel("Mean of the 7 subtasks (%), 95 % CI")
    ax.grid(axis="x", color=LIGHT, lw=0.6)
    ax.set_title("Ablations of S (same cache, no extra calls) · McNemar vs S", fontsize=12, color=GREY, loc="left")
    save(fig, out, "ablations.png")


def fig_errors(traces: dict, skip: set[str], out: Path) -> None:
    groups = ["**all**"] + [g for g in traces if g in {"DS-MVTec", "MVTec-AD", "MVTec-LOCO", "VisA", "GoodsAD"}
                            and g not in skip]
    fig, ax = plt.subplots(figsize=(12, 0.75 * len(groups) + 1.6))
    for i, g in enumerate(groups):
        counts = traces[g]
        total = sum(counts.values())
        left = 0.0
        for key, _, c in ERRORS:
            share = 100 * counts.get(key, 0) / total if total else 0
            ax.barh(i, share, left=left, color=c, height=0.62)
            if share >= 5:
                ax.text(left + share / 2, i, f"{share:.0f} %", ha="center", va="center", color=CREAM, fontsize=12)
            left += share
        ax.text(101, i, f"n = {total}", va="center", fontsize=11, color=GREY)
    ax.set_yticks(range(len(groups)), ["All failures" if g == "**all**" else g for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Share of the questions S gets wrong (%)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in ERRORS], labels=[t for _, t, _ in ERRORS],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    ax.set_title("Where the failures come from: was the right answer ever on the table?", fontsize=12, color=GREY,
                 loc="left")
    save(fig, out, "errors.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev", type=Path, required=True, help="dev summary.json (compare_runs.py), run labels as in MAIN")
    ap.add_argument("--test", type=Path, help="test summary.json; when given, the figures show test, dev in grey")
    ap.add_argument("--errors", type=Path, help="traces.json (error_analysis.py) of the split shown")
    ap.add_argument("--skip-source", action="append", default=[], help="source to leave out (too few images)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dev, test = load(args.dev), load(args.test)
    main_summ = test or dev
    skip = set(args.skip_source)
    args.out.mkdir(parents=True, exist_ok=True)
    fig_means(dev, test, MAIN, args.out)
    fig_tradeoff(main_summ, args.out)
    fig_sources(main_summ, MAIN, skip, args.out)
    fig_ablations(main_summ, args.out)
    if args.errors:
        fig_errors(load(args.errors), skip, args.out)


if __name__ == "__main__":
    main()
