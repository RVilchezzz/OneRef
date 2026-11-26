# OneRef: multi-agent visual inspection from a single reference (IA025, Unicamp 2026)

Final project of **IA025 – Introduction to Deep Learning** (Unicamp 2026 S2). Team: Peter Roque Vilchez ·
Charles Maxime Doudy. Deliveries: Oct 29 (plan) · Nov 12 (preliminary) · **Nov 26 (final)**.

A **training-free** multi-agent system for industrial visual inspection. It gets **one photo of the part and
one normal reference photo**, and answers the multiple-choice questions of the **MMAD** benchmark (ICLR 2025:
8,366 images, 39,670 questions, 7 subtasks, 4 sources). No fine-tuning, no added text, no tuning on the test.

**Question studied:** how much of the multi-agent gain reported by SiGMA (85.2 %) survives under a *strict*
protocol (1 reference, no added text, no tuning on MMAD)? **Success:** the full system S beats, with McNemar
p < 0.05, both the best single reader with the same evidence (B1 = Qwen + expert) and an equal-cost majority vote (B3).
A well-measured negative result is also valid.

## Status and results (Nov 25)

**Answer:** under the strict protocol, the team of agents does **not** beat one reader that gets the same evidence.
What helps is the visual expert's evidence. Test, 200 stratified images (952 questions), run once with the config
frozen on Nov 19 ([`results/test/test200_2026-11-25/summary.md`](results/test/test200_2026-11-25/summary.md)):

| System (4B, local) | Mean of 7 subtasks [95 % CI] | McNemar vs S |
|---|---|---|
| **S**: expert + Qwen + InternVL + verifier + judge | **70.1** [66.9, 73.2] | — |
| B1, Qwen + expert (hint and crop) | 70.3 [67.0, 73.5] | 44 vs 44 questions, p = 1.0 |
| S without the judge | 68.7 [65.0, 72.1] | p = 0.47 |
| Qwen alone, no expert | 64.5 [61.5, 67.6] | p < 0.001 |
| InternVL + expert | 62.6 [59.3, 65.9] | p < 0.001 |

- The expert's evidence adds +5.8 points to Qwen (p < 0.001); the team adds nothing on top (−0.2, p = 1.0).
- S makes fewer false alarms (48 % vs 53 % of good parts) with the same missed defects (18 % vs 17 %).
- Dev (835 images) showed the same picture: S 66.5, Qwen + expert 65.9 (p = 0.28), B3 65.6
  ([`results/compare/2026-11-19_dev_frozen/summary.md`](results/compare/2026-11-19_dev_frozen/summary.md)).
- B3 (equal-compute vote) on the test is still running; the 2,000-image test is planned for the article.
  The protocol change was written before any test answer: [`docs/reports/2026-11-25_test_protocol_change.md`](docs/reports/2026-11-25_test_protocol_change.md).

| | |
|---|---|
| Frozen config | [`configs/2026-11-19_frozen.env`](configs/2026-11-19_frozen.env) |
| Test driver | [`scripts/run_test200.sh`](scripts/run_test200.sh) (200 images) · [`scripts/run_test.sh`](scripts/run_test.sh) (2,000) |
| Code | 50 GPU-free tests pass; `Colab/` is an identical copy |
| 8B version | Not part of the IA025 results; template [`configs/2026-11-19_8B_colab_template.env`](configs/2026-11-19_8B_colab_template.env) |

## How it works

A fixed Python workflow, not a free-form agent. Each step is a **pass** over the whole split: it caches every
image's state to disk, frees the GPU, then the next pass runs. Only one model is on the GPU at a time, and a
crashed run resumes from the last saved image.

```
image + reference (random_templates[0])
  → 1_expert            DINOv3 ViT-B/16, frozen: anomaly map, 3×3 region, zoomed crop, score (no LLM)
  → 2_reader_a          Qwen3-VL (4B local / 8B Colab): every question, with the expert's hint
  → 3_reader_b          InternVL3.5 (4B / 8B): the same questions, another model family
  → 4_critic            consistency rules + Gemma 4 LLM review; flags conflicts
  → 5_/6_ reopen        each reader re-answers only its own flagged questions, once max
  → 6b_judge            Gemma 4: only questions where A and B disagree, picks one of their answers
  → 7_aggregator        weighted vote; the expert votes on discrimination (per-source weights, fixed on dev)
```

Agents are [pydantic-ai](https://ai.pydantic.dev) agents on an OpenAI-compatible endpoint (Ollama locally,
vLLM on Colab) and exchange Pydantic-validated JSON: one retry on a bad format, then the question counts as an
error. Every call is logged per image in `outputs/<run>/telemetry.jsonl`.

## Quick start

```bash
uv sync                                              # or: python -m venv .venv && pip install -e .
PYTHONPATH=src .venv/bin/python -m pytest -q tests   # 50 GPU-free tests
PYTHONPATH=src .venv/bin/python -m inspector.main --split dev --limit 5 --mock --run-name mock01
```

A real run on the laptop (4B, Ollama; the dev images come from `scripts/fetch_mmad_subset.py --split dev`):

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve              # in another terminal
set -a; source configs/2026-11-19_frozen.env; set +a
PYTHONPATH=src .venv/bin/python -m inspector.main --split dev --per-source 2 --run-name try01
PYTHONPATH=src .venv/bin/python scripts/compare_runs.py S=outputs/try01
```

`--split test` is refused without `--frozen-config`. Never tune on test.

## Repository map

| Path | Content |
|---|---|
| [`src/inspector/`](src/inspector/) | The package: `main.py` (CLI), `config/settings.py` (every `ONEREF_*` setting), `schemas/`, `agents/` (expert, readers, critic, judge, aggregator, pydantic-ai plumbing, mocks), `orchestrator/` (pass list + pass runner with per-image cache), `utils/` (metrics, splits, telemetry) |
| [`tests/`](tests/) | `test_workflow.py` (GPU-free, mock models) and `test_llm_connection.py` (live endpoint check) |
| [`configs/`](configs/) | One `.env` per run, dated. **`2026-11-19_frozen.env`** = the 4B system S for the test; `2026-11-19_8B_colab_template.env` = the 8B starting point |
| [`scripts/`](scripts/) | Everything outside the pipeline (table below) |
| [`data/splits/`](data/splits/) | Dev and test image ids (`dev_ids.json`, `test_ids.json`, `test_sample2000_ids.json`). `data/MMAD/` (images) is gitignored |
| [`results/`](results/) | Every number of the reports: comparison tables, run snapshots, error analysis, figures. Index: [`results/README.md`](results/README.md) |
| [`docs/`](docs/) | `full_plan.md` (authoritative plan), `executive_summary.md`, `references.md`, `reports/` (dated reports + `final_report.md` skeleton), `slides/` (one folder per delivery), `guides/`, `mmad/` (leaderboard snapshot), `course/` |
| [`Colab/`](Colab/) | Self-contained copy of `src/inspector/` + `OneRef_Colab.ipynb` (vLLM, outputs on Drive). Re-sync after any code change (`Colab/README.md`) |
| [`Notebook/`](Notebook/) | `OneRef.ipynb`: the whole package in one notebook |
| `papers/` | PDFs (gitignored, `scripts/download_papers.sh`) |
| `outputs/` | Live runs: per-pass cache, telemetry, logs (gitignored) |

### Scripts

| Script | Does | GPU |
|---|---|---|
| `fetch_mmad_subset.py` | Downloads only the images of a split (HTTP range reads in the HF zips) | no |
| `download_mmad.py` · `download_papers.sh` | Full or sample MMAD download · papers | no |
| `eval_expert.py` | Scores the expert alone: AUROC per source, peak in mask | yes |
| `compare_runs.py` | Comparison table: subtasks, mean + bootstrap CI, FPR, misses, per source, McNemar, oracle | no |
| `reaggregate.py` | Re-applies the aggregator with other settings (B1, B2, no judge…) from a run's cache | no |
| `sweep_discrimination.py` | Grid of the expert's vote threshold × weight, `--by-source` (dev only) | no |
| `reopen_stats.py` | Per reader: reopened, fixed and broken answers after the verifier | no |
| `official_summary.py` | MMAD's official `summary.py` metric, next to ours | no |
| `error_analysis.py` | Where each failure was lost + the 100-case sheet to label | no |
| `make_figures.py` | The five figures of the report and slides | no |
| **`run_test.sh`** | **The single test run**, unattended: S → re-aggregations → B3 → tables → official metric → errors → figures | yes |
| `ablation_dev_full_S.sh` · `quick_reader_baseline.py` | Record of how the Nov 17 dev ablations / the Nov 10 first reader test were made | yes |

## Guides

| Guide | For |
|---|---|
| [`docs/guides/guide_8B.fr.md`](docs/guides/guide_8B.fr.md) | **Running S with 8B readers on Colab** (FR): what changes, steps, protocol, known traps |
| [`docs/guides/guide_backends.md`](docs/guides/guide_backends.md) | Every model backend: `.env`, hosted APIs (Gemini, Groq, xAI, HF router, report as upper bound), vLLM, Ollama |
| [`docs/guides/guide_base.md`](docs/guides/guide_base.md) · [`guide_notebook.md`](docs/guides/guide_notebook.md) · [`guide_colab.md`](docs/guides/guide_colab.md) | Step-by-step (FR): command line, single notebook, Colab |
| [`docs/guides/guide_pedagogique.fr.md`](docs/guides/guide_pedagogique.fr.md) | Teaching guide to the project idea (FR, early version: the subtask list there is not MMAD's) |

## Strict protocol (do not break)

- Reference = only `random_templates[0]` of each `mmad.json` entry: no `similar_templates`, no `domain_knowledge.json`.
- Dev = 10 % stratified (835 images, fixed seed) for prompts and calibration. Test = 2,000 stratified images,
  run **once** with a frozen config.
- Never use `--available-only` for a real run: with part of the images the split changes.
- Discrimination is scored with balanced accuracy; FPR and missed-defect rate are reported too. Statistics:
  exact McNemar and a 1,000-resample image-level bootstrap 95 % CI.
- No model trained on MMAD's sources (data leakage).

MMAD data licenses are non-commercial (MVTec AD and MVTec LOCO: CC BY-NC-SA 4.0; VisA: CC BY 4.0; GoodsAD:
research use).
