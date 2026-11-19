# Results

Every number in the reports and slides comes from a file in this folder. Everything here is on **dev**
(835 images, 3,937 questions) unless the folder name says otherwise; the test has not been run yet.
`outputs/` (live runs) is gitignored; the shareable copies of the runs are in `runs/`.

**Start here:** [`compare/2026-11-19_dev_frozen/summary.md`](compare/2026-11-19_dev_frozen/summary.md), the dev
table of every system with the frozen aggregation settings (S 66.5, B1 65.9, B3 65.6, B2 60.2).

## `compare/`: comparison tables (`summary.md` to read, `summary.json` for scripts)

| Folder | Question it answers | Answer (dev) |
|---|---|---|
| `2026-11-16_hint/` | Does the expert's hint help the reader? (Qwen3-VL-4B, 82 images) | Yes: 68.0 vs 61.2, mostly localization |
| `2026-11-16_smoke_critic/` | Gemma 4 E2B or E4B as verifier? (9 images) | E2B: as good, 40 % faster |
| `2026-11-17_dev_full_S/` | Full S on all dev, and its ablations: `S/`, `ablations/`, `verifier/` (Gemma vs rules vs none, reopen stats), `discrimination*/` and `calibration_C/` (expert vote sweep), `judge/` | Judge +2.1 (p = 0.004); verifier +0.2 (n.s.) |
| `2026-11-18_dev_B3/` | Iso-compute vote (Qwen × 3 samples) vs S | B3 65.6, S vs B3 p = 0.14 |
| `2026-11-18_official_check/` | Does MMAD's official `summary.py` agree with our metric? | Within 0.2 points (S 66.6 official) |
| `2026-11-19_dev_frozen/` | All systems with the frozen config | The reference dev table |

## Other folders

| Folder | Content |
|---|---|
| `runs/` | Shareable run states (see [`runs/README.md`](runs/README.md)): `2026-11-17_dev_full_S/` = the full dev run, as `states.tar.xz` + logs; older smoke runs and the expert evaluation |
| `expert_eval/` | The DINOv3 / DINOv2 expert alone: AUROC per source, peak-in-mask rate (`summary.json`, `per_image.jsonl`) |
| `error_analysis/dev_S_frozen/` | Where each of the 1,316 dev failures of S was lost (`traces.md`) and the 100-case sheet to label by hand (`sheet.csv`, `sheet.html`) |
| `figures/dev/` | The five report/slide figures on dev (`scripts/make_figures.py`) |
| `pipeline_smoke/` | Per-question tables of the first end-to-end smoke runs (Nov 11) |

After the test, `scripts/run_test.sh` writes `test/frozen_2026-11-19/`, `error_analysis/test_S/` and `figures/test/`.
