# Full system S on dev at the 4B scale (Nov 16–17, 2026)

835 dev images, 3,937 questions, run locally (RTX 3060 Laptop 6 GB, Ollama). Config:
`configs/2026-11-17_small_models.env`, plus the judge (`ONEREF_JUDGE_MODEL=gemma4:e2b`) for `dev_full_S_judge`.
The frozen test config built from these results is `configs/2026-11-19_frozen.env`.
Scores and McNemar tables: `results/compare/2026-11-17_dev_full_S/`.

| File | Content |
|---|---|
| `states.tar.xz` | Per-image `InspectionState` JSON after each pass + `telemetry.jsonl` (prompts, raw replies, tokens, timings) |
| `logs/` | Console logs of the runs and of `scripts/ablation_dev_full_S.sh` |
| `reagg_variants/` | The Settings overrides of every CPU re-aggregation variant (`scripts/reaggregate.py`) |

Unpack into `outputs/` to re-aggregate or re-score without the GPU:

```bash
tar -xJf results/runs/2026-11-17_dev_full_S/states.tar.xz -C outputs
PYTHONPATH=src:scripts .venv/bin/python scripts/compare_runs.py S=outputs/dev_full_S judge=outputs/dev_full_S_judge
```

The archive holds:
- `dev_full_S/`: every pass, 1_expert … 7_aggregator.
- `dev_full_S_judge/` and `dev_full_S_rules/`: only the passes that differ from `dev_full_S`. Passes 1–6 of the judge run
  and 1–3 of the rules run are identical copies, so to resume them copy those folders from `dev_full_S/` first.

Not included: `expert_artifacts/` (heatmaps and crops, 4.1 GB of PNG). Pass 1 regenerates them in about 20 minutes.

Runtime on the laptop: expert 20 min · reader A 3 h 53 · reader B 3 h 43 · critic 22 min · reopens 1 h 26 ·
judge 18 min. That is about 10 h for 835 images, so a 2,000-image test takes about 24 h.
