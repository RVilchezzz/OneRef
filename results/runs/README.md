# Run outputs (snapshot of `outputs/`, Nov 10–11, 2026)

`outputs/` is gitignored, so this folder is a shareable copy of every run made on Peter's laptop.
The full story and the numbers are in [`docs/reports/2026-11-11_first_end_to_end_run.md`](../../docs/reports/2026-11-11_first_end_to_end_run.md).

- **Logs and answers** (`*.json`, `*.jsonl`): copied **unchanged**.
- **Images** (heatmaps, crops, gallery): re-saved as **JPEG, max side 800 px** to keep the repo light
  (189 MB → 10 MB). The JSON files still name the original `.png` files under `outputs/`; the copy here has
  the same name with `.jpg`.

| Folder | What it is |
|---|---|
| `expert_eval/dev_dinov3-vitb16_v1/` | Visual expert alone, DINOv3 ViT-B/16, 835 dev images: `per_image.jsonl`, `summary.json`, `gallery/` |
| `expert_eval/dev_dinov2-base_v1/` | Same with DINOv2-base |
| `expert_eval/smoke_dinov2/` | First 20-image smoke test of `scripts/eval_expert.py` |
| `smoke_ollama_qwen4b_ps2/` | Run 1 of the full pipeline: stopped in the reader pass (context overflow, see report §4) |
| `smoke_ollama_qwen4b_ps2_v2/` | Run 2: 9 dev images, full pipeline, Qwen3-VL-4B via Ollama |
| `smoke_ollama_qwen4b_ps4/` | Run 3: 18 dev images, same configuration |

Inside a pipeline run:

| Path | Content |
|---|---|
| `telemetry.jsonl` | One line per agent call and per pass: prompts, raw model replies, tokens, timings |
| `1_expert/` … `7_aggregator/` | The full state of each image after each pass (`InspectionState` as JSON); `7_aggregator/` is the final one |
| `expert_artifacts/` | The expert's heatmap and the zoomed crop sent to the readers, per image |

Per-question tables (question, options, truth, each reader's answer before/after reopen, final answer) are in
`results/pipeline_smoke/<run>/per_question.csv`, with `summary.json` next to them.

Images derive from MMAD (MVTec AD and MVTec LOCO: CC BY-NC-SA 4.0; VisA: CC BY 4.0; GoodsAD: research use).
