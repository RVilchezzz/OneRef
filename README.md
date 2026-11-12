# OneRef: multi-agent visual inspection from a single reference (IA025, Unicamp 2026)

Final project for **IA025 – Deep Learning**. A **training-free** multi-agent system for industrial visual inspection: it gets **one photo of the part and one normal reference photo**, and answers the multiple-choice questions of the **MMAD** benchmark (8,366 images, 39,670 questions, 7 subtasks). No fine-tuning, no extra text, no tuning on the test split.

**Question studied:** how much of the multi-agent gain reported by SiGMA (85.2 %) survives under a *strict* protocol (1 reference, no added text, no tuning on MMAD)? **Success:** the full system beats, with a McNemar test (p < 0.05), both the best single model (Qwen3-VL-8B, 73.91 %) and a majority vote of equal cost. A well-measured negative result is also valid.

## How it works

A fixed Python workflow (not a free-form agent):

```
image + reference ──► Visual expert (DINOv3: anomaly map, 3×3 region, zoomed crop, score)
                  ──► Reader A (Qwen3-VL-8B)  ┐
                  ──► Reader B (InternVL3-8B) ┴─► Critic (may reopen a conflict once) ─► Aggregator (weighted vote)
```

- Agents exchange **Pydantic-validated JSON**; one retry on bad format, then the question counts as an error.
- **Pass-based execution**: each agent runs over the whole split, caches its outputs to disk, then the next one runs. Runs are **resumable**.
- Everything is logged as JSONL per image (`outputs/<run>/telemetry.jsonl`).
- Readers and critic are [pydantic-ai](https://ai.pydantic.dev) agents on an **OpenAI-compatible endpoint** (hosted API or local vLLM). Only the DINOv3 expert runs locally.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and **Python ≥ 3.11**.
2. Install the dependencies (creates `.venv` from `pyproject.toml` / `uv.lock`; torch + transformers need a few GB of disk):
   ```bash
   uv sync
   ```
3. Create a `.env` file at the project root (it is gitignored, never commit it):

   | Variable | Role |
   |---|---|
   | `HF_TOKEN` | Hugging Face token: DINOv3 weights (accept Meta's licence on the model page) and the MMAD download |
   | `ONEREF_LLM_BASE_URL` | OpenAI-compatible endpoint (default: HF router `https://router.huggingface.co/v1`) |
   | `ONEREF_LLM_API_KEY` | key for that endpoint (falls back to `HF_TOKEN`) |
   | `ONEREF_READER_A_MODEL`, `ONEREF_READER_B_MODEL`, `ONEREF_CRITIC_MODEL` | model ids |
   | `ONEREF_DEVICE` | `cuda` or `cpu` (DINOv3 expert) |

   Free example with Gemini (OpenAI-compatible endpoint, key from <https://aistudio.google.com/apikey>):
   ```
   ONEREF_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   ONEREF_LLM_API_KEY=<gemini key>
   ONEREF_READER_A_MODEL=gemini-3.8-flash
   ONEREF_READER_B_MODEL=gemini-3.5-flash-lite
   ONEREF_CRITIC_MODEL=gemini-3.8-flash
   ```
   The HF router bills Inference Providers on prepaid credits (error 402 without credits). Check which model ids it serves with `GET <base_url>/models`.

   **Backup on quota exhaustion:** with a hosted `ONEREF_USE_GROK` (Groq) and the Gemini variables above (`ONEREF_LLM_*`), any API error from Groq (429 quota, 5xx...) makes the same call retry on the Gemini model of that role (pydantic-ai `FallbackModel`). Disable with `ONEREF_BACKUP_ENABLED=false`. Extra keys rotate the same way: `ONEREF_GROK_API_KEYS=k2,k3,...` (Groq) and `ONEREF_LLM_API_KEYS=k2,k3,...` (Gemini); order: Groq keys, then Gemini keys.

   **Grok / Groq instead** (the `GROK` variables work for any OpenAI-compatible host; a key starting with `gsk_` is **Groq**, use `ONEREF_GROK_BASE_URL=https://api.groq.com/openai/v1` and a model such as `qwen/qwen3.8-27b`; xAI Grok keys use `https://api.x.ai/v1`): set `ONEREF_USE_GROK=true`, `ONEREF_GROK_API_KEY=<key>` (or `XAI_API_KEY`) and the model ids `ONEREF_GROK_READER_A_MODEL`, `ONEREF_GROK_READER_B_MODEL`, `ONEREF_GROK_CRITIC_MODEL` (placeholders `grok-4`: check vision-capable ids on <https://docs.x.ai>). It fills `ONEREF_REMOTE` for the three roles, so no vLLM is started. Hosted: report as upper bound.

   **Local vLLM instead** (protocol-faithful ≤ 8B models, Linux/Colab, `pip install vllm`): `ONEREF_LLM_PROVIDER=vllm`, `ONEREF_LLM_BASE_URL=http://localhost:8000/v1`, `ONEREF_LLM_API_KEY=EMPTY`, and `ONEREF_VLLM_AUTOSTART=true` to start/stop one `vllm serve` per pass. See `Colab/README.md`.

> Hosted models larger or different from the ≤ 8B local models leave the strict protocol: report those results as an **upper bound**, not as the strict-protocol result.

## vLLM on Colab (the original plan: Qwen3-VL-8B and InternVL3-8B, no hosted API)

Downloading the model weights from Hugging Face is free; only the HF *router* is billed. Running the models yourself on a Colab GPU costs no Hugging Face credits, only Colab compute units.

**How it works.** `vllm serve <model>` loads one model on the GPU and opens an OpenAI-compatible HTTP server (`http://localhost:8000/v1`). The agents call it exactly like any hosted endpoint, so no code changes. With `ONEREF_VLLM_AUTOSTART=true`, each pass of the pipeline starts its own `vllm serve` for its model and stops it at the end of the pass, so only one 8B model is on the GPU at a time (pass 1: expert DINOv3, then reader A, reader B, critic).

**Steps** (details in [`Colab/README.md`](Colab/README.md)):
1. Copy the [`Colab/`](Colab/) folder to Google Drive (`MyDrive/OneRef/Colab`) and MMAD to `MyDrive/OneRef/MMAD` (or use the optional download cell 5b).
2. In Colab, pick a **GPU runtime: L4 (recommended, bf16)**. On a T4 (16 GB, no bf16), use AWQ-quantized checkpoints with `--dtype half`; the checkpoint ids in the notebook are not verified.
3. Open [`Colab/OneRef_Colab.ipynb`](Colab/OneRef_Colab.ipynb) and run the cells in order: GPU check, Drive mount, `pip install` (vLLM may replace torch: restart the runtime if Colab asks), `HF_TOKEN` as a Colab secret (DINOv3 licence), environment variables, `--mock` smoke test, 50-image probe, full dev run.
4. Outputs go to Drive (`outputs/`), so an interrupted run resumes where it stopped.

**With `Notebook/OneRef.ipynb` instead**, which defaults to the hosted router, add these variables to its configuration cell and run `!pip install vllm`:
```
ONEREF_LLM_PROVIDER=vllm
ONEREF_LLM_BASE_URL=http://localhost:8000/v1
ONEREF_LLM_API_KEY=EMPTY
ONEREF_VLLM_AUTOSTART=true
```
`ONEREF_VLLM_AUTOSTART=false` means you start `vllm serve <model>` yourself and the pass only checks that the server answers with the right model.

**Locally** (RTX 3060, 6 GB): vLLM does not run on Windows and 6 GB is too small for an 8B model in bf16. Use an OpenAI-compatible local server such as Ollama or LM Studio with a small quantized model (e.g. Qwen3-VL-4B 4-bit) for debugging: set `ONEREF_LLM_BASE_URL` to its URL (Ollama: `http://localhost:11434/v1`), the model name in `ONEREF_READER_A_MODEL`, and keep `ONEREF_VLLM_AUTOSTART=false`.

> Never run on a real Colab yet: the vLLM install, GPU memory use and T4 checkpoint ids are unverified. The `Colab/` folder is a copy of `src/inspector/`; re-sync it after code changes (see `Colab/README.md`).

## Tests

```bash
uv run pytest                                   # GPU-free suite (mock agents, no network)
uv run pytest tests/test_llm_connection.py -s   # live check of the LLM endpoint (needs .env; skipped without a key)
```

The live test sends a text prompt and an image to `ONEREF_READER_A_MODEL`, and retries on `ONEREF_FALLBACK_MODEL` if that fails.

## Workflow, from A to Z

1. **Get MMAD.** To test the pipeline, download only a **sample** (`mmad.json` + one small zip, DS-MVTec 1.6 GB by default; zips are deleted after extraction) and add `--available-only` to every run command below so only the downloaded images are used:
   ```bash
   uv run python scripts/download_mmad.py --sample                 # or --zips VisA.zip ; --dest D:/MMAD (then ONEREF_MMAD_ROOT=D:/MMAD)
   uv run python -m inspector.main --split dev --mock --limit 20 --available-only --run-name smoke
   ```
   Full dataset (`--full`, ~28 GB) otherwise. Details: Images: [Hugging Face `jiang-cc/MMAD`](https://huggingface.co/datasets/jiang-cc/MMAD) (~28 GB of zips, unzip them); annotations: `mmad.json` from [GitHub `jam-cc/MMAD`](https://github.com/jam-cc/MMAD). Put them under `data/MMAD/` so that the image paths in `mmad.json` resolve (`data/MMAD/mmad.json`). Other locations: `ONEREF_MMAD_ROOT`, `ONEREF_MMAD_JSON`. A download cell is in `Notebook/OneRef.ipynb`.
2. **Smoke test, no GPU, no network:**
   ```bash
   uv run python -m inspector.main --split dev --mock --limit 20 --run-name smoke
   ```
3. **Cost probe on 50 images** (measure time and quota before sizing the run):
   ```bash
   uv run python -m inspector.main --split dev --limit 50 --run-name probe
   ```
4. **Full dev run** (10 % of MMAD, ~840 images, stratified, fixed seed). Prompts, thresholds and vote weights are tuned **on dev only**:
   ```bash
   uv run python -m inspector.main --split dev --run-name dev01
   ```
   Relaunch the same command after an interruption: finished images are read from the cache.
5. **Freeze the configuration** (weights, thresholds, prompts) and run the **test split once**:
   ```bash
   uv run python -m inspector.main --split test --frozen-config --run-name test01
   ```
   The CLI refuses the test split without `--frozen-config`. Never tune on test.
6. **Read the results.** The CLI prints accuracy per subtask, the mean of the 7 subtasks with a 1,000-resample bootstrap 95 % CI, and writes `outputs/<run>/telemetry.jsonl` (per-image agent outputs plus a final `final_metrics` event). Splits are saved in `data/splits/`.

Strict-protocol rules (do not break): the reference is only `random_templates[0]`; no `similar_templates`, no `domain_knowledge.json`; discrimination is reported with balanced accuracy.

## Step-by-step guides (French, copy-paste)

- [`docs/guides/guide_base.md`](docs/guides/guide_base.md): command-line version, locally (PowerShell, uv or pip).
- [`docs/guides/guide_notebook.md`](docs/guides/guide_notebook.md): single-file `Notebook/OneRef.ipynb`.
- [`docs/guides/guide_colab.md`](docs/guides/guide_colab.md): Colab with local vLLM (Qwen3-VL-8B, InternVL3-8B).

## Notebooks

- [`Notebook/OneRef.ipynb`](Notebook/OneRef.ipynb): the whole package in one self-contained notebook (cells recreate `inspector/`, then tests and runs).
- [`Colab/`](Colab/): copy of the code plus a Colab notebook with local vLLM and outputs on Google Drive (see `Colab/README.md`). It is a copy: re-sync it after code changes.

## Repo layout

| Path | Content |
|---|---|
| `src/inspector/` | package: `config/` (settings, `ONEREF_` env prefix), `schemas/`, `agents/`, `orchestrator/`, `utils/` (metrics, splits, telemetry), `main.py` (CLI) |
| `tests/` | `test_workflow.py` (GPU-free), `test_llm_connection.py` (live) |
| `docs/` | plan, executive summary, references, slides |
| `data/`, `outputs/`, `.env` | gitignored: dataset, run outputs, secrets |

| Document | Content |
|---|---|
| [`docs/executive_summary.md`](docs/executive_summary.md) | One-page summary |
| [`docs/full_plan.md`](docs/full_plan.md) | Full plan: state of the art, workflow, dataset, validation, risks |
| [`docs/references.md`](docs/references.md) | References |
| [`docs/slides/`](docs/slides/) | Slides |

**Not yet validated:** the real DINOv3 expert and a full end-to-end run on MMAD (only the mock pipeline, the unit tests and the live LLM check have run).

**Deliveries:** Oct 29 (plan) · Nov 12 (preliminary) · Nov 26 (final)

**Team:** Peter Roque Vilchez · Charles Maxime Doudy
