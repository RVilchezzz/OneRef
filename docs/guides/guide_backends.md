# Model backends: hosted APIs, vLLM, local servers

Moved from the root README on Nov 19. The 4B results use **Ollama locally** (`configs/2026-11-19_frozen.env`); the 8B version uses **vLLM on Colab** (`docs/guides/guide_8B.fr.md`). Hosted APIs (Gemini, Groq, xAI, HF router) leave the strict local protocol: report their results as an upper bound only.

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

## vLLM on Colab

Downloading the model weights from Hugging Face is free; only the HF *router* is billed. Running the models yourself on a Colab GPU costs no Hugging Face credits, only Colab compute units.

**How it works.** `vllm serve <model>` loads one model on the GPU and opens an OpenAI-compatible HTTP server (`http://localhost:8000/v1`). The agents call it exactly like any hosted endpoint, so no code changes. With `ONEREF_VLLM_AUTOSTART=true`, each pass of the pipeline starts its own `vllm serve` for its model and stops it at the end of the pass, so only one 8B model is on the GPU at a time (pass 1: expert DINOv3, then reader A, reader B, critic).

**Steps** (details in [`Colab/README.md`](../../Colab/README.md)):
1. Copy the [`Colab/`](../../Colab/) folder to Google Drive (`MyDrive/OneRef/Colab`) and MMAD to `MyDrive/OneRef/MMAD` (or use the optional download cell 5b).
2. In Colab, pick a **GPU runtime: L4 (recommended, bf16)**. On a T4 (16 GB, no bf16), use AWQ-quantized checkpoints with `--dtype half`; the checkpoint ids in the notebook are not verified.
3. Open [`Colab/OneRef_Colab.ipynb`](../../Colab/OneRef_Colab.ipynb) and run the cells in order: GPU check, Drive mount, `pip install` (vLLM may replace torch: restart the runtime if Colab asks), `HF_TOKEN` as a Colab secret (DINOv3 licence), environment variables, `--mock` smoke test, 50-image probe, full dev run.
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

## Live check of an endpoint

```bash
uv run pytest                                   # GPU-free suite (mock agents, no network)
uv run pytest tests/test_llm_connection.py -s   # live check of the LLM endpoint (needs .env; skipped without a key)
```

The live test sends a text prompt and an image to `ONEREF_READER_A_MODEL`, and retries on `ONEREF_FALLBACK_MODEL` if that fails.

