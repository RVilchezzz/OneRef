# OneRef: project guide for Claude

Final project of **IA025 – Introduction to Deep Learning** (Unicamp 2026 S2). Team: Peter Roque Vilchez, Charles Maxime Doudy.
Deliveries: Oct 29 (plan, done) · **Nov 12 (preliminary, on dev)** · **Nov 26 (final)**. Test week: Nov 19–23.

## What it is
A **training-free multi-agent system** for industrial visual inspection from **one photo of the part + one normal reference photo**. It orchestrates open pre-trained models (≤ 8B), no fine-tuning. Evaluated on the **MMAD** benchmark (ICLR 2025: 8,366 images, 39,670 multiple-choice questions, 7 subtasks, 4 sources: MVTec AD, VisA, MVTec LOCO, GoodsAD).

**Question studied:** how much of the multi-agent gain reported by SiGMA (85.2 %) survives under a *strict* protocol (1 reference, no added text, no tuning on MMAD)?
**Success:** the full system beats, with McNemar p < 0.05, both the best single model (Qwen3-VL-8B, 73.91 %) and the equal-budget majority vote. A well-measured negative result is also valid.

## Architecture (orchestrated workflow, fixed flow in Python, not free-form agents)
Orchestrator → **Visual expert** (DINOv3 frozen, DINOv2 fallback: patch-distance anomaly map, 3×3 region, zoomed crop, score) → **Reader A** (Qwen3-VL-8B) and **Reader B** (InternVL3-8B) → **Verifier** (Qwen3-VL-8B critic, may reopen conflicts **once max**) → **Aggregator** (weighted vote per reader/subtask, weights fixed on dev; expert votes on discrimination).
- Agents exchange **Pydantic-validated JSON**; one retry on bad format, then the question counts as an error.
- Everything logged as JSONL per image.
- **Pass-based offline execution**: each agent runs over the whole split, caches outputs to disk, frees the GPU, then the next agent runs. Never two 8B models loaded at once.
- Keep it simple: plain Python loop; LangGraph and MCP are optional extras, not requirements.

## Strict protocol rules (do not break)
- Reference = **only `random_templates[0]`** of each `mmad.json` entry. No `similar_templates`, no `domain_knowledge.json` (optional separate "knowledge track" only).
- Split: **dev 10 %** (~840 images, stratified by source/category/normal-defective, fixed seed) for prompts and calibration; **test** (~2,000 stratified images, up to the full 90 %) run **once** with a frozen config. Never tune on test.
- Metric: accuracy per subtask + mean of 7; discrimination uses **balanced accuracy**; also report FPR and missed-defect rate. Use MMAD's `evaluation/examples/helper/summary.py`.
- Stats: McNemar + 1,000-resample image-level bootstrap 95 % CI. Error analysis: 100 failures, categories perception / reasoning / MAST orchestration / annotation.
- Systems to compare: B0 random, B1 Qwen3-VL-8B, B2 InternVL3-8B (+ Qwen3-VL-4B), B3 iso-compute vote, B4 heterogeneous vote, **S** full system. Ablations of S: no expert, no verifier, identical readers, expert as raw map/calibrated map/text, routing by subtask. Reuse cached outputs across baselines.

## Compute
- Local: RTX 3060 Laptop 6 GB: DINOv3 expert, pipeline debugging with Qwen3-VL-4B 4-bit.
- Colab Pro (100 units): 8B readers on **L4 (bf16) or T4 (4-bit), never A100**; serve with vLLM; measure units on 50 questions before fixing test size; save to Drive every N questions.

## Repo layout
- `docs/full_plan.md`: authoritative plan (state of the art, workflow, dataset, validation, risks). `executive_summary.md`, `references.md`, `mmad/LEADERBOARD_snapshot_2026-10-23.md`, `course/` (template), `slides/` (Delivery I deck, `project/` = HTML slides, speaker scripts EN/FR).
- `src/inspector/` (package `inspector`): `config/settings.py` (env prefix `ONEREF_`), `schemas/` (mmad, agent_outputs, state), `agents/` (base, llm = pydantic-ai plumbing + vLLM server per pass, visual_expert, reader_mllm, critic_verifier, aggregator, mock), `orchestrator/` (graph = ordered steps, pass_runner = one pass over the whole split + per-image cache + GPU free), `utils/` (telemetry JSONL + optional Logfire, metrics, splits), `main.py` CLI. `tests/test_workflow.py` (GPU-free).
- Run: `pip install -r requirements.txt`; `python -m pytest`; `set PYTHONPATH=src` then `python -m inspector.main --split dev --limit 50 --run-name dev01` (`--mock` = no GPU; `--split test` needs `--frozen-config`).
- Agents follow the PasteurAIze pattern (pydantic-ai): readers and the LLM critic are `Agent(OpenAIChatModel(..., OpenAIProvider(base_url)), output_type=PromptedOutput([...]), deps_type=...)`, images as `BinaryContent`, `ModelRetry` in `output_validator`s, the critic's rules exposed as the `run_consistency_rules` tool, `logfire.instrument_pydantic_ai()` when Logfire is on. The orchestrator is NOT an LLM: fixed flow in `orchestrator/`.
- MLLMs are served by an OpenAI-compatible endpoint (`ONEREF_LLM_BASE_URL`, default `http://localhost:8000/v1`): either start `vllm serve <model>` yourself for each pass, or set `ONEREF_VLLM_AUTOSTART=true` so each pass starts/stops its own server. `ONEREF_USE_LLM_CRITIC=false` = rules-only critic (ablation). Tests use pydantic-ai `FunctionModel` (`agents/mock.py`), never a real endpoint.
- Real model paths (DINOv3 expert, Qwen3-VL / InternVL3 via vLLM) are **not yet run/validated**; only mock + unit tests were executed.
- `configs/`, `notebooks/`, `data/splits/`: still empty (splits are generated by the CLI).
- `scripts/download_papers.sh`: fetches papers into `papers/` (PDFs are gitignored).
- Gitignored: `data/MMAD/`, `outputs/`, weights (`*.safetensors`, `*.bin`), zips, PDFs in `papers/`, `.venv/`.
- Parent folder (`../`) also holds the Delivery I slides/PDF and the attached paper `2601.13671v1.pdf` (Skan AI position paper, used as conceptual framework).

## Gotchas
- Leaderboard numbers are **not all comparable** (some train on MMAD, use more references, or inject text). Always state the protocol when citing; SiGMA's 85.2 % uses up to 256 normal images and team selection on MMAD answers.
- Cite original sources, not the Skan AI paper (arXiv 2601.13671): it has no experiments and several wrong author attributions (Agent Laboratory, AgentCoder, KoMA, 2402.01680).
- DINOv3 weights need accepting Meta's license on Hugging Face before download.
- Expected weak spot: MVTec LOCO logical defects (missing part) for a patch-based expert; GoodsAD is not aligned and VisA has several objects per image → report per source.
- Scope limit: single view only (one camera station); multi-view (Real-IAD) is future work.
- Docs are written in English; French notes/scripts exist for the team's use. MMAD data licenses are non-commercial.

## graphify
The knowledge graph lives in the parent folder (`../../../graphify-out/`). Run `graphify query "<question>"` before grepping, and `graphify update .` after code changes (see parent CLAUDE.md).
