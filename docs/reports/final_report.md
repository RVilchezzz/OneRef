# OneRef: a training-free multi-agent inspector from a single reference, measured under the strict MMAD protocol

IA025 – Introduction to Deep Learning, Unicamp 2026 S2 · Final delivery (Nov 26, 2026)
Peter Roque Vilchez · Charles Maxime Doudy

> **Skeleton (Nov 19).**
> - The dev numbers are final. `TBD(test)` marks a number that the frozen test run fills in.
> - The test results come from `results/test/frozen_2026-11-19/` (`summary.md`, `official/summary.md`) and the
>   error-analysis sheet from `results/error_analysis/test_S/`.
> - `[owner]` tags who drafts each section. `[fig]` marks a figure still to make.
> - The section order follows the course template (`docs/course/Modelo_Plano_do_Projeto_Final.pdf`): description,
>   methodology, datasets, metrics, results, references, schedule.

## Abstract  [Peter, last]
5–6 sentences.
- **Question:** how much of the reported multi-agent gain on MMAD survives under the strict protocol (1 normal reference, no
  added text, no tuning on MMAD) with open 4B models on one 6 GB laptop GPU?
- **System:** DINOv3 visual expert, two heterogeneous MLLM readers (Qwen3-VL-4B, InternVL3.5-4B), Gemma 4 E2B verifier and
  conflict judge.
- **Result:** S `TBD(test)` vs best single model `TBD(test)` (McNemar p = `TBD(test)`) and vs a same-compute vote
  `TBD(test)`. On dev: 66.5 vs 65.9 (p = 0.28) and 65.6 (p = 0.14).
- **Take-away:** which components help (the judge, the expert hint), which do not (the verifier, the expert vote, sampling
  the same model again), and the 34 % of failures that better orchestration could recover.

## 1. Description of the project  [Peter]
### 1.1 Problem
- A production line switches products, and only one photo of a good part is available. There is no data or time to train a
  detector.
- MLLMs can inspect from instructions, but on MMAD the best is GPT-4o at 74.9 %, against 86.7 % for expert inspectors (Jiang
  et al., ICLR 2025).
### 1.2 Hypothesis
- A training-free team (visual expert + two readers + verifier + judge) beats the best single model of the same size **and**
  a majority vote with the same number of calls, with McNemar p < 0.05 on the test sample.
- A well-measured negative result is also valid (MAST, Cemri et al. 2025).
### 1.3 Contribution
1. The first training-free multi-agent system of ≤ 4B models evaluated under the strict MMAD protocol.
2. Per-component ablations with paired tests.
3. A failure taxonomy that shows where the agents lose correct answers.
### 1.4 Scope changes since the plan  [both]
- **8B → 4B** (Nov 18). Every run must fit a single laptop GPU (RTX 3060, 6 GB) for the test week. B1 is therefore
  Qwen3-VL-4B under the same protocol. The 73.9 % of Qwen3-VL-8B is cited only as external reference.
- **Colab dropped.** Everything runs locally through Ollama, one model at a time, with a pass-based cache.
- **Conflict judge added** (Nov 17), after the verifier turned out neutral on dev. It is the only component added after the
  plan. It was tuned on dev only, before the test freeze.
- **Test = 2,000-image stratified sample** of the 90 % test split, not the full split (compute; ~24 h for S alone).

## 2. Related work  [Charles]
Condense `docs/full_plan.md` §2 to about one page.
- Orchestration patterns: Anthropic 2024 (workflows vs agents), multi-agent debate (Du et al. 2024), verification as a role
  (AgentCoder).
- Failures of multi-agent systems: MAST (Cemri et al., NeurIPS 2025). The gains over best-of-N are often minimal, so **each
  agent needs an ablation**.
- MLLMs on MMAD:
  - The MMAD findings: classic detector maps hurt, while an oracle mask gives +24.9 on discrimination.
  - The leaderboard table with the protocol of each entry: SiGMA 85.2 with 256 normal images and MMAD text; AD-Copilot,
    JUDO, OmniAD trained; Reason-IAD with domain text; Qwen3-VL-8B 73.9 strict.
- The gap: no training-free multi-agent system of small open models under the strict protocol.
- Conceptual framework: Adimulam et al. 2026 (position paper, no experiments; cite the primary sources instead).

## 3. Methodology  [Peter: 3.1–3.4 · Charles: 3.5]
### 3.1 Workflow
`[fig]` The pipeline diagram (reuse the Delivery I `metodologia` figure; add the judge).
- It is a fixed workflow, not an autonomous agent, in the Anthropic sense.
- Each pass writes one JSON state per image (`InspectionState`), so any pass can be re-run or re-aggregated without the GPU.

| Pass | Agent | Model | Output |
|---|---|---|---|
| 1 | Visual expert | DINOv3 ViT-B/16, frozen | Patch-distance anomaly map vs the reference, score, crop, region hint |
| 2 | Reader A | Qwen3-VL-4B-Instruct | One letter per question, with the expert hint in the prompt |
| 3 | Reader B | InternVL3.5-4B (Q4_K_M GGUF) | Same, independent |
| 4 | Verifier | Gemma 4 E2B + consistency rules | Flags contradictions (e.g. "no defect" + defect type), max 1 reopen |
| 5–6 | Reopen | Readers A/B | Revised answers on flagged questions |
| 6b | Conflict judge | Gemma 4 E2B | Only where the voters disagree: picks a side (override) |
| 7 | Aggregator | Rules | Weighted vote: readers + expert on discrimination (per-source weights) |

### 3.2 The visual expert
- Patch features of the part and the reference, then nearest-neighbour distance, then a normalized map. The map gives the
  score, the bounding box of the hot region, and a 3×3 position hint ("top-left"…).
- Calibration on dev: `anomaly_norm_low/high` = 0.065 / 0.265, and per-source thresholds (DS-MVTec 0.7, LOCO 0.7, VisA 0.3).
- **Known blind spots:** GoodsAD is not aligned, so AUROC is 0.51 (chance); LOCO has logical defects; VisA has several objects
  per image.
- `[fig]` Two hits (MVTec, VisA) and two misses (GoodsAD, LOCO). These come from `docs/slides/delivery_II/assets/`.
### 3.3 Readers, verifier, judge
- Prompts: the `full` hint. Answer format: one letter, parsed strictly. An unparsable answer counts as wrong.
- Verifier rules and the reopen budget.
- Judge: what it sees (both answers, the rationales, the expert summary), and `reasoning_effort: none`.
### 3.4 Baselines (same images, same protocol)
| Id | System | Calls / question |
|---|---|---|
| B1 | Qwen3-VL-4B alone | 1 |
| B2 | InternVL3.5-4B alone | 1 |
| B3 | Qwen3-VL-4B sampled k = 3 (T = 0.7) + majority vote: iso-compute | 3 |
| S | Full system | 2.64 (10,402 calls / 3,937 dev questions) |

B1, B2 and the ablations are re-aggregations of S's own cache, with no extra calls. They therefore see exactly the same
model outputs.
### 3.5 Implementation and compute  [Charles]
- Python 3.12, pydantic-ai, Ollama (`OLLAMA_CONTEXT_LENGTH=8192`), RTX 3060 Laptop with 6 GB.
- Throughput: about 10 h per 835 dev images for S, and about 2 images per minute for B3.
- Reproducibility:
  - frozen env file `configs/2026-11-19_frozen.env`;
  - seeds;
  - `scripts/run_test.sh`;
  - dev caches in `results/runs/2026-11-17_dev_full_S/states.tar.xz`.

## 4. Dataset  [Charles]
- **MMAD** (corrected revision of Oct 22, 2026): 8,366 images, 39,670 questions, 38 categories, 4 sources. `[table]` The
  sources from `full_plan.md` §5.
- **Split:**
  - dev is 10 %: 835 images, 3,937 questions. Every design choice was made on dev.
  - test is 90 %, run **once** with `--frozen-config` on a stratified sample of 2,000 images (9,490 questions). The ids are in
    `data/splits/test_sample2000_ids.json` and do not overlap with dev.
- **Strict protocol:**
  - the reference is `random_templates[0]` only;
  - no `domain_knowledge.json`;
  - no MMAD answers used for tuning outside dev.

## 5. Metrics  [Charles]
- Accuracy per subtask (7) and their mean. Anomaly discrimination uses balanced accuracy.
- FPR (good parts called defective) and missed-defect rate. These show the trade-off that the judge moves.
- **Official MMAD metric** (`summary.py`): per-dataset mean, DS-MVTec merged with MVTec-AD, invalid answers dropped.
  - Re-implemented in `scripts/official_summary.py` and checked to two decimals against the original script.
  - We report both. On dev they differ by ≤ 0.2 points.
- **Significance:** exact McNemar test on paired questions (S vs each baseline), plus a 95 % bootstrap CI by image.

## 6. Results  [Peter]
### 6.1 Main result (test, frozen)
| System | Mean [95 % CI] | Official | FPR | Missed | McNemar vs S |
|---|---|---|---|---|---|
| B1 Qwen3-VL-4B | `TBD(test)` | | | | |
| B2 InternVL3.5-4B | `TBD(test)` | | | | |
| B3 Qwen × 3 (iso-compute) | `TBD(test)` | | | | |
| **S** | `TBD(test)` | | | | — |

Dev reference (frozen aggregation): S 66.5 [64.8, 68.2] · B1 65.9 (p = 0.28) · B3 65.6 (p = 0.14) · B2 60.2 (p < 1e-20).
Official metric on dev: S 66.63 · B1 65.95 · B2 60.25.

`[fig]` A bar chart of the means with CIs, dev and test side by side. External references as dashed lines: Qwen3-VL-8B 73.9,
GPT-4o 74.9, humans 86.7.
### 6.2 Per subtask and per source
- `[table]` The 7 subtasks × systems (test). The pattern on dev:
  - S gains on discrimination (+1.8), description (+3.4) and analysis (+3.2);
  - it loses on object classification (−2.6) and localization (−1.9).
- `[table]` Per source (test). On dev: DS-MVTec 75.7 · VisA 69.4 · GoodsAD 62.9 · LOCO 58.5.
### 6.3 Ablations (test, from the cache)
| Variant | Dev | Test | p vs S (test) |
|---|---|---|---|
| A + B vote, no verifier, no judge | 64.4 | `TBD(test)` | |
| S without judge | 64.6 | `TBD(test)` | |
| S without expert vote | — | `TBD(test)` | |
| Verifier on vs off (default aggregation, dev) | 64.4 vs 64.2, p = 0.26 | — | |
| Expert hint vs no hint (82 dev images, Nov 16) | +6.8 | — | |
### 6.4 Error analysis  [both, Nov 21–23]
- Trace categories (automatic) on the test failures. On dev, of the 1,316 failures:
  - 66 % no agent ever right;
  - 26 % lost at the judge;
  - 8 % lost at the reopen.
- **Manual labels** on 100 stratified test failures: perception / reasoning / orchestration / annotation. Report the counts,
  Cohen's κ between Peter and Maxime on a shared subset of 20, and 3–4 illustrated cases.
- Map each category to MAST: orchestration ≈ inter-agent misalignment + task verification.
### 6.5 Cost
- Calls, tokens and wall-clock per image for S vs B1 vs B3, from `telemetry/`.
- `[fig]` Accuracy vs calls per question.

## 7. Discussion  [both]
- **Why the team barely beats one model at 4B:**
  1. The second reader is 5.7 points weaker.
  2. The verifier fixes about as much as it breaks: 281 answers become right and 173 become wrong.
  3. The expert helps through its hint, not its vote.
  4. The blind spots (GoodsAD, LOCO logical defects) stay hard for every agent.
- **What does work:**
  - The heterogeneity of the two readers: the oracle reaches 76.2 % on dev, and the judge recovers part of that gap.
  - Sampling the same model again does not help (B3 ≈ B1). So the gain comes from **different** models, not from more calls.
- **Comparison to SiGMA's 85.2:** which of their advantages we removed (256 normals, MMAD text, team selection on MMAD,
  9B models), and an estimate of how much each one is worth.
- **Threats to validity:**
  - The test is a sample, not the full split.
  - Ollama's 4-bit quantization.
  - The judge was designed after looking at dev.
  - LLM non-determinism at T = 0 under Ollama.

## 8. Conclusion and future work  [Peter]
- Answer the question in one paragraph.
- Future work:
  - stronger second reader at the same size;
  - an expert that is aware of alignment, for GoodsAD;
  - a learned or calibrated judge;
  - several references (the protocol relaxed, reported separately).

## References  [Charles]
From `docs/references.md`. Cite the primary sources, not Adimulam et al. as an intermediary.

## Appendix
- A. Prompts (reader, verifier, judge).
- B. The frozen config, line by line with its rationale.
- C. Full per-subtask × per-source tables (test, official format).
- D. Schedule, planned vs done: Delivery I plan → Nov 12 → Nov 19 freeze → Nov 26.
