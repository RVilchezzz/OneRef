# OneRef: data, visual expert and first end-to-end runs (Nov 10–11, 2026)

Work log and results of the two days before Delivery II. Everything below was run on Peter's laptop, on the **dev split only** (never on test). Every number is recomputed from the saved run files; nothing is estimated unless it says so.

**Where the evidence is**
- Run files: [`results/runs/`](../../results/runs/), a copy of the gitignored `outputs/` folder.
- Per-question tables: [`results/pipeline_smoke/`](../../results/pipeline_smoke/).
- Expert results: [`results/expert_eval/`](../../results/expert_eval/).
- Code: branch `peter/expert-dev` (PR #3).

---

## 1. Summary

- **The whole multi-agent workflow ran end to end on real MMAD images.** The steps were: visual expert → reader A → reader B → verifier → reopen of the flagged questions → vote.
- **Largest run:** 18 dev images (2–4 per source), 88 questions, 184 model calls, **0 invalid answers**.
  - Mean of the 7 subtasks: **71.2 %**, 95 % CI [52.4 %, 85.2 %].
  - The verifier reopened 4 questions.
  - This was a smoke test with a small reader: Qwen3-VL-4B quantized, served by Ollama, as both readers. It proves the pipeline works, not that it beats any baseline.
- **The visual expert alone was measured on all 835 dev images.**
  - It works on aligned photos: AUROC (area under the ROC curve; 0.5 = chance) is 0.84–0.88 on MVTec AD.
  - It fails when the single reference is another product: AUROC 0.51 on GoodsAD.
  - DINOv3 localizes better than DINOv2: on MVTec AD, the map's hottest point falls on the true defect 77 % of the time, against 51 %.
- **Three bugs were found by running on real data, and all three are fixed:**
  - the loader crashed on the real file;
  - large GoodsAD images overflowed the reader's context;
  - one refused request aborted a whole pass.
  
  A fourth problem was a calibration constant on the wrong scale; it was refitted on dev.
- **New finding:** in the pipeline, discrimination rejects many good parts (false-positive rate 70 % on the 10 normal images of run 3). The expert's hint pushes the readers toward "defective", so its weight must depend on how reliable it is.

---

## 2. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3060 Laptop, 6 GB (driver 615.78) |
| Python | 3.14.8, project venv `.venv/` (gitignored) |
| Libraries | torch 2.11.0+cu128, transformers 5.22.2, pydantic 2.14.3, pydantic-ai-slim 2.66.0, numpy 2.5.4, Pillow 13.0 |
| Visual expert | `facebook/dinov3-vitb16-pretrain-lvd1689m` (frozen, 327 MB, gated: license accepted on HF); fallback `facebook/dinov2-base` |
| Reader (smoke test) | `qwen3-vl:4b-instruct` (Q4 quantized, 3.3 GB) served by **Ollama 0.40.7**, an OpenAI-compatible local server |
| Data | `mmad.json`, SHA-256 `c219c6b0…08143`: matches the corrected revision pinned in the plan |

---

## 3. Data

### 3.1 Loader audit on the real `mmad.json`
- 8,366 images, **8,366 unique ids**, matching the paper.
- Labels: 5,076 defective and 3,290 normal. The defective flag (taken from `mask_path`) disagrees with the folder name (`/good/`) in **0** cases.
- The 9 raw question types map to the 7 official subtasks. "Object Analysis", "Object Structure" and "Object Details" are merged: 3,084 + 3,052 + 3,024 = **9,160**, as in the plan.

### 3.2 Split (`src/inspector/utils/splits.py`, seed 0, stratified by source × category × normal/defective)
| | Images | Questions |
|---|---|---|
| Dev (10 %) | **835** | **3,937** |
| Test pool (90 %) | 7,531 | 35,733 |

- Dev by source: GoodsAD 296, VisA 215, DS-MVTec 166, MVTec LOCO 156, MVTec-AD 2.
- Dev uses 742 distinct references (`random_templates[0]`).
- The id lists are committed: `data/splits/dev_ids.json` and `test_ids.json`.

### 3.3 Downloading only what we need (`scripts/fetch_mmad_subset.py`)
The MMAD zips on Hugging Face accept HTTP range requests (downloading only a byte range of a file). The script:
1. reads each zip's table of contents (its central directory);
2. downloads only the members a split needs: the query image, its single reference and, with `--masks`, the masks, which are used for evaluation only.

| | Files | Size |
|---|---|---|
| Full MMAD zips | — | ~28 GB |
| **Dev split with references and masks** | **2,084** (DS-MVTec 256, MVTec-AD 141, VisA 540, MVTec LOCO 414, GoodsAD 706) | **2.1 GB** |

0 files missing. The script can be re-run safely, because it skips files that are already on disk.

---

## 4. Bugs found by running on real data, and fixes

| # | Problem | Evidence | Fix | Commit |
|---|---|---|---|---|
| 1 | The loader crashed on the real file | `ValueError: Unknown MMAD subtask type: 'Anomaly Detection'`: mmad.json calls discrimination "Anomaly Detection" and the mapping only knew "Anomaly Discrimination". The unit tests used fake data, so they passed 20/20 | Mapping added (`src/` and `Colab/`) + test with the 9 real type names | `711294a` |
| 2 | Large images overflowed the reader's context | Run 1: `request (12492 tokens) exceeds the available context size (8192 tokens)` on a GoodsAD photo (images were sent at full resolution, ×3 per question) | Images larger than `reader_image_max_side` = 768 px are sent as downscaled JPEGs (`agents/llm.py: image_part`). Average input in runs 2–3: ~3,470 tokens per call | `353647e` |
| 3 | One refused request stopped the whole pass | The exception above was not caught: run 1 aborted after 3 images | A request the server rejects (context too long, invalid request) is a **missing answer that counts as an error** and stays in the denominator, as the plan requires (`agents/reader_mllm.py`) | `353647e` |
| 4 | Expert threshold on the wrong scale | DINOv3 distances are 0.11–0.31 (median), but the calibration range was 0.2–0.8, so the "defect" flag almost never fired: balanced accuracy 0.49 | Refitted on dev (§6.4); recorded in `configs/2026-11-12_local_ollama_smoke.env` | `c15bc0b` |

---

## 5. Tests

`PYTHONPATH=src python -m pytest`: **22 passed**, both in `src/` and in the `Colab/` copy. There were 20 before this work.

| New test | Checks |
|---|---|
| `test_parse_subtask_accepts_every_type_in_mmad_json` | The 9 type strings of the real file map to the right subtask (regression test for bug 1) |
| `test_sample_per_source_is_balanced_and_deterministic` | The new `--per-source K` sampling picks K images per source, half normal and half defective, and the same images for the same seed |

---

## 6. Visual expert alone on the 835 dev images

### 6.1 Method (`scripts/eval_expert.py`, using `agents/visual_expert.py`)
1. **Features.** Both images are resized to 448 px. A frozen DINO model extracts one feature vector per patch (a small square of the image):
   - DINOv3: 16 px patches, 28×28 grid;
   - DINOv2: 14 px patches, 32×32 grid.
2. **Anomaly score per patch.** For each patch of the part, the score is 1 − cosine similarity with the **most similar patch of the reference**. Using the most similar patch tolerates small shifts in position.
3. **Image score.** The image score is the peak (the maximum over patches). The peak's location gives the 3×3 region and the zoomed crop.
4. **Evaluation.** Masks are used **only to score**:
   - AUROC of the peak, normal vs. defective;
   - balanced accuracy at the threshold;
   - "peak in mask": the share of defective images whose hottest point falls on the true defect.

### 6.2 Results
| Source (dev) | Images (defective) | AUROC DINOv2 | AUROC DINOv3 | Peak in mask DINOv2 | Peak in mask DINOv3 |
|---|---|---|---|---|---|
| MVTec AD (DS-MVTec) | 166 (122) | **0.883** | 0.843 | 50.8 % | **77.0 %** |
| VisA | 215 (120) | 0.768 | 0.783 | 41.7 % | 40.8 % |
| MVTec LOCO | 156 (99) | 0.635 | 0.622 | 37.4 % | 43.4 % |
| GoodsAD | 296 (166) | 0.545 | 0.513 | 12.0 % | 10.2 % |
| **All** | 835 (507) | **0.608** | 0.588 | 33.3 % | **40.0 %** |

- Speed: 0.34 s per image with DINOv2 and 0.35 s with DINOv3; all of dev takes about 5 min.
- Median peak, normal vs. defective: 0.509 vs. 0.617 with DINOv2, and 0.211 vs. 0.308 with DINOv3.

**Reading:**
- **DINOv3 is the better *localizer*:** 77 % vs. 51 % on MVTec AD. The two models are equivalent for detection.
- **GoodsAD is at chance:** the single random reference is often another product, such as a different brand of can or another box.
- **LOCO is weak, as expected:** its defects are logical (a missing separator, a missing label), not local stains.

### 6.3 Score variants tried (DINOv2, AUROC)
| Source | peak | peak − mean | peak / mean |
|---|---|---|---|
| All | **0.608** | 0.614 | 0.541 |
| MVTec AD | **0.883** | 0.865 | 0.644 |
| VisA | **0.768** | 0.736 | 0.568 |
| MVTec LOCO | **0.635** | 0.613 | 0.538 |
| GoodsAD | 0.545 | 0.522 | 0.462 |

The idea was to cancel the "whole image is different" effect of a mismatched reference. It doesn't help, so we keep the plain peak.

### 6.4 Threshold calibration (dev)
- A single global threshold on the DINOv3 peak, **0.165**, maximizes balanced accuracy on dev: **0.49 → 0.61**.
- It is applied in the config as `anomaly_norm_low/high` = 0.065/0.265, which puts the 0.5 score exactly at 0.165.
- Per source: MVTec AD 0.727, VisA 0.695, MVTec LOCO 0.588, GoodsAD 0.500.
- **Caveat:** the threshold is fitted and measured on the same dev images, so these numbers are optimistic (in-sample).

### 6.5 Heatmap examples
They are in `docs/slides/delivery_II/assets/` and on the "Reference · part · our DINOv3 map" slide:
- **MVTec AD grid:** the hot spot sits on the defect.
- **VisA chewing gum:** the hot spot sits on the tear.
- **GoodsAD food box:** the reference is a green box and the part is a pink one, so the whole box lights up.
- **MVTec LOCO juice:** the reference is mango juice and the part is banana juice, so the map flags the label (a normal variation) instead of the defect.

---

## 7. Full pipeline runs

### 7.1 Setup
- **Serving.** `ollama serve` with `OLLAMA_CONTEXT_LENGTH=8192`, then `ollama pull qwen3-vl:4b-instruct`. The pipeline's readers talk to it through `ONEREF_LLM_BASE_URL=http://localhost:11434/v1`.
- **Config:** [`configs/2026-11-12_local_ollama_smoke.env`](../../configs/2026-11-12_local_ollama_smoke.env).
  - Reader A = reader B = the 4B model. This is the "identical readers" ablation, not the planned Qwen-8B + InternVL3-8B pair.
  - **Rules-only verifier** (`ONEREF_USE_LLM_CRITIC=false`): no LLM critic pass.
  - Expert threshold from §6.4.
- **Image selection:** new option `--per-source K` (`utils/splits.py: sample_per_source`), with K/2 normal and K/2 defective images per source, seed 0.
- **Command:**
  ```bash
  set -a; source configs/2026-11-12_local_ollama_smoke.env; set +a
  PYTHONPATH=src .venv/bin/python -m inspector.main --split dev --per-source 4 --run-name smoke_ollama_qwen4b_ps4
  ```
- **Pass-based execution:** each agent runs over all images, caches every state to `<run>/<step>/`, frees the GPU, and only then does the next pass start.

### 7.2 Run 1, `smoke_ollama_qwen4b_ps2`: stopped
- The expert pass finished: 9/9 images in 10 s.
- The reader pass answered 3 images, all MVTec, with valid JSON. Example answer, on a normal transistor: `{"selected_option": "B", "confidence": 0.95, "brief_rationale": "The object shows no visible defects compared to the reference…"}`, about 3,400 input tokens.
- Then a GoodsAD photo hit bugs 2 and 3 (§4).

### 7.3 Runs 2 and 3: results
| | Run 2 `…_ps2_v2` | Run 3 `…_ps4` |
|---|---|---|
| Images (defective) | 9 (4) | 18 (8) |
| Per source | DS-MVTec 2, GoodsAD 2, MVTec-AD 1, LOCO 2, VisA 2 | DS-MVTec 4, GoodsAD 4, MVTec-AD 2, LOCO 4, VisA 4 |
| Questions | 45 | 88 |
| Anomaly discrimination (balanced) | 57.5 % | 52.5 % |
| Defect classification | 50.0 % | 62.5 % |
| Defect localization | 25.0 % | 50.0 % |
| Defect description | 50.0 % | 62.5 % |
| Defect analysis | 75.0 % | 87.5 % |
| Object classification | 100.0 % | 100.0 % |
| Object analysis | 93.3 % | 83.3 % |
| **Mean of 7 subtasks** | **64.4 %** [37.0, 81.9] | **71.2 %** [52.4, 85.2] |
| Good parts called defective (FPR) | 60 % | 70 % |
| Defects missed | 25 % | 25 % |
| Reader calls / invalid | 94 / 0 | 184 / 0 |
| Avg. input tokens per call | 3,471 | 3,476 |
| Questions reopened by the verifier | 2 (2 images) | 4 (4 images) |
| Reader A = reader B | 45 / 45 | 87 / 88 |

- The 95 % CI is an image-level bootstrap with 1,000 resamples.
- The CLI printed [37.0, 82.1] and [53.8, 85.6] for the same runs. The difference is only the order in which the bootstrap resamples images; the means are identical.

**Pass timings (seconds)**
| Pass | Run 2 | Run 3 |
|---|---|---|
| 1 Expert (DINOv3) | 10.7 | 18.0 |
| 2 Reader A | 178.9 | 322.9 |
| 3 Reader B | 180.6 | 322.0 |
| 4 Verifier (rules) | 0.5 | 0.4 |
| 5 Reader A reopen | 8.8 | 15.7 |
| 6 Reader B reopen | 3.5 | 5.7 |
| 7 Aggregator | 0.6 | 0.4 |

That is about 3.7–4.0 s per question per reader on the 6 GB GPU, and about 11 min of total wall time for run 3.

### 7.4 Run 3, image by image
| Image | Source | Truth | Expert region | Expert score | Reopened Qs | Correct |
|---|---|---|---|---|---|---|
| cable/image/missing_cable/009 | DS-MVTec | defective | center | 1.00 | 0 | 5/5 |
| carpet/image/good/019 | DS-MVTec | normal | bottom left | 0.73 | 1 | 3/5 |
| tile/image/oil/010 | DS-MVTec | defective | bottom center | 1.00 | 1 | 3/5 |
| transistor/image/good/003 | DS-MVTec | normal | center | 0.28 | 0 | 5/5 |
| cigarette_box/test/good/039_003 | GoodsAD | normal | center | 1.00 | 0 | 3/5 |
| drink_can/test/good/019_002 | GoodsAD | normal | bottom center | 1.00 | 0 | 4/5 |
| food_box/test/deformation/027_008 | GoodsAD | defective | center | 1.00 | 0 | 2/4 |
| food_package/test/broken/069_009 | GoodsAD | defective | center | 1.00 | 0 | 4/5 |
| pill/test/good/010 | MVTec-AD | normal | middle left | 0.27 | 0 | 5/5 |
| pill/test/good/011 | MVTec-AD | normal | bottom center | 0.40 | 0 | 3/5 |
| juice_bottle/test/good/010 | MVTec LOCO | normal | center | 0.81 | 0 | 4/5 |
| pushpins/test/logical_anomalies/071 | MVTec LOCO | defective | top left | 0.55 | 1 | 4/4 |
| pushpins/test/logical_anomalies/080 | MVTec LOCO | defective | top right | 0.40 | 0 | 0/5 |
| screw_bag/test/good/118 | MVTec LOCO | normal | middle left | 0.73 | 0 | 4/5 |
| candle/test/good/0730 | VisA | normal | middle left | 0.32 | 0 | 5/5 |
| macaroni2/test/bad/058 | VisA | defective | center | 0.85 | 1 | 4/5 |
| macaroni2/test/good/0294 | VisA | normal | top left | 0.85 | 0 | 2/5 |
| pcb3/test/bad/068 | VisA | defective | middle left | 0.32 | 0 | 4/5 |

- The expert score is the calibrated score in [0, 1]; it flags "defect" at ≥ 0.5.
- Every GoodsAD image gets 1.00, normal ones included. That is the mismatched-reference problem of §6.2.

### 7.5 Four cases in detail (run 3)
These are on the "Four real cases" slide. All quotes are reader A's rationale as logged.

**Cable, `missing_cable/009` (MVTec AD), all correct.**
- Expert: center, score 1.00.
- *Where is the defect?* A. outer sheath, **B. central part of the cable**, C. end, D. brown wire. Truth B, answer B, confidence 0.95.
- *Type?* **A. Missing component**, correct. *Appearance?* **B. Two large voids where wires should be**, correct. *Effect?* **C. Missing cable components, critical for functionality**, correct.

**Tile, `oil/010` (MVTec AD), the verifier reopens.**
- Expert: bottom center, 1.00.
- *Where is the defect?* A. Top-right, B. Center-right, **C. Center-left**, D. Bottom-left. Both readers answered D.
- Verifier message: *"reader_a: chose 'bottom left' but the expert's strongest region is 'bottom center'. reader_b: chose 'bottom left' but…"* → the question was reopened once with that message.
- After the reopen, both readers kept D, so the answer is still wrong (truth C).
- *Type?* The readers answered A. crack; the truth is C. discoloration.
- Lesson: the reopen mechanism works, but here it didn't change the answer.

**Drink can, `good/019_002` (GoodsAD), false alarm.**
- Expert: bottom center, 1.00, on a **normal** can. The reference is a can of another brand.
- *Is there any defect?* A. No, B. Yes. Truth A; answer **B** (confidence 0.95): *"The close-up shows a clear defect: the can's label is misaligned and distorted…"*
- The object questions were all right, e.g. it read the Chinese text and answered "beer can".

**Pushpins, `logical_anomalies/080` (MVTec LOCO), the reader follows the hint.**
- Expert: top right, 0.40, below the flag.
- *Is there any defect?* Truth A (yes); answer B (no): *"All pins appear uniform and intact…"*
- *Where is the defect?* Truth A (bottom center); answer **C (top right)**: *"The visual expert hint indicates the top right region as the most different, and the zoomed crop confirms a defect there."*
- *Appearance?* Truth B (missing separator); answer A (bent pins). Result: 0/5.
- Lesson: the hint is strong in both directions. An unreliable hint drags the reader with it.

### 7.6 What the runs tell us
1. **The engineering works.**
   - All 7 passes ran, one model at a time on a 6 GB GPU.
   - JSON validation, caching and reopens behaved as designed, with 0 invalid answers.
2. **Object questions are easy and defect questions are hard,** as MMAD reports.
3. **Discrimination is the weak point:** it rejected 70 % of the good parts in run 3, with the expert scoring every GoodsAD image 1.00. The fix is in the plan: the expert's weight and its hint should depend on how reliable it was on dev per subtask and source, possibly without a hint when the score is unreliable.
4. **Identical readers are useless as a "team":** they agreed 87/88 times. This is direct evidence for the planned heterogeneous pair (Qwen3-VL-8B + InternVL3-8B).

---

## 8. What did not work, or was dropped
- **Hugging Face download of Qwen3-VL-4B (bf16, 8.9 GB):**
  - the Xet transfer ran at ~110 KB/s;
  - with Xet disabled it ran at ~2 MB/s, about 1 h.
  
  Replaced by Ollama's 3.3 GB quantized model. The partial download stays in the HF cache.
- **`scripts/quick_reader_baseline.py`:** a transformers 4-bit reader with and without the expert hint. It was written and syntax-checked but **never run**, because of the download above. It has `--bits 16` for a Colab L4 and could give the first with/without-hint comparison on 8B.
- **Installing Ollama without sudo** (1.5 GB GitHub tarball) was started and then cancelled for `pacman -S ollama-cuda` (766 MB), which Peter installed from a terminal.

---

## 9. Limitations
- **Small reader:** 4B, quantized to 4 bits. The plan's readers are 8B models in bf16 on Colab.
- **Identical readers:** both readers were the same model. InternVL3 was not run.
- **Tiny n:** 9 and 18 images. The confidence intervals are 30–45 points wide.
- **Rules-only verifier:** the LLM critic was not run.
- **In-sample calibration:** the expert threshold was fitted and evaluated on the same dev images.
- **No without-expert baseline:** there is no run without the expert yet, so the effect of the hint is **not measured**; the examples are qualitative.
- **Single seed.**
- **No official cross-check:** the numbers have not been compared with MMAD's official `summary.py`.

---

## 10. How to reproduce
```bash
# environment (once)
python3 -m venv .venv
.venv/bin/pip install pydantic pydantic-settings "pydantic-ai-slim[openai]" pytest numpy pillow scipy "transformers>=4.56"
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/hf auth login            # token with access to facebook/dinov3-vitb16-pretrain-lvd1689m

# data: mmad.json + only the dev images (2.1 GB)
mkdir -p data/MMAD
curl -L -o data/MMAD/mmad.json https://huggingface.co/datasets/jiang-cc/MMAD/resolve/main/mmad.json
.venv/bin/python scripts/fetch_mmad_subset.py --split dev --masks

# tests
PYTHONPATH=src .venv/bin/python -m pytest -q

# visual expert alone on dev
.venv/bin/python scripts/eval_expert.py --split dev --run-name dev_dinov3-vitb16_v1
ONEREF_EXPERT_MODEL=facebook/dinov2-base .venv/bin/python scripts/eval_expert.py --split dev --run-name dev_dinov2-base_v1

# full pipeline with a local reader (Arch: sudo pacman -S ollama-cuda)
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &
ollama pull qwen3-vl:4b-instruct
set -a; source configs/2026-11-12_local_ollama_smoke.env; set +a
PYTHONPATH=src .venv/bin/python -m inspector.main --split dev --per-source 4 --run-name smoke_ollama_qwen4b_ps4
```

---

## 11. Next steps
1. **Colab, L4:** B1 (Qwen3-VL-8B) and B2 (InternVL3-8B) on dev. Start with a 50-question cost test, then the 3,937 questions. Also run `quick_reader_baseline.py --bits 16` to measure the hint's effect with and without it.
2. **Run the full system on dev** with the heterogeneous readers and the LLM critic.
3. **Calibrate on dev:**
   - the expert's vote weight and when to show its hint, per subtask and per source;
   - the aggregator's weights.
4. **Cross-check** our metrics against MMAD's `summary.py`.
5. **Freeze the config on Nov 19** (`v-frozen`), then run the test split once.

---

## 12. File index
| Path | What |
|---|---|
| `scripts/fetch_mmad_subset.py` | Download a split's images, references and masks via HTTP range requests |
| `scripts/eval_expert.py` | Visual expert alone: AUROC, balanced accuracy, peak-in-mask, per source |
| `scripts/quick_reader_baseline.py` | Reader with and without the expert's hint (transformers; not run yet) |
| `src/inspector/main.py` | CLI; new `--per-source K` |
| `src/inspector/utils/splits.py` | `make_split`, `sample_per_source` |
| `src/inspector/agents/llm.py`, `reader_mllm.py` | Image downscaling; refused requests counted as errors |
| `configs/2026-11-12_local_ollama_smoke.env` | Config of runs 2–3, with the calibration rationale |
| `data/splits/*.json` | Dev and test ids |
| `results/expert_eval/` | Expert summaries and per-image scores (DINOv2, DINOv3) |
| `results/pipeline_smoke/<run>/` | `summary.json` + `per_question.csv` for runs 2 and 3 |
| `results/runs/` | Copy of `outputs/`: telemetry, per-pass states, heatmaps and crops (JPEG) |
| `docs/slides/delivery_II/` | Delivery II slides and their images |

| Commit (branch `peter/expert-dev`) | Content |
|---|---|
| `711294a` | Loader fix, dev/test split, subset fetcher, expert evaluation |
| `79376f2`, `1594169` | Delivery II slides, reader smoke-test script, expert results |
| `c15bc0b` | `--per-source`, local Ollama config |
| `353647e` | Image downscaling, refused requests counted as errors, first end-to-end run |
| `4da56e4`, `f6a77a2` | Slides with the 18-image run and the four examples |
