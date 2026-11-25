# Nov 16–18: the full system on dev at the 4B scale, and the frozen test config

Summary for the Nov 19 monitoring session. Everything below is on **dev** (835 images, 3,937 questions, strict
protocol: one reference `random_templates[0]`, no added text). The test has not been touched.

> **Naming (Nov 25):** the row called B1 here is Qwen3-VL-4B **with the expert's hint text and crop** ("Qwen + expert"),
> not the model alone. The true single model ("Qwen alone", no expert evidence) was measured on 82 dev images only:
> 61.2 vs 68.3 for Qwen + expert and 68.6 for S (same images; S vs Qwen alone p = 0.041, S vs Qwen + expert p = 0.47).
> See `docs/reports/2026-11-25_test_protocol_change.md`.

## 1. Decision: 4B only, local
The 8B models and Colab are dropped (Nov 18).
- Every system runs on the team laptop (RTX 3060 6 GB, Ollama, 4-bit models), one model at a time.
- B1 is now **Qwen3-VL-4B** under the same protocol (with the expert's hint, see the naming note).
- The 8B numbers from the MMAD paper and leaderboard are cited only as external reference.

| Role | Model |
|---|---|
| Visual expert | DINOv3 ViT-B/16 (frozen), patch distance to the reference |
| Reader A | Qwen3-VL-4B-Instruct |
| Reader B | InternVL3.5-4B (Q4_K_M) |
| Verifier (critic) | Gemma 4 E2B + consistency rules, one reopen max |
| Conflict judge | Gemma 4 E2B, only on questions where the voters disagree |

## 2. Results on dev (mean of the 7 subtasks, image-level bootstrap 95 % CI)
All rows use the frozen aggregation settings (re-aggregated from the same cache, `scripts/run_test.sh` step 2 applied to dev).

| System | Mean | FPR | Missed defects | McNemar vs S |
|---|---|---|---|---|
| B1 Qwen3-VL-4B + expert | 65.9 [64.3, 67.5] | 51.7 | 21.8 | p = 0.28 |
| B2 InternVL3.5-4B alone | 60.2 [58.5, 61.8] | 27.9 | 54.2 | p < 1e-20 |
| A + B vote, no verifier | 64.4 [62.7, 66.1] | 45.2 | 27.6 | p = 0.001 |
| S without judge | 64.6 [63.0, 66.3] | 55.1 | 18.8 | p = 0.009 |
| **S (frozen)** | **66.5 [64.8, 68.2]** | 43.7 | 26.4 | — |
| B3 iso-compute (Qwen 4B × 3 samples, T = 0.7) | 65.6 [64.0, 67.3] | 52.9 | 22.4 | p = 0.14 |

Per source (S): DS-MVTec 75.7 · VisA 69.4 · GoodsAD 62.9 · MVTec-LOCO 58.5. MVTec-AD has only 2 dev images,
so its number is not meaningful.

Check against MMAD's official `summary.py` (`scripts/official_summary.py`, `results/compare/2026-11-18_official_check/`).
The official script averages per dataset (DS-MVTec and MVTec-AD merged) and drops invalid answers. Our
re-implementation matches it to two decimals: S 66.63 · B1 65.95 · B2 60.25 · S without judge 64.61.
That is within 0.2 points of our pooled metric, so the conclusions do not depend on the averaging.

## 3. What each component does (ablations, dev)
- **The conflict judge is the only component with a significant gain**: +1.9 points over S without the judge
  (p = 0.009). It mainly lowers the false-positive rate, from 55 % to 44 %.
- **The verifier is neutral**: 64.4 with it vs 64.2 without (p = 0.26, default aggregation settings,
  `results/compare/2026-11-17_dev_full_S/verifier/`). With Gemma it reopens 1,457 answers and
  changes 554 of them. Of those, 281 become right and 173 become wrong.
- **Expert hint to the readers**: +6.8 points on 82 dev images (Nov 16). This is the clearest gain of the visual
  expert.
- **Expert vote on discrimination**: no effect once the judge is on. It is kept off on GoodsAD, where the
  expert's AUROC is 0.51 (chance).
- **Heterogeneous readers**: InternVL is 5.7 points below Qwen, so the A + B vote costs more than it gains. Their
  errors are complementary, though: the oracle reaches 76.2 %. The judge recovers part of that gap.

Error traces on dev (`scripts/error_analysis.py`, `results/error_analysis/dev_S_frozen/`). S gets 1,316 of the
3,937 questions wrong. In 34 % of those failures some agent had the right answer at some point:
- 26 % are lost at the judge: it sides with the wrong voter;
- 8 % are lost at the reopen: a reader changes a right answer.

The remaining 66 % were never answered right by any agent (perception or reasoning). The 100-case labelling
sheet is in `sheet.html` / `sheet.csv`.

## 4. Reading so far
At 4B, the multi-agent system **matches** the best single model but does not beat it significantly
(+0.6 points, p = 0.28), and it is +0.9 over the iso-compute baseline B3 (p = 0.14). Sampling Qwen
3 times does not help at all (B3 vs B1: 65 of 3,937 answers change, p = 0.32), so S's small edge comes from
the heterogeneous judge, not from the extra calls. Unless the test sample turns this around, the likely outcome is a **negative result,
well measured**. The ablations already explain it:
1. The second reader is weaker.
2. The verifier fixes about as much as it breaks.
3. The expert helps through its hint, not through its vote.
4. The images where the expert is blind (GoodsAD unaligned, LOCO logical defects) stay hard for everyone.

## 5. Test plan (frozen, not yet run)
- **Config**: `configs/2026-11-19_frozen.env`. It reproduces 66.5 on the dev cache.
- **Sample**: 2,000 stratified test images, 9,490 questions (`data/splits/test_sample2000_ids.json`, no overlap
  with dev). The images are downloaded.
- **Driver**: `scripts/run_test.sh`, about 40 h on the laptop. It runs S once, then B1, B2 and the ablations from
  the cache with no extra calls, then B3, then the final McNemar table in `results/test/frozen_2026-11-19/`.
- **Success test**: S vs B1 and S vs B3, McNemar p < 0.05.

## 6. Questions for the monitoring session
1. Is a 4B-only study acceptable, with the 8B numbers as external reference?
2. Is a negative result, explained by the ablations, enough for the final delivery?
3. Are 2,000 test images (9,490 questions) enough? The full 90 % does not fit the test week on one laptop.
4. Error analysis: 100 test failures labelled perception / reasoning / orchestration / annotation. Is that the
   expected depth?
