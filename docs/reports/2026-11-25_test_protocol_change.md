# Nov 25: change of the test protocol, written before any test answer was produced

Until now no model has answered any test question (`outputs/` holds no test run). The system S is the one frozen on
Nov 19 (`configs/2026-11-19_frozen.env`); **no setting, prompt or line of the pipeline changes**.

## What changes
1. **Size for the presentation (Nov 26): 200 test images**, 953 questions: `--split test --sample 200`, stratified by
   source, category and normal/defective, seed 0. They are a subset of the 2,000 images fixed on Nov 18
   (`data/splits/test_sample2000_ids.json`). Reason: time. The 2,000-image run needs about 45 h on the laptop.
2. **The 2,000-image test is planned for the article (Dec 3)**, on Colab, with the same frozen config and the same
   Ollama model files. It is reported as the main test result if it completes; the 200-image result is then
   reported next to it. Because some answers are seen first on 200 images, **nothing may be tuned in between**.
3. **Baseline naming.** The row called "B1 Qwen3-VL-4B alone" in the documents up to Nov 19 is in fact Qwen3-VL-4B
   **with the expert's hint text and zoomed crop**. It is renamed **"Qwen + expert"**. A true single-model baseline,
   **"Qwen alone"** (no hint, no crop, no expert vote; `configs/2026-11-25_single_model_nohint.env`), is added.

## Comparisons fixed in advance (exact McNemar on paired questions)
- S vs Qwen alone: the gain of the whole system over one model.
- S vs Qwen + expert: the gain of the team over the same model with the same evidence.
- S vs B3 (Qwen + expert, 3 samples, majority vote): equal compute. Run last; reported if it finishes in time.

The success test of the plan (S beats the best single system and B3) is judged against **Qwen + expert** and B3,
the harder comparison. Beating only "Qwen alone" does not count as success.

## Secondary analysis, fixed now from dev
"S routed": the answer of Qwen + expert (before the verifier) for `defect_localization`, `defect_classification`
and `object_classification`, and the answer of S for the other subtasks. On dev: 67.2 in-sample, 66.9 in
10-fold cross-validation by image (S 66.5, Qwen + expert 65.9; median p = 0.08). Exploratory.

## Expected power (dev effect, bootstrap by image)
With 200 images the chance of a significant S > Qwen + expert (p < 0.05) is about 6 %, with 2,000 about 40 %. A
non-significant result on 200 images is therefore expected and is not evidence of no effect.
