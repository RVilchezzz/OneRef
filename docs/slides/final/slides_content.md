# Slide skeleton (final delivery, Nov 26)

The plan is 13 slides, about 15 minutes. The length has not been confirmed: ask at the monitoring session and cut slides 5
and 11 if it is 10 minutes.

Format: the same as Delivery II.
- HTML fragments on a 1920×1080 canvas in `docs/slides/final/project/`, built from this file.
- Same palette and fonts: Source Serif 4 + IBM Plex Sans, navy `#15253A`, cream `#F6F4EF`, rust `#B04A17`.

`TBD(test)` = number filled from `results/test/frozen_2026-11-19/`. `[fig]` = figure still to make. Speaker = who presents.

| # | id | Title (claim) | Content | Figure / source | Speaker |
|---|---|---|---|---|---|
| 1 | `cover` | OneRef: final results | Team, date, one-line answer | Part + expert map (reuse from Delivery II) | Peter |
| 2 | `pergunta` | Does a team of small models beat one model, with one reference? | The task in one picture; the strict protocol; hypothesis + success test (McNemar p < 0.05 vs B1 and B3) | Reference / part / question (reuse from Delivery I `descricao`) | Peter |
| 3 | `mudancas` | What changed since the plan | 8B → 4B on one laptop GPU; Colab dropped; judge added on dev; test = 2,000-image sample | Small "plan → done" table | Peter |
| 4 | `sistema` | Five agents, one fixed workflow | The pipeline with the judge; calls per question 2.64 vs 1 (B1) vs 3 (B3) | `[fig]` Updated pipeline diagram | Charles |
| 5 | `experto` | The expert sees structural defects, and is blind to logic and to pose | AUROC per source; hint vs no hint +6.8 | Hit/miss maps (Delivery II assets) | Charles |
| 6 | `protocolo` | One test run, frozen beforehand | Dev 10 % / test 90 %; 2,000 images, 9,490 questions; frozen config; official metric checked to 2 decimals | Split diagram | Charles |
| 7 | `resultado` | **Main result**: S `TBD(test)` vs B1 `TBD(test)` (p = `TBD(test)`) | Bars with CIs: B2, B3, B1, S; dashed external refs (8B 73.9, GPT-4o 74.9, humans 86.7). Dev values in grey | `[fig]` From `summary.json` | Peter |
| 8 | `tradeoff` | The judge trades false alarms for misses | FPR vs missed, per system (dev: S 43.7 / 26.4 vs B1 51.7 / 21.8) | `[fig]` Scatter FPR × missed | Peter |
| 9 | `ablacoes` | Only the judge adds a significant gain | Ablation table with p-values: no judge, no verifier, no expert vote; B3 ≈ B1 → gain from heterogeneity, not from more calls | Table | Charles |
| 10 | `fontes` | Where it works and where it doesn't | Per source × system; LOCO and GoodsAD lowest | `[fig]` Heatmap source × system | Charles |
| 11 | `erros` | 1/3 of the failures had the right answer somewhere | Trace categories (dev: 26 % judge, 8 % reopen, 66 % nobody) + manual labels on 100 test failures; 2 illustrated cases | `[fig]` Stacked bar + 2 cases from `sheet.html` | Both |
| 12 | `conclusao` | Answer to the question | One sentence answer; 3 lessons (heterogeneity > repetition; verify with evidence, not opinion; the expert is useful as a hint); comparison with SiGMA's advantages; future work | Text | Peter |
| 13 | `obrigado` | Thank you, questions | Repo link, reproducibility (`run_test.sh`, frozen env) | — | Both |

## Backup slides (for Q&A)
- B1. Per-subtask table (test).
- B2. Cost: calls, tokens, wall-clock per image.
- B3. The prompts of the reader and the judge.
- B4. The official-metric table per dataset.
- B5. Threats to validity: sample vs full split, 4-bit quantization, judge designed on dev.

## Figures to generate (script: `scripts/make_figures.py`, to do after the test)
1. Means with CIs, dev + test: slide 7, report §6.1.
2. FPR × missed scatter: slide 8.
3. Source × system heatmap: slide 10, report §6.2.
4. Error categories stacked bar: slide 11, report §6.4.
5. Updated pipeline diagram (manual): slide 4, report §3.1.
