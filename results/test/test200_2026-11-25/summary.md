| Metric | S | qwen_alone | qwen_expert | S_nojudge | internvl_expert | B3_k3 |
|---|---|---|---|---|---|---|
| Images / questions | 200 / 952 | 200 / 952 | 200 / 952 | 200 / 952 | 200 / 952 | 200 / 952 |
| anomaly_discrimination | 67.2 | 58.5 | 65.1 | 66.1 | 62.1 | 64.5 |
| defect_classification | 52.3 | 50.5 | 57.8 | 51.4 | 44.0 | 56.9 |
| defect_localization | 52.6 | 32.5 | 57.0 | 55.3 | 50.9 | 57.0 |
| defect_description | 65.5 | 64.6 | 64.6 | 61.1 | 56.6 | 61.9 |
| defect_analysis | 89.5 | 82.5 | 85.1 | 81.6 | 80.7 | 85.1 |
| object_classification | 87.2 | 85.9 | 87.2 | 88.5 | 71.8 | 87.2 |
| object_analysis | 76.4 | 77.3 | 75.6 | 77.3 | 72.0 | 75.1 |
| **Mean of subtasks** [95 % CI] | **70.1** [66.9, 73.2] | **64.5** [61.5, 67.6] | **70.3** [67.0, 73.5] | **68.7** [65.0, 72.1] | **62.6** [59.3, 65.9] | **69.7** [66.3, 72.8] |
| FPR (good parts called defective) | 48.1 | 1.3 | 53.2 | 57.0 | 31.6 | 54.4 |
| Missed defects | 17.5 | 81.7 | 16.7 | 10.8 | 44.2 | 16.7 |
| Mean, DS-MVTec | 82.2 | 75.3 | 85.7 | 81.8 | 72.6 | 85.2 |
| Mean, GoodsAD | 63.0 | 59.2 | 60.5 | 61.8 | 57.2 | 59.9 |
| Mean, MVTec-AD | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Mean, MVTec-LOCO | 59.4 | 58.5 | 61.8 | 57.7 | 54.9 | 60.3 |
| Mean, VisA | 76.0 | 66.0 | 74.4 | 74.4 | 65.3 | 74.3 |

| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |
|---|---|---|---|---|---|---|---|
| S vs qwen_alone | 952 | 507 | 166 | 93 | 186 | 80.5 | 6.73e-06 |
| S vs qwen_expert | 952 | 629 | 44 | 44 | 235 | 75.3 | 1 |
| S vs S_nojudge | 952 | 622 | 51 | 43 | 236 | 75.2 | 0.47 |
| S vs internvl_expert | 952 | 543 | 130 | 58 | 221 | 76.8 | 1.6e-07 |
| S vs B3_k3 | 952 | 625 | 48 | 42 | 237 | 75.1 | 0.598 |
| qwen_alone vs qwen_expert | 952 | 515 | 85 | 158 | 194 | 79.6 | 3.29e-06 |
| qwen_alone vs S_nojudge | 952 | 488 | 112 | 177 | 175 | 81.6 | 0.000157 |
| qwen_alone vs internvl_expert | 952 | 452 | 148 | 149 | 203 | 78.7 | 1 |
| qwen_alone vs B3_k3 | 952 | 511 | 89 | 156 | 196 | 79.4 | 2.22e-05 |
| qwen_expert vs S_nojudge | 952 | 620 | 53 | 45 | 234 | 75.4 | 0.48 |
| qwen_expert vs internvl_expert | 952 | 511 | 162 | 90 | 189 | 80.1 | 6.75e-06 |
| qwen_expert vs B3_k3 | 952 | 665 | 8 | 2 | 277 | 70.9 | 0.109 |
| S_nojudge vs internvl_expert | 952 | 540 | 125 | 61 | 226 | 76.3 | 3.11e-06 |
| S_nojudge vs B3_k3 | 952 | 615 | 50 | 52 | 235 | 75.3 | 0.921 |
| internvl_expert vs B3_k3 | 952 | 511 | 90 | 156 | 195 | 79.5 | 3.08e-05 |
