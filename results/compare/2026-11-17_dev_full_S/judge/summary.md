| Metric | B1_qwen | S_nojudge | judge_override | judge_vote | judge_vote1.5 | override_C2 | vote1.5_C2 |
|---|---|---|---|---|---|---|---|
| Images / questions | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 |
| anomaly_discrimination | 63.2 | 61.4 | 65.0 | 65.0 | 65.0 | 65.0 | 65.0 |
| defect_classification | 46.5 | 47.2 | 46.1 | 46.1 | 46.1 | 46.1 | 46.1 |
| defect_localization | 55.0 | 50.6 | 53.1 | 53.1 | 53.1 | 53.1 | 53.1 |
| defect_description | 58.0 | 56.7 | 61.4 | 61.4 | 61.4 | 61.4 | 61.4 |
| defect_analysis | 79.5 | 80.2 | 82.7 | 82.7 | 82.7 | 82.7 | 82.7 |
| object_classification | 88.0 | 84.5 | 85.4 | 85.4 | 85.4 | 85.4 | 85.4 |
| object_analysis | 70.7 | 70.2 | 71.7 | 71.7 | 71.7 | 71.7 | 71.7 |
| **Mean of subtasks** [95 % CI] | **65.9** [64.3, 67.5] | **64.4** [62.8, 66.1] | **66.5** [64.8, 68.2] | **66.5** [64.8, 68.2] | **66.5** [64.8, 68.2] | **66.5** [64.8, 68.2] | **66.5** [64.8, 68.2] |
| FPR (good parts called defective) | 51.7 | 60.7 | 43.7 | 43.7 | 43.7 | 43.7 | 43.7 |
| Missed defects | 21.8 | 16.5 | 26.4 | 26.4 | 26.4 | 26.4 | 26.4 |
| Mean, DS-MVTec | 76.4 | 73.7 | 75.7 | 75.7 | 75.7 | 75.7 | 75.7 |
| Mean, GoodsAD | 61.5 | 59.7 | 62.9 | 62.9 | 62.9 | 62.9 | 62.9 |
| Mean, MVTec-AD | 77.8 | 88.9 | 72.2 | 72.2 | 72.2 | 72.2 | 72.2 |
| Mean, MVTec-LOCO | 56.1 | 56.8 | 58.5 | 58.5 | 58.5 | 58.5 | 58.5 |
| Mean, VisA | 69.8 | 68.1 | 69.4 | 69.4 | 69.4 | 69.4 | 69.4 |

| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |
|---|---|---|---|---|---|---|---|
| B1_qwen vs S_nojudge | 3937 | 2342 | 255 | 217 | 1123 | 71.5 | 0.0884 |
| B1_qwen vs judge_override | 3937 | 2379 | 218 | 242 | 1098 | 72.1 | 0.284 |
| B1_qwen vs judge_vote | 3937 | 2379 | 218 | 242 | 1098 | 72.1 | 0.284 |
| B1_qwen vs judge_vote1.5 | 3937 | 2379 | 218 | 242 | 1098 | 72.1 | 0.284 |
| B1_qwen vs override_C2 | 3937 | 2379 | 218 | 242 | 1098 | 72.1 | 0.284 |
| B1_qwen vs vote1.5_C2 | 3937 | 2379 | 218 | 242 | 1098 | 72.1 | 0.284 |
| S_nojudge vs judge_override | 3937 | 2369 | 190 | 252 | 1126 | 71.4 | 0.00367 |
| S_nojudge vs judge_vote | 3937 | 2369 | 190 | 252 | 1126 | 71.4 | 0.00367 |
| S_nojudge vs judge_vote1.5 | 3937 | 2369 | 190 | 252 | 1126 | 71.4 | 0.00367 |
| S_nojudge vs override_C2 | 3937 | 2369 | 190 | 252 | 1126 | 71.4 | 0.00367 |
| S_nojudge vs vote1.5_C2 | 3937 | 2369 | 190 | 252 | 1126 | 71.4 | 0.00367 |
| judge_override vs judge_vote | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_override vs judge_vote1.5 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_override vs override_C2 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_override vs vote1.5_C2 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_vote vs judge_vote1.5 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_vote vs override_C2 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_vote vs vote1.5_C2 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_vote1.5 vs override_C2 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| judge_vote1.5 vs vote1.5_C2 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
| override_C2 vs vote1.5_C2 | 3937 | 2621 | 0 | 0 | 1316 | 66.6 | 1 |
