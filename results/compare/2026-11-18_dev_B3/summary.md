| Metric | S_judge | B1_qwen | B3_k3 |
|---|---|---|---|
| Images / questions | 835 / 3937 | 835 / 3937 | 835 / 3937 |
| anomaly_discrimination | 65.0 | 63.2 | 62.3 |
| defect_classification | 46.1 | 46.5 | 45.9 |
| defect_localization | 53.1 | 55.0 | 55.0 |
| defect_description | 61.4 | 58.0 | 58.0 |
| defect_analysis | 82.7 | 79.5 | 78.9 |
| object_classification | 85.4 | 88.0 | 88.3 |
| object_analysis | 71.7 | 70.7 | 71.0 |
| **Mean of subtasks** [95 % CI] | **66.5** [64.8, 68.2] | **65.9** [64.3, 67.5] | **65.6** [64.0, 67.3] |
| FPR (good parts called defective) | 43.7 | 51.7 | 52.9 |
| Missed defects | 26.4 | 21.8 | 22.4 |
| Mean, DS-MVTec | 75.7 | 76.4 | 75.9 |
| Mean, GoodsAD | 62.9 | 61.5 | 61.5 |
| Mean, MVTec-AD | 72.2 | 77.8 | 77.8 |
| Mean, MVTec-LOCO | 58.5 | 56.1 | 55.8 |
| Mean, VisA | 69.4 | 69.8 | 69.5 |

| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |
|---|---|---|---|---|---|---|---|
| S_judge vs B1_qwen | 3937 | 2379 | 242 | 218 | 1098 | 72.1 | 0.284 |
| S_judge vs B3_k3 | 3937 | 2365 | 256 | 223 | 1093 | 72.2 | 0.144 |
| B1_qwen vs B3_k3 | 3937 | 2560 | 37 | 28 | 1312 | 66.7 | 0.321 |
