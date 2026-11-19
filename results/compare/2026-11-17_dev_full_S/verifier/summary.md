| Metric | gemma | rules | noverif |
|---|---|---|---|
| Images / questions | 835 / 3937 | 835 / 3937 | 835 / 3937 |
| anomaly_discrimination | 61.4 | 61.3 | 62.3 |
| defect_classification | 47.2 | 45.0 | 45.0 |
| defect_localization | 50.6 | 51.0 | 52.9 |
| defect_description | 56.7 | 56.3 | 56.3 |
| defect_analysis | 80.2 | 80.2 | 80.2 |
| object_classification | 84.5 | 82.8 | 82.8 |
| object_analysis | 70.2 | 69.9 | 69.9 |
| **Mean of subtasks** [95 % CI] | **64.4** [62.8, 66.1] | **63.8** [62.1, 65.5] | **64.2** [62.6, 65.9] |
| FPR (good parts called defective) | 60.7 | 61.0 | 51.7 |
| Missed defects | 16.5 | 16.5 | 23.6 |
| Mean, DS-MVTec | 73.7 | 72.6 | 73.5 |
| Mean, GoodsAD | 59.7 | 60.1 | 60.8 |
| Mean, MVTec-AD | 88.9 | 88.9 | 88.9 |
| Mean, MVTec-LOCO | 56.8 | 55.0 | 55.2 |
| Mean, VisA | 68.1 | 67.8 | 67.8 |

| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |
|---|---|---|---|---|---|---|---|
| gemma vs rules | 3937 | 2497 | 62 | 43 | 1335 | 66.1 | 0.0785 |
| gemma vs noverif | 3937 | 2464 | 95 | 79 | 1299 | 67.0 | 0.255 |
| rules vs noverif | 3937 | 2499 | 41 | 44 | 1353 | 65.6 | 0.828 |
