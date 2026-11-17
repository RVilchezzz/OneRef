| Metric | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| Images / questions | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 |
| anomaly_discrimination | 61.4 | 62.0 | 63.0 | 62.9 |
| defect_classification | 47.2 | 47.2 | 47.2 | 47.2 |
| defect_localization | 50.6 | 50.6 | 50.6 | 50.6 |
| defect_description | 56.7 | 56.7 | 56.7 | 56.7 |
| defect_analysis | 80.2 | 80.2 | 80.2 | 80.2 |
| object_classification | 84.5 | 84.5 | 84.5 | 84.5 |
| object_analysis | 70.2 | 70.2 | 70.2 | 70.2 |
| **Mean of subtasks** [95 % CI] | **64.4** [62.8, 66.1] | **64.5** [62.9, 66.2] | **64.6** [63.0, 66.3] | **64.6** [63.0, 66.3] |
| FPR (good parts called defective) | 60.7 | 58.2 | 55.1 | 51.1 |
| Missed defects | 16.5 | 17.9 | 18.8 | 23.2 |
| Mean, DS-MVTec | 73.7 | 73.7 | 74.1 | 74.9 |
| Mean, GoodsAD | 59.7 | 59.9 | 59.9 | 59.9 |
| Mean, MVTec-AD | 88.9 | 88.9 | 88.9 | 88.9 |
| Mean, MVTec-LOCO | 56.8 | 56.8 | 57.2 | 57.2 |
| Mean, VisA | 68.1 | 68.1 | 68.3 | 68.3 |

| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |
|---|---|---|---|---|---|---|---|
| C0 vs C1 | 3937 | 2552 | 7 | 8 | 1370 | 65.2 | 1 |
| C0 vs C2 | 3937 | 2538 | 21 | 27 | 1351 | 65.7 | 0.471 |
| C0 vs C3 | 3937 | 2516 | 43 | 40 | 1338 | 66.0 | 0.826 |
| C1 vs C2 | 3937 | 2546 | 14 | 19 | 1358 | 65.5 | 0.487 |
| C1 vs C3 | 3937 | 2524 | 36 | 32 | 1345 | 65.8 | 0.716 |
| C2 vs C3 | 3937 | 2543 | 22 | 13 | 1359 | 65.5 | 0.175 |
