| Metric | hint | nohint |
|---|---|---|
| Images / questions | 82 / 394 | 82 / 394 |
| anomaly_discrimination | 57.6 | 56.3 |
| defect_classification | 43.2 | 40.5 |
| defect_localization | 68.4 | 39.5 |
| defect_description | 66.7 | 66.7 |
| defect_analysis | 75.0 | 67.5 |
| object_classification | 92.5 | 82.5 |
| object_analysis | 72.7 | 75.2 |
| **Mean of subtasks** [95 % CI] | **68.0** [62.6, 73.6] | **61.2** [56.3, 66.5] |
| FPR (good parts called defective) | 52.4 | 2.4 |
| Missed defects | 32.5 | 85.0 |
| Mean, DS-MVTec | 79.7 | 70.7 |
| Mean, GoodsAD | 72.0 | 58.8 |
| Mean, MVTec-AD | 94.4 | 94.4 |
| Mean, MVTec-LOCO | 52.0 | 49.3 |
| Mean, VisA | 67.9 | 64.3 |

| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |
|---|---|---|---|---|---|---|---|
| hint vs nohint | 394 | 214 | 54 | 38 | 88 | 77.7 | 0.117 |
