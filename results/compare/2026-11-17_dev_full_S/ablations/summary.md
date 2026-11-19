| Metric | S | B1_qwen | B2_internvl | AB_noverif | AB_noverif_noexpertvote | S_noexpertvote | S_A_only |
|---|---|---|---|---|---|---|---|
| Images / questions | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 | 835 / 3937 |
| anomaly_discrimination | 61.4 | 63.2 | 59.5 | 62.3 | 59.9 | 61.9 | 62.8 |
| defect_classification | 47.2 | 46.5 | 42.2 | 45.0 | 45.0 | 47.2 | 48.1 |
| defect_localization | 50.6 | 55.0 | 50.6 | 52.9 | 52.9 | 50.6 | 52.9 |
| defect_description | 56.7 | 58.0 | 55.0 | 56.3 | 56.3 | 56.7 | 57.6 |
| defect_analysis | 80.2 | 79.5 | 79.5 | 80.2 | 80.2 | 80.2 | 79.5 |
| object_classification | 84.5 | 88.0 | 68.6 | 82.8 | 82.8 | 84.5 | 88.3 |
| object_analysis | 70.2 | 70.7 | 66.2 | 69.9 | 69.9 | 70.2 | 70.9 |
| **Mean of subtasks** [95 % CI] | **64.4** [62.8, 66.1] | **65.9** [64.3, 67.5] | **60.2** [58.5, 61.9] | **64.2** [62.6, 65.9] | **63.9** [62.2, 65.6] | **64.5** [62.8, 66.2] | **65.7** [64.1, 67.3] |
| FPR (good parts called defective) | 60.7 | 51.7 | 26.9 | 51.7 | 41.8 | 52.6 | 61.9 |
| Missed defects | 16.5 | 21.8 | 54.0 | 23.6 | 38.5 | 23.6 | 12.5 |
| Mean, DS-MVTec | 73.7 | 76.4 | 65.7 | 73.5 | 72.7 | 74.6 | 76.1 |
| Mean, GoodsAD | 59.7 | 61.5 | 56.8 | 60.8 | 61.2 | 59.9 | 60.4 |
| Mean, MVTec-AD | 88.9 | 77.8 | 88.9 | 88.9 | 88.9 | 88.9 | 77.8 |
| Mean, MVTec-LOCO | 56.8 | 56.1 | 52.7 | 55.2 | 54.6 | 56.6 | 57.5 |
| Mean, VisA | 68.1 | 69.8 | 65.0 | 67.8 | 67.0 | 67.9 | 69.9 |

| Pair | Paired Qs | Both right | Only first | Only second | Both wrong | Oracle acc. | McNemar p |
|---|---|---|---|---|---|---|---|
| S vs B1_qwen | 3937 | 2342 | 217 | 255 | 1123 | 71.5 | 0.0884 |
| S vs B2_internvl | 3937 | 2077 | 482 | 282 | 1096 | 72.2 | 4.47e-13 |
| S vs AB_noverif | 3937 | 2464 | 95 | 79 | 1299 | 67.0 | 0.255 |
| S vs AB_noverif_noexpertvote | 3937 | 2387 | 172 | 113 | 1265 | 67.9 | 0.000568 |
| S vs S_noexpertvote | 3937 | 2521 | 38 | 28 | 1350 | 65.7 | 0.268 |
| S vs S_A_only | 3937 | 2425 | 134 | 184 | 1194 | 69.7 | 0.00591 |
| B1_qwen vs B2_internvl | 3937 | 1957 | 640 | 402 | 938 | 76.2 | 1.66e-13 |
| B1_qwen vs AB_noverif | 3937 | 2384 | 213 | 159 | 1181 | 70.0 | 0.00592 |
| B1_qwen vs AB_noverif_noexpertvote | 3937 | 2324 | 273 | 176 | 1164 | 70.4 | 5.44e-06 |
| B1_qwen vs S_noexpertvote | 3937 | 2330 | 267 | 219 | 1121 | 71.5 | 0.0329 |
| B1_qwen vs S_A_only | 3937 | 2509 | 88 | 100 | 1240 | 68.5 | 0.422 |
| B2_internvl vs AB_noverif | 3937 | 2116 | 243 | 427 | 1151 | 70.8 | 1.13e-12 |
| B2_internvl vs AB_noverif_noexpertvote | 3937 | 2133 | 226 | 367 | 1211 | 69.2 | 7.67e-09 |
| B2_internvl vs S_noexpertvote | 3937 | 2100 | 259 | 449 | 1129 | 71.3 | 8.97e-13 |
| B2_internvl vs S_A_only | 3937 | 1957 | 402 | 652 | 926 | 76.5 | 1.3e-14 |
| AB_noverif vs AB_noverif_noexpertvote | 3937 | 2466 | 77 | 34 | 1360 | 65.5 | 5.49e-05 |
| AB_noverif vs S_noexpertvote | 3937 | 2449 | 94 | 100 | 1294 | 67.1 | 0.72 |
| AB_noverif vs S_A_only | 3937 | 2340 | 203 | 269 | 1125 | 71.4 | 0.00273 |
| AB_noverif_noexpertvote vs S_noexpertvote | 3937 | 2415 | 85 | 134 | 1303 | 66.9 | 0.00113 |
| AB_noverif_noexpertvote vs S_A_only | 3937 | 2265 | 235 | 344 | 1093 | 72.2 | 6.77e-06 |
| S_noexpertvote vs S_A_only | 3937 | 2389 | 160 | 220 | 1168 | 70.3 | 0.00243 |
