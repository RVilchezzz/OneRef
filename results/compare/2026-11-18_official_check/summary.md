| Run | Questions | Invalid | Official (drops invalid) | Official, invalid = wrong | Ours (pooled) | Label mismatches |
|---|---|---|---|---|---|---|
| S_frozen | 3937 | 0 | **66.6** | 66.6 | 66.5 | 0 |
| B1_qwen | 3937 | 0 | **66.0** | 66.0 | 65.9 | 0 |
| B2_internvl | 3937 | 8 | **60.3** | 60.1 | 60.2 | 0 |
| S_nojudge | 3937 | 0 | **64.6** | 64.6 | 64.4 | 0 |

Official table, S_frozen (per dataset; DS-MVTec + MVTec-AD merged):

| Dataset | Anomaly Detection | Defect Classification | Defect Description | Defect Analysis | Defect Localization | Object Classification | Object Analysis | Average | Overkill | Miss |
|---|---|---|---|---|---|---|---|---|---|---|
| MVTec-AD | 79.0 | 60.3 | 68.0 | 87.7 | 64.7 | 89.1 | 80.6 | 75.6 | 30.4 | 11.5 |
| MVTec-LOCO | 58.4 | 26.0 | 49.0 | 71.0 | 54.2 | 81.1 | 70.1 | 58.5 | 43.9 | 39.4 |
| VisA | 74.2 | 47.1 | 63.9 | 82.1 | 55.0 | 91.4 | 72.3 | 69.4 | 11.6 | 40.0 |
| GoodsAD | 53.8 | 46.8 | 62.1 | 86.6 | 41.5 | 81.2 | 68.3 | 62.9 | 72.8 | 19.6 |

Official table, B1_qwen (per dataset; DS-MVTec + MVTec-AD merged):

| Dataset | Anomaly Detection | Defect Classification | Defect Description | Defect Analysis | Defect Localization | Object Classification | Object Analysis | Average | Overkill | Miss |
|---|---|---|---|---|---|---|---|---|---|---|
| MVTec-AD | 76.6 | 63.6 | 66.4 | 86.9 | 68.1 | 91.3 | 82.1 | 76.4 | 37.0 | 9.8 |
| MVTec-LOCO | 55.0 | 20.8 | 46.9 | 65.6 | 56.2 | 83.0 | 65.2 | 56.1 | 45.6 | 44.4 |
| VisA | 75.4 | 52.9 | 61.3 | 77.8 | 55.8 | 94.6 | 70.6 | 69.8 | 15.8 | 33.3 |
| GoodsAD | 52.1 | 43.7 | 55.3 | 83.8 | 42.9 | 83.8 | 68.9 | 61.5 | 87.2 | 8.6 |

Official table, B2_internvl (per dataset; DS-MVTec + MVTec-AD merged):

| Dataset | Anomaly Detection | Defect Classification | Defect Description | Defect Analysis | Defect Localization | Object Classification | Object Analysis | Average | Overkill | Miss |
|---|---|---|---|---|---|---|---|---|---|---|
| MVTec-AD | 66.5 | 50.0 | 61.5 | 85.2 | 64.4 | 62.2 | 74.6 | 66.4 | 17.8 | 49.2 |
| MVTec-LOCO | 53.6 | 25.0 | 36.5 | 68.8 | 50.0 | 66.0 | 69.1 | 52.7 | 40.4 | 52.5 |
| VisA | 63.8 | 47.1 | 64.4 | 82.1 | 52.9 | 77.4 | 68.6 | 65.2 | 3.2 | 69.2 |
| GoodsAD | 55.2 | 43.7 | 54.5 | 79.6 | 38.1 | 65.8 | 60.5 | 56.8 | 42.4 | 47.2 |

Official table, S_nojudge (per dataset; DS-MVTec + MVTec-AD merged):

| Dataset | Anomaly Detection | Defect Classification | Defect Description | Defect Analysis | Defect Localization | Object Classification | Object Analysis | Average | Overkill | Miss |
|---|---|---|---|---|---|---|---|---|---|---|
| MVTec-AD | 71.4 | 59.5 | 69.7 | 86.9 | 63.0 | 89.1 | 77.6 | 73.9 | 45.7 | 11.5 |
| MVTec-LOCO | 57.2 | 27.1 | 41.7 | 68.8 | 49.0 | 83.0 | 70.7 | 56.8 | 61.4 | 24.2 |
| VisA | 72.8 | 53.8 | 59.7 | 77.8 | 52.5 | 88.2 | 71.6 | 68.1 | 16.8 | 37.5 |
| GoodsAD | 50.4 | 44.4 | 53.0 | 83.8 | 40.1 | 80.3 | 65.9 | 59.7 | 99.2 | 0.0 |
