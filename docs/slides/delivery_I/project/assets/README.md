# Slide images

The slides reference uploaded copies of these images by id (`/_blob/<id>`). This folder keeps the originals, with the id of each file so they can be matched.

| File | Blob id | Source | License |
|---|---|---|---|
| `mvtec_bottle_ref.jpg` | `4d6a9c5d27f68aee399da8830e59fe7e` | MMAD `MVTec-AD/bottle/train/good/004.png` (`random_templates[0]` of the bottle below) | CC BY-NC-SA 4.0 |
| `mvtec_bottle_part.jpg` | `a59475f857611381ceb6257ce3f906b3` | MMAD `DS-MVTec/bottle/image/broken_large/000.png` | CC BY-NC-SA 4.0 |
| `mvtec_bottle_map.jpg` | `df8cc22ae2a4799f961c87d3cabb7372` | Ours: 1-shot patch kNN (nearest-neighbour patch distance) with DINOv2 ViT-S/14 on the pair above, illustrative only | derived from CC BY-NC-SA 4.0 |
| `visa_candle_ref.jpg` | `e4981a92e43babf68c0e2a6a78eadb6a` | MMAD `VisA/candle/train/good/0102.JPG` | CC BY 4.0 |
| `visa_candle_part.jpg` | `9e2e44ddaf9e817abf26308baf42d69c` | MMAD `VisA/candle/test/bad/000.JPG` | CC BY 4.0 |
| `loco_juice_ref.jpg` | `1635b7cf89be7899985c6abe2d8a2468` | MMAD `MVTec-LOCO/juice_bottle/train/good/022.png` | CC BY-NC-SA 4.0 |
| `loco_juice_part.jpg` | `acc20f728bc67a9e117651550e472d52` | MMAD `MVTec-LOCO/juice_bottle/test/logical_anomalies/000.png` | CC BY-NC-SA 4.0 |
| `goodsad_can_ref.jpg` | `04f787fef00d0c657ce022016af1ded0` | MMAD `GoodsAD/drink_can/train/good/006_003.jpg` | Research use |
| `goodsad_can_part.jpg` | `1fd8e0b1dc1f840e41239e1d8cc46983` | MMAD `GoodsAD/drink_can/test/surface_damage/022_007.jpg` | Research use |
| `fig_mmad_subtasks.png` | `46832523e73155bf92700e9a06dbebcf` | Jiang et al., MMAD (ICLR 2025), Fig. 2 | Paper figure, cited |
| `fig_mmad_radar.png` | `72f26b1295761cee5ac91e4ef1684562` | MMAD, Fig. 1 right (not used in the deck yet) | Paper figure, cited |
| `fig_mmad_sunburst.png` | `c1fc776b6b157e3c7a78381c234f2e87` | MMAD, Fig. 1 left | Paper figure, cited |
| `fig_sigma_overview.jpg` | `42b765089dcdffc977fd9e275d197bdb` | SiGMA, arXiv 2610.12310, Fig. 2 | Paper figure, cited |
| `fig_orch_roles.png` | `6b4c1da2707cd7da5f6c56900b692e41` | Adimulam et al., arXiv 2601.13671, Fig. 2 | Paper figure, cited |
| `fig_orch_evolution.png` | `76aa25c803e7e27fcc61e92b7ef4253b` | Adimulam et al., Fig. 1 (not used in the deck yet) | Paper figure, cited |

**How the MMAD images were fetched:** only these files were read out of the zips on Hugging Face (`jiang-cc/MMAD`, `main`) with HTTP range requests, so the full 28 GB was not downloaded. The questions and answers quoted on the slides come from `mmad.json` (SHA-256 `c219c6b0…08143`, the corrected revision used by the project).
