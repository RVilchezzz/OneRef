# Slides

| Path | Content |
|---|---|
| `project/deck.json` | Slide order, sections and fonts (Source Serif 4 + IBM Plex Sans) |
| `project/slides/<id>.html` | One file per slide |
| `project/assets/` | Copies of the images the slides show, with their sources and licenses (`assets/README.md`) |
| `slides_content.md` | Plain text of every slide and its speaker notes, for quick reading |
| `delivery_I_plan.pptx` | PPTX export of the first version (9 slides, text only) |
| `speaker_script_delivery_I.en.md` / `.fr.md` | Word-for-word 15-min script per slide (EN / FR), speaker split, slide improvements and jury Q&A |
| `OneRef – Preliminary Results (Delivery II).pdf` | Delivery II deck (11 slides, Nov 12) |
| `speaker_script_delivery_II.en.md` / `.fr.md` / `.es.md` | Word-for-word 10-min script for the Delivery II deck (EN / FR / ES), speaker split, slide improvements and jury Q&A |
| `charles_part_delivery_II.fr.md` | Charles's part of Delivery II in detail (FR): what the GPU-free tests, the 18-image run and the four cases actually did, limits, jury Q&A |
| `../../../IA025 – Final Project Plan (Delivery I) – notes EN.pptx` / `notes FR.pptx` | Copies of the final deck with the script in the speaker notes (project folder, outside the repo) |

**Slide order:**
1. `cover`: title, with a real good/defective MMAD pair
2. `descricao` (description): the task as a picture (reference + part + real question), accuracy gap, hypothesis
3. `motivacao` (motivation): MMAD result "MLLMs do better when told where to look", SiGMA figure and how our setup differs
4. `metodologia` (methodology): one real image through the 5-step workflow, with our DINOv2 anomaly map
5. `orquestracao` (orchestration): Adimulam et al. roles figure mapped to our agents
6. `hardware`: local GPU ↔ Drive ↔ Colab diagram and pass-based execution
7. `datasets`: MMAD composition figure, sources table, dev/test split
8. `dados` (data): real reference/defect pairs from the 4 sources
9. `subtarefas` (subtasks): MMAD example questions and counts per subtask
10. `metricas` (metrics): baseline ladder B0 to S and metric cards
11. `resultados` (results): leaderboard bar chart with the 73.9% bar to beat
12. `referencias` (references)
13. `cronograma` (schedule): timeline with the Nov 19 freeze
14. `obrigado` (thank you): questions and repo link

The content is all in English. The ids are leftovers from the first draft; they are internal names only.

**Format.** One HTML fragment per slide on a fixed 1920×1080 canvas, plus `deck.json` for the order and fonts.
- The HTML files are slide fragments, not standalone pages: opening one in a browser shows the content but not the deck viewer.
- `slides_content.md` has the plain text and speaker notes of every slide, usable as a script for any other format (Marp, reveal.js, PowerPoint).

**Known to-dos:**
- `metodologia`: says "no model training". If the optional small comparator head on DINOv3 is adopted, this must change.
- `resultados`: to be replaced with real results for Delivery II.
- `metodologia`: the anomaly map is an illustrative DINOv2 run made for the slides; replace it with the real DINOv3 expert's output for Delivery II.
- Export a PPTX of this version and save it as `delivery_I_plan_v2.pptx`.
