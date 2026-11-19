# Slide content (Delivery I)

Plain text of each slide, in `deck.json` order. The editable source is `project/slides/<id>.html`.


## 1. `cover`

- IA025 · Deep Learning · Unicamp 2026
- OneRef
- Multi-agent visual inspection from a single reference: a team of pre-trained models finds defects in industrial parts from one photo of a good part, validated on the MMAD benchmark
- Team: Peter Roque Vilchez · Charles Maxime Doudy
- Delivery I · October 29, 2026
- Reference
- Part to inspect
- [image: Good bottle seen from above (reference, MVTec AD)]
- [image: Bottle with a broken rim (defective part, MVTec AD)]

> Speaker notes: IA025 final project. Course topic: multi-agent systems and agent orchestration, using pre-trained models as agents with validation of the results. The two photos on the right are real MMAD images: a good bottle (the only reference the system gets) and a bottle with a broken rim.

## 2. `descricao`

- Project description
- The task: one good photo, one part, a question
- Reference (good)
- Part to inspect
- Where is the defect?
- A. Top right / B. Top left / C. Center / D. Bottom right ✓
- Hypothesis
- A training-free team (a visual expert, two MLLM readers and a verifier) beats the best single model and a same-cost majority vote.
- Mean MMAD accuracy (axis from 50%)
- Expert human inspectors
- 86.7%
- GPT-4o (closed)
- 74.9%
- Qwen3-VL-8B (open): our baseline
- 73.9%
- A 13-point gap to humans: that is the room we try to close.
- Real MMAD example (MVTec AD bottle, question and answer from mmad.json). Accuracy: Jiang et al., ICLR 2025; official MMAD leaderboard, Oct 2026.
- [image: Good bottle, reference]
- [image: Bottle with broken rim]

> Speaker notes: This is exactly what the system receives: one photo of a good part (the reference), the part to inspect, and multiple-choice questions. This bottle has 5 questions in MMAD: is there a defect, which type, where, what it looks like, and its effect. When a production line switches products there is no time or data to train a detector, only one good photo. MLLMs (multimodal large language models, LLMs that also see images) can do this from instructions, but they are still 13 points below expert inspectors. The hypothesis is measurable: we compare against the best single model and against voting with the same number of calls.

## 3. `motivacao`

- Motivation
- MLLMs reason well once told where to look
- MMAD, InternVL2-40B: accuracy with different “experts”
- Anomaly discrimination
- MLLM alone
- 63.8
- + PatchCore map
- 60.0
- + AnomalyCLIP map
- 59.6
- + true location
- 88.7
- Defect localization
- MLLM alone
- 52.9
- + best detector
- 60.4
- + true location
- 81.0
- Closest work: SiGMA (Oct 2026), 85.2% with a DINOv3 expert
- Condition
- SiGMA
- Ours
- Normal images
- up to 256
- 1
- Tuned on MMAD
- yes, team + params
- dev split only
- GPUs
- 4 × L20
- Colab L4 / T4
- Left: Jiang et al., MMAD (ICLR 2025), Table 6; “true location” = ground-truth mask given as text (upper bound). Right: SiGMA, arXiv 2610.12310, Fig. 2.
- [image: SiGMA overview: searcher, MLLM agents, DINOv3 defect expert and reliability controller]

> Speaker notes: This is why the core agent is a visual expert. In MMAD, when the MLLM is told where the defect is (ground truth turned into text), discrimination jumps by about 25 points and localization by 28. Off-the-shelf anomaly detectors (PatchCore, AnomalyCLIP) are not precise enough and even hurt discrimination, because they also fire on normal images. SiGMA shows a multi-agent team with a DINOv3 expert reaches 85.2%, but with up to 256 normal images, retrieved domain text, a team chosen on MMAD answers and 4 GPUs. Our question: how much of that gain survives with one reference photo and no tuning on the test?

## 4. `metodologia`

- Methodology
- One image through the workflow
- Input
- 1 reference + the part
- “Where is the defect?” A / B / C / D
- 1 · Visual expert
- Compares patches with the reference → map, 3×3 region, crop, score
- 2 · Two readers
- Qwen3-VL-8B
- answer · confidence
- InternVL3-8B
- answer · confidence
- Same photos + the expert’s hint; JSON reply
- 3 · Verifier
- ✓ valid schema / ✓ answers consistent / ✓ agrees with expert
- Conflict → reopen once, with the evidence
- 4 · Aggregator
- Weighted vote per reader and subtask; the expert votes on “defect?”
- Final answer + report
- 5 · Log
- Every message, time and token count saved as JSONL per image. No model is trained; the Python orchestrator runs this fixed workflow.
- Real MMAD pair. Map: our illustrative 1-shot patch comparison with DINOv2 ViT-S/14 on this pair (the project will use DINOv3). True answer: bottom right.
- [image: Reference: good bottle]
- [image: Part: broken bottle]
- [image: Anomaly map over the bottle: hot region on the broken rim, bottom right]

> Speaker notes: Walk through one real image. Input: the good bottle and the broken one, plus the question. Step 1, the visual expert (frozen DINOv3) splits both photos into small patches and, for each patch of the part, measures how far it is from the most similar patch of the reference. The heat map you see was computed by us on this exact pair with DINOv2, and it lights up the broken rim. The peak is turned into text, a cell of a 3×3 grid such as "bottom right, high score", and a zoomed crop. (On this pair our quick DINOv2 version put the peak in the bottom-center cell, next to the true bottom-right answer: calibrating this is part of Delivery II.) Step 2, two readers from different model families answer in JSON. Step 3, the verifier checks the format, consistency and agreement with the expert, and reopens at most once to avoid loops. Step 4, the aggregator votes with weights fixed on the dev split. Step 5, everything is logged.

## 5. `orquestracao`

- Methodology · orchestration
- Each agent has one role in the framework
- Adimulam et al., arXiv 2601.13671, Fig. 2
- Orchestration layer
- Python orchestrator: fixed plan per subtask
- Worker agents
- Visual expert · Reader A · Reader B
- Service agent (quality control)
- Verifier: schema, consistency, reopen once
- Support · state
- Aggregator + JSONL log of every message
- Design lessons: start with a simple fixed workflow (Anthropic, 2024); multi-agent gains are often small, so every agent must earn its place in an ablation (MAST, NeurIPS 2025).
- [image: Orchestration layer assigning tasks to worker, service and support agents]

> Speaker notes: The attached paper (Adimulam et al.) is a position paper: it gives the vocabulary, not experiments. We take three ideas from it: the roles (worker, service, support agents), an orchestration layer that decides who does what and in which order, and validating every output against a schema. On the right is how our five components map onto those roles. Two lessons from the literature keep the design honest: Anthropic recommends starting with a simple fixed workflow, and MAST (a study of why multi-agent systems fail) shows gains over a single agent or best-of-N are often minimal, so each agent will be removed in an ablation to prove it helps.

## 6. `hardware`

- Compute resources
- Yes: a laptop GPU, Colab and a shared cache
- Local · Peter
- RTX 3060 Laptop, 6 GB
- DINOv3 visual expert; debug the whole pipeline with Qwen3-VL-4B in 4-bit
- Shared
- Google Drive
- MMAD images + one JSONL cache per agent, resumable
- Cloud · Charles
- Colab Pro, L4 / T4
- 8B readers served with vLLM, baselines and final test (100 units, no A100)
- Pass-based execution: one model on the GPU at a time
- Expert pass
- Reader A pass
- Reader B pass
- Verifier pass
- Aggregate
- Budget: cost measured on 50 questions before the test; baselines B1, B2 and B4 reuse the cached reader answers.

> Speaker notes: We do have hardware. Locally, a 6 GB laptop GPU is enough for the DINOv3 expert and for debugging the full pipeline with a small 4B model in 4-bit. The two 8B readers run on Colab Pro with vLLM, a fast batched inference server, on L4 or T4 GPUs; we avoid the A100 because it burns compute units 4 to 5 times faster. Each agent runs over the whole split, saves its answers to Drive and frees the GPU, so two 8B models are never loaded together and a disconnect does not lose work. Charles has no local GPU and works directly on Colab.

## 7. `datasets`

- Datasets
- MMAD: 8,366 images, 39,670 questions
- Jiang et al., MMAD (ICLR 2025), Fig. 1
- Source
- Normal
- Defective
- License
- MVTec AD
- 465
- 1,226
- CC BY-NC-SA 4.0
- VisA
- 943
- 1,198
- CC BY 4.0
- MVTec LOCO
- 573
- 993
- CC BY-NC-SA 4.0
- GoodsAD
- 1,309
- 1,659
- Research use
- Our split (fixed seed, stratified)
- Dev
- Test pool · 90%
- Dev 10% (~840 images): prompts and weights are chosen here. Test: ~2,000 images from the 90%, run once with the frozen config.
- Use: evaluation only, no training. github.com/jam-cc/MMAD · huggingface.co/datasets/jiang-cc/MMAD (corrected revision, Oct 22, 2026).
- [image: MMAD composition: normal and abnormal images, 7 subtasks, 38 categories from 4 datasets]

> Speaker notes: MMAD is a public benchmark (ICLR 2025) built on four industrial inspection datasets with 38 product categories. The figure shows its composition: normal and abnormal images in the center, the 7 subtasks in the middle ring and the categories outside. It comes fully labeled, so we annotate nothing; labels are only used to score answers. MMAD is test-only, so we make our own split: 10% for development, where we are allowed to look at answers while choosing prompts and weights, and a test sample from the rest, run once after the configuration is frozen on Nov 19. Limitation: one view per part; Real-IAD (5 views) is an optional extension.

## 8. `dados`

- Datasets · what the system sees
- Real pairs from the four sources
- MVTec AD
- Reference
- Defect
- Bottle: large break on the rim
- VisA
- Reference
- Defect
- Candles: small piece missing, top left
- MVTec LOCO
- Reference
- Defect
- Logical: the label is missing
- GoodsAD
- Reference
- Defect
- Can: scratch at the bottom
- The one reference (random_templates[0]) can be another variant: a different juice or can brand. The system must tell variation from defect.
- Images from MMAD: MVTec AD and MVTec LOCO (CC BY-NC-SA 4.0), VisA (CC BY 4.0), GoodsAD (research use). Defect descriptions from mmad.json.
- [image: Good bottle]
- [image: Bottle with broken rim]
- [image: Four good candles]
- [image: Four candles, top-left one has a missing piece]
- [image: Juice bottle with both labels]
- [image: Juice bottle with no labels]
- [image: Good drink can of another brand]
- [image: Drink can with a scratch at the bottom]

> Speaker notes: These are real MMAD pairs, exactly as the strict protocol gives them: the reference is always the first random normal template of that category. Defects range from obvious (a broken bottle) to tiny (a chip on one of four candles) to logical (a label that should be there and is not, which does not look like a stain at all). The bottom box is an important catch we found while preparing these slides: in MVTec LOCO and GoodsAD the single reference can be another product variant, so a naive pixel or patch comparison flags the whole object. That is why the readers and the verifier matter, and why we break results down by source.

## 9. `subtarefas`

- Datasets · the questions
- 7 subtasks, all multiple choice
- Subtask
- Questions
- Options
- Anomaly discrimination
- 8,297
- yes / no
- Defect classification
- 4,688
- 4
- Defect localization
- 4,878
- 4
- Defect description
- 4,710
- 4
- Defect analysis
- 4,782
- 4
- Object classification
- 3,155
- 4
- Object analysis
- 9,160
- 4
- Score = accuracy per subtask, then the plain mean of the 7 (balanced accuracy for discrimination).
- Left: Jiang et al., MMAD (ICLR 2025), Fig. 2. Counts: corrected mmad.json (Oct 22, 2026), verified by us. Answers are used only to score, never as input.
- [image: One example question per MMAD subtask, with the correct option highlighted]

> Speaker notes: Every image comes with several of these questions, already answered in mmad.json. The figure shows one example per subtask with the correct option highlighted. Discrimination is yes or no; everything else has four options, so random guessing gives 50% and 25%. Models do well on object questions and poorly on defect questions, which is where a visual expert should help most. The official metric is accuracy per subtask and the simple mean of the seven.

## 10. `metricas`

- Metrics
- What the full system must beat
- B0
- Random guess
- floor
- B1
- Qwen3-VL-8B, 1 call
- best single
- B2
- InternVL3-8B · Qwen3-VL-4B
- diversity, size
- B3
- B1 × k samples, vote
- same compute
- B4
- B1 + B2 vote
- no expert
- S
- Full system
- + ablations
- Success: S beats B1 and B3 with a McNemar test (paired, same questions). A well-measured negative result is also valid.
- Accuracy
- Per subtask, mean of 7, by source
- Inspection errors
- Good parts rejected, defects missed
- Significance
- McNemar + 95% bootstrap intervals
- Cost
- Calls, tokens, seconds per image
- Error analysis: 100 failures of S labeled as perception, reasoning, orchestration (MAST taxonomy) or annotation error.

> Speaker notes: The baselines are designed to rule out easy explanations. B1 is the best single open model. B3 gives that same model the same number of calls as our system and takes a majority vote, so if S wins it is not just more compute. B4 votes two different models without the expert or verifier, to isolate the value of diversity. Then the ablations remove one agent at a time. McNemar is a paired test: it compares, question by question, where one system is right and the other wrong. Besides accuracy we report the inspection errors that matter in a factory and the cost per image.

## 11. `resultados`

- Results
- The goal is to beat 73.9% with significance
- MMAD mean accuracy, 1-shot (axis from 50%)
- Human expert inspector
- 86.7
- SiGMA (other protocol)
- 85.2
- AD-Copilot 7B (trained)
- 82.3
- GPT-4o
- 74.9
- Qwen3-VL-8B: our B1
- 73.9
- Qwen2.5-VL-7B
- 72.2
- InternVL3.5-8B
- 65.3
- Our system S
- Nov 26
- Preliminary · Nov 12
- B1, B2 and the 4B model on dev; visual expert with anomaly maps; Colab throughput.
- Final · Nov 26
- S vs every baseline on test; largest gain expected in discrimination and localization.
- No results of our own yet. Sources: official MMAD leaderboard (Oct 2026); SiGMA, arXiv 2610.12310 (not comparable: 256 references, tuned on MMAD).

> Speaker notes: This is the leaderboard we position ourselves in. Our baseline, Qwen3-VL-8B, sits at 73.9%, about a point below GPT-4o. The vertical line marks the bar we must clear. SiGMA is drawn dashed because it uses a different, much easier protocol. AD-Copilot is trained on industrial data, which we do not do. The empty dashed row is our system: its value will come from the test split, run once on Nov 26 after the configuration is frozen. For Delivery II we will show the baselines on the dev split and the working visual expert.

## 12. `referencias`

- References
- Theoretical basis and related work
- Adimulam, Gupta and Kumar (2026). The Orchestration of Multi-Agent Systems. arXiv 2601.13671.
- Guo et al. (2024). Large Language Model based Multi-Agents: A Survey. arXiv 2402.01680.
- Anthropic (2024). Building effective agents.
- Du et al. (ICML 2024). Multiagent Debate. arXiv 2305.14325.
- Huang et al. AgentCoder. arXiv 2312.13010.
- Cemri et al. (NeurIPS 2025). Why Do Multi-Agent LLM Systems Fail? arXiv 2503.13657.
- Jiang et al. (ICLR 2025). MMAD: benchmark of MLLMs for industrial anomaly detection. arXiv 2410.09453.
- Zhang et al. (2026). SiGMA: Is In-Domain Training Enough? arXiv 2610.12310.
- Chen et al. (2025). Echo: Can MLLMs be Guided to Improve IAD? arXiv 2501.15795.
- Jiang et al. (2026). AD-Copilot. arXiv 2603.13779.
- Bergmann et al. MVTec AD (CVPR 2019) and MVTec LOCO (IJCV 2022); Zou et al. VisA (ECCV 2022); Zhang et al. GoodsAD (2023).
- Siméoni et al. (2025). DINOv3. arXiv 2508.10104 · Wang et al. (CVPR 2024). Real-IAD, multi-view. arXiv 2403.12580.

## 13. `cronograma`

- Schedule
- Four weeks, one frozen config
- Oct 29 · now
- I · Plan
- State of the art, data and models chosen, workflow and metrics
- Nov 12
- II · Preliminary
- Harness, splits, B1 and B2 on dev, DINOv3 expert with maps, Colab throughput
- Nov 19 · freeze
- Config frozen
- Readers, verifier, aggregator and weights fixed on dev; git tag v-frozen
- Nov 26
- III · Final
- Test run once, B3 and ablations, error analysis, report
- Peter (local GPU): visual expert, evaluation harness, error analysis
- Charles (Colab): readers, verifier, aggregator, baseline runs
- Both: prompts on dev, report and slides. The test only runs after the freeze, so nothing is tuned on the final result.

> Speaker notes: Four weeks. Delivery II on Nov 12 is about infrastructure and baselines: the harness that reproduces the official MMAD score, the fixed splits, the single-model baselines on dev and the visual expert producing maps. On Nov 19 we freeze every prompt, weight and threshold and tag the commit. Only then do we run the test split once, plus the same-compute vote and the ablations, and finish with the error analysis. The split of work follows the hardware: Peter has the local GPU, Charles works on Colab.

## 14. `obrigado`

- OneRef · IA025 final project
- Thank you! Questions?
- Code, plan and slides: github.com/RVilchezzz/OneRef
- Peter Roque Vilchez · Charles Maxime Doudy
- [image: Anomaly map over the broken bottle]

> Speaker notes: Thank you. The repository is public and has the plan, the slides and soon the code.
