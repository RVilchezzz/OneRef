# Speaker script: Delivery I (15 min)

Multi-agent visual inspector · IA025 · Unicamp 2026 · Peter Roque Vilchez & Charles Maxime Doudy

Deck: `IA025 – Final Project Plan (Delivery I).pptx` (14 slides). Each `Script` block below is also in the speaker notes of `IA025 – Final Project Plan (Delivery I) – notes EN.pptx`. French version: `speaker_script_delivery_I.fr.md`.

## Timing and speakers

About 1,900 spoken words, roughly 14.5 min at 130 words/min.

| # | Slide | Speaker | Time |
|---|---|---|---|
| 1 | Cover | Charles | 30 s |
| 2 | Project description | Charles | 90 s |
| 3 | Motivation | Charles | 90 s |
| 4 | Methodology | Peter | 120 s |
| 5 | Orchestration | Charles | 75 s |
| 6 | Compute resources | Peter | 60 s |
| 7 | Datasets | Peter | 60 s |
| 8 | What the system sees | Peter | 55 s |
| 9 | The questions | Peter | 45 s |
| 10 | Metrics | Charles | 90 s |
| 11 | Results | Charles | 60 s |
| 12 | References | Peter | 15 s |
| 13 | Schedule | Peter | 45 s |
| 14 | Thank you | Charles | 15 s |

Charles (about 7.5 min) covers the problem, the readers, the verifier, the aggregator and the evaluation. Peter (about 6.5 min) covers the visual expert, the hardware, the data and the schedule. Each of you answers the jury's questions on your own part.

## Know by heart

1. **The gaps.** Expert humans score **86.7 %** (86.65). GPT-4o scores **74.9 %**, which is **11.8 points** lower. Qwen3-VL-8B, our B1, scores **73.9 %**, which is **12.8 points** lower. Say "about 12 points behind GPT-4o, about 13 behind our open baseline". Do not say "13 points between GPT-4o and humans".
2. **SiGMA (85.2 %)** uses "a different, more favorable protocol". Never say "cheating". The facts: up to 256 normal images per category, MMAD text retrieval, a team and hyperparameters chosen using MMAD answers, and 4 × L20 GPUs. The official leaderboard itself says that exclusion "means a protocol difference, not an allegation of misconduct".
3. **Our own limit.** Our dev split (10 %) comes from MMAD, so we are not eligible for the strict leaderboard track either. Dev and test are disjoint, the configuration is frozen on Nov 19, and the test runs once. Say this before the jury does.

## Handovers

- 3 → 4, Charles: "Peter will now walk one image through the workflow."
- 4 → 5, Peter: "Charles will now show how these pieces map onto the orchestration framework."
- 5 → 6, Charles: "Peter, how do we run this on our hardware?"
- 9 → 10, Peter: "Charles will now explain how we will judge all this."
- 11 → 12, Charles: "Peter will close with the references and the schedule."
- 13 → 14, Peter: "Charles?"

---

## 1 · Cover: Multi-agent visual inspector

### Visual content
- Kicker: IA025 · Deep Learning · Unicamp 2026
- Title: **Multi-agent visual inspector**
- Subtitle: a team of pre-trained models finds defects in industrial parts from **one photo of a good part**, validated on the **MMAD** benchmark
- Team: Peter Roque Vilchez · Charles Maxime Doudy · Delivery I · October 29, 2026
- Images: Reference (good bottle, MVTec AD) | Part to inspect (bottle with a broken rim)

### Script — Charles (~30 s, ~70 words)
Good morning. I am Charles Maxime Doudy, and with Peter Roque Vilchez we present our IA025 final project: a multi-agent visual inspector. The idea fits in one sentence: a team of pre-trained models finds defects in an industrial part using a single photo of a good part, with no training, validated on the public MMAD benchmark. On the right, a real example: a good bottle, and a bottle with a broken rim.

### To add / improve
- Add the research question in one line under the title: "How much of the multi-agent gain survives with one reference and no tuning?"
- Label the left photo "Reference: the only normal image the system gets".
- The repo link on slide 14 (`MULTIAGENT_INSPECTOR`) is different from the one in `slides_content.md` (`OneRef`). Check which one works.

### Jury question
**"Is this really a multi-agent system, or just a pipeline?"**
In Anthropic's terms, it is an *orchestrated workflow*: the control flow is fixed in code. Each agent is still a separate model with its own role, its own input/output schema and logged messages, which is exactly the worker / service / support split in Adimulam et al. We chose a fixed flow on purpose: it is reproducible, and MAST shows that free-form coordination is where multi-agent systems fail most. The one decision an agent takes on its own is the verifier's reopen.

---

## 2 · Project description: one good photo, one part, a question

### Visual content
- Reference (good) | Part to inspect | Question "Where is the defect?" A. Top right / B. Top left / C. Center / **D. Bottom right ✓**
- Hypothesis: a **training-free** team (visual expert, two MLLM readers, verifier) beats the best single model **and** a same-cost majority vote
- Bars (axis from 50 %): Expert human inspectors **86.7 %** · GPT-4o **74.9 %** · Qwen3-VL-8B (open, our baseline) **73.9 %**
- Caption: "A 13-point gap to humans"

### Script — Charles (~90 s, ~195 words)
Here is exactly what the system receives. One photo of a good part, the reference. The part to inspect. And a multiple-choice question, here: where is the defect? The answer is D, bottom right. In the MMAD benchmark, this single bottle comes with five such questions: is there a defect, which type, where, what it looks like, and what its consequence is.

Why does this setting matter? When a production line changes product, there is no time and no data to train a dedicated detector. Often you have one good photo. Multimodal language models, MLLMs, can follow such instructions without training. But they are not good enough yet. Expert human inspectors reach 86.7 percent on MMAD. GPT-4o reaches 74.9, about twelve points below. Qwen3-VL-8B, the open model we use as our baseline, reaches 73.9, about thirteen points below. That gap is the room we try to close.

Our hypothesis is measurable: a training-free team of open models of at most eight billion parameters, a visual expert, two MLLM readers and a verifier, beats both the best single model and a majority vote that spends the same compute. If it does not, we want to know why.

### To add / improve
- Change the caption "A 13-point gap to humans" to **"−12.8 pts vs our baseline · −11.8 vs GPT-4o"**.
- Add a bar for **ordinary (untrained) humans, 78.7 %**: MLLMs are even below non-experts, which is a strong argument.
- Show the bottle's 5 question types as small chips (defect? / type / where / looks like / effect).

### Jury question
**"Real inspection is open-ended. Why evaluate with multiple-choice questions?"**
Multiple choice gives an objective, reproducible score, and it makes 39,670 questions comparable across every published model. The chance level (50 % / 25 %) is handled explicitly by baseline B0. The system still writes a free-text report for the operator, but only the multiple-choice answers are scored, and we say so as a limitation.

---

## 3 · Motivation: MLLMs reason well once told where to look

### Visual content
- Left, MMAD (InternVL2-40B), Jiang et al. ICLR 2025, Table 6:
  - Anomaly discrimination: MLLM alone **63.8** · + PatchCore map 60.0 · + AnomalyCLIP map 59.6 · + true location **88.7**
  - Defect localization: MLLM alone **52.9** · + best detector 60.4 · + true location **81.0**
- Right: SiGMA (arXiv 2610.12310) figure + table: Normal images **up to 256** vs **1** · Tuned on MMAD: yes (team + params) vs dev split only · GPUs **4 × L20** vs Colab L4 / T4

### Script — Charles (~90 s, ~195 words)
Why a multi-agent design, and why a visual expert at its core? The answer comes from an experiment in the MMAD paper itself. On the left, InternVL2-40B answers alone: 63.8 percent on defect discrimination. If we add the anomaly map of a classic detector, PatchCore or AnomalyCLIP, it gets worse, around 60: these detectors also fire on normal parts. But if we tell the model where the defect really is, using the ground-truth mask turned into text, it jumps to 88.7. Localization goes from 52.9 to 81. This is an oracle, an upper bound, but the message is clear: MLLMs reason well once they know where to look. Their bottleneck is fine perception.

The closest work is SiGMA, published this month. A team of MLLMs with a DINOv3 visual expert reaches 85.2 percent, close to human experts. But it works under a different, more favorable protocol: up to 256 normal images per category instead of one, retrieved domain text, a team and hyperparameters chosen on MMAD answers, and four L20 GPUs. So our research question is: how much of that gain survives with one reference photo, open models, and no tuning on the test? Peter will now walk one image through the workflow.

### To add / improve
- Hatch the "+ true location" bars and label them **"oracle (upper bound)"**, with the deltas in bold: **+24.9** and **+28.1**.
- Title the SiGMA table **"Different protocol"** and add a row "Domain text: yes / no". Avoid any wording that suggests misconduct.
- Footnote: the official leaderboard places SiGMA outside the main track, as a protocol difference.

### Jury question
**"If classic detector maps hurt the MLLM, why would your DINOv3 map help?"**
In MMAD the PatchCore and AnomalyCLIP maps were handed over raw and also fired on normal images, which hurt discrimination; localization still gained +7.5. We do three things differently. The expert's output is turned into a region on a 3×3 grid in the vocabulary of the answer options, plus a crop. It votes only on discrimination, with a weight set on dev, so the readers keep the final say. And SiGMA shows that a DINOv3 expert does help inside a team. If it does not help here, the ablations will show it, including "raw map vs calibrated map vs text only".

---

## 4 · Methodology: one image through the workflow

### Visual content
- **Input**: 1 reference + the part · "Where is the defect?" A / B / C / D
- **1 · Visual expert**: compares patches with the reference → map, 3×3 region, crop, score
- **2 · Two readers**: Qwen3-VL-8B and InternVL3-8B → answer · confidence (same photos + the expert's hint; JSON reply)
- **3 · Verifier**: ✓ valid schema / ✓ answers consistent / ✓ agrees with expert · conflict → **reopen once**, with the evidence
- **4 · Aggregator**: weighted vote per reader and subtask; the expert votes on "defect?"
- **5 · Log**: every message, time and token count as JSONL per image
- Images: reference, part, anomaly map (illustrative DINOv2 ViT-S/14 run)

### Script — Peter (~120 s, ~260 words)
Thank you, Charles. Let me walk one real image through the workflow. The input is the good bottle, the broken one, and the question: where is the defect?

Step one, the visual expert. It is a frozen DINOv3 encoder; nothing is trained. It cuts both photos into small patches. For each patch of the part, it finds the most similar patch in the reference and measures the distance. Far from everything normal means suspicious. That gives this heat map, which we computed ourselves on this exact pair with DINOv2, the previous version. It lights up the broken rim. The expert then turns the peak into words, for example "bottom right, high score", a cell of a three-by-three grid in the vocabulary of MMAD's answer options, plus a zoomed crop. To be transparent: on this pair our quick DINOv2 version put the peak in the bottom-center cell, next to the true answer. Calibrating that is a Delivery II task.

Step two, two readers from different model families, Qwen3-VL-8B and InternVL3-8B. They see the same photos plus the expert's hint, and reply in JSON: an answer and a confidence.

Step three, the verifier. It checks the format, the consistency between answers, for example "no defect" together with "the defect is in the center", and agreement with the expert. On a conflict it reopens the question once, with the evidence, never more, to avoid loops.

Step four, the aggregator votes with weights fixed on the dev split. Step five, every message, time and token count is logged. Charles will now show how these pieces map onto the orchestration framework.

### To add / improve
- Draw the reopen loop as a dashed arrow from the verifier back to the readers, labeled **"max 1"**.
- Label the expert → aggregator arrow "votes on discrimination only".
- Caption the map honestly: "illustrative DINOv2 run; peak in the bottom-center cell, true answer bottom right".
- Show an illustrative JSON reply: `{"answer": "D", "confidence": 0.82}`.
- Say on the slide that the verifier is Qwen3-VL-8B with a critic prompt.

### Jury question
**"Your own map puts the peak in the wrong cell. Won't the expert mislead the readers?"**
That is exactly why the expert never decides alone. Bottom-center vs bottom-right is a boundary case of the 3×3 grid. The readers still see the full images and the crop, and the verifier reopens only on a clear conflict. On dev we will calibrate the threshold and filter connected regions. One option is to report the two top cells when the peak sits on a border. The "raw map / calibrated map / text only" ablation measures which format really helps.

---

## 5 · Orchestration: each agent has one role in the framework

### Visual content
- Figure: Adimulam et al., arXiv 2601.13671, Fig. 2
- **Orchestration layer** → Python orchestrator: fixed plan per subtask
- **Worker agents** → Visual expert · Reader A · Reader B
- **Service agent (quality control)** → Verifier: schema, consistency, reopen once
- **Support · state** → Aggregator + JSONL log of every message
- Design lessons: start with a simple fixed workflow (Anthropic, 2024); multi-agent gains are often small, so every agent must earn its place in an ablation (MAST, NeurIPS 2025)

### Script — Charles (~75 s, ~170 words)
Thank you, Peter. The paper attached to this course, Adimulam and colleagues, 2026, is a position paper. It has no experiments, so we use it as a vocabulary and a design template, not as evidence. We take three ideas from it. First, roles. Worker agents do the task: our visual expert and the two readers. A service agent does quality control: our verifier. Support agents keep the state: our aggregator and the JSONL log. Second, an orchestration layer that decides who runs, and in which order. For us it is plain Python with a fixed plan per subtask. Third, validating every output against a schema, which we do with Pydantic.

Two lessons keep the design honest. Anthropic recommends starting with the simplest fixed workflow and adding agents only when a metric improves. And MAST, a NeurIPS 2025 study of why multi-agent systems fail, found that gains over a single agent or best-of-N sampling are often minimal. So every agent must earn its place in an ablation. Peter, how do we run this on our hardware?

### To add / improve
- Add a small "What we leave out" column: the MCP / A2A protocols (overkill for 4–5 agents in one process; exposing the expert over MCP is optional) and the paper's industry figures (taken from blogs and consulting reports, not from experiments).
- Footnote: "Several references in the paper have wrong authors; we cite the original papers."

### Jury question
**"The paper recommends MCP and A2A. Why don't you use them?"**
They standardize communication between many teams and services in production. With four or five agents in a single Python process, typed JSON messages validated with Pydantic give the guarantee the paper actually cares about, schema validation, without network overhead. Exposing the visual expert as an MCP tool is an optional extension. It would not change any result.

---

## 6 · Compute resources: a laptop GPU, Colab and a shared cache

### Visual content
- **Local · Peter**: RTX 3060 Laptop, **6 GB** → DINOv3 visual expert; debug the whole pipeline with Qwen3-VL-4B in 4-bit
- **Shared · Google Drive**: MMAD images + one JSONL cache per agent, resumable
- **Cloud · Charles**: Colab Pro, **L4 / T4** → 8B readers served with **vLLM**, baselines and final test (**100 units, no A100**)
- Pass-based execution, one model on the GPU at a time: Expert pass → Reader A pass → Reader B pass → Verifier pass → Aggregate
- Budget: cost measured on 50 questions before the test; B1, B2 and B4 reuse the cached reader answers

### Script — Peter (~60 s, ~135 words)
Yes, we have the hardware for this, if we are careful. Locally, I have an RTX 3060 laptop GPU with 6 gigabytes. That is enough for the DINOv3 expert, and for debugging the whole pipeline with a small 4-billion-parameter Qwen model in 4-bit. The two 8B readers run on Colab Pro, on L4 or T4 GPUs, served with vLLM, which batches many questions at once. We never use the A100: it burns compute units several times faster. The key idea is pass-based execution: each agent runs over the whole split, writes its answers to Drive, and frees the GPU. So two 8B models are never loaded together, and a disconnect loses nothing. Baselines B1, B2 and B4 reuse those cached answers for free. And before the test, we measure the cost on 50 questions.

### To add / improve
- Add the third-party cost estimates: **T4 ≈ 1.76 units/h, L4 ≈ 2.5–5, A100 ≈ 13–15**.
- Add plan B: **+100 units for US$ 9.99** before shrinking the test.
- After Delivery II, replace the estimates with the measured units per question.

### Jury question
**"Are 100 Colab units really enough?"**
The cost is in MLLM calls, and the cache removes most of them. B1, B2 and B4 reuse the full system's reader answers; only S's own calls (readers, verifier, reopens) and B3's extra samples are new. 100 units is roughly 57 hours on a T4 or 20–40 hours on an L4. We measure units per question on 50 questions before fixing the test size (1,000–2,000 images). If we run short, we buy 100 more units for US$ 9.99 rather than shrink the test.

---

## 7 · Datasets: MMAD, 8,366 images, 39,670 questions

### Visual content
- Figure: Jiang et al., MMAD (ICLR 2025), Fig. 1 (normal / abnormal, 7 subtasks, 38 categories)
- Table (normal / defective / license): MVTec AD **465 / 1,226** CC BY-NC-SA 4.0 · VisA **943 / 1,198** CC BY 4.0 · MVTec LOCO **573 / 993** CC BY-NC-SA 4.0 · GoodsAD **1,309 / 1,659** research use
- Split (fixed seed, stratified): **Dev 10 % (~840 images)** for prompts and weights · **Test ~2,000 images** from the other 90 %, run once with the frozen config
- Evaluation only, no training. Corrected revision, Oct 22, 2026

### Script — Peter (~60 s, ~135 words)
Our data is MMAD, a public benchmark from ICLR 2025: 8,366 images and 39,670 multiple-choice questions, built on four industrial datasets, MVTec AD, VisA, MVTec LOCO and GoodsAD, covering 38 product categories. It is fully labeled, so we annotate nothing, and we use the labels only to score. MMAD is test-only, so we make our own split, with a fixed seed, stratified by source and category. Ten percent, about 840 images, is our dev set, where we may look at answers while choosing prompts and weights. From the other ninety percent we draw about 2,000 test images, run once, after the configuration is frozen on November 19. Because we tune on part of MMAD, we do not claim a place in the strict leaderboard track; we compare all baselines on the same split.

### To add / improve
- Add the totals: **3,290 normal / 5,076 defective**, 38 categories.
- Add one honest line: "Dev comes from MMAD → not eligible for the strict main track (declared)".
- Mention that the `mmad.json` SHA-256 was checked against the authors' file.
- Licenses: non-commercial is fine for a course project, but blocks direct commercial use.

### Jury question
**"Your test set is a 2,000-image sample. How can you compare with the leaderboard?"**
Not exactly, and we say so. The comparison that counts is internal: every baseline is rerun on the same images and questions. Leaderboard numbers are context only. The sample is stratified by source, category and normal/defective, and comes with 95 % bootstrap intervals. If throughput allows, we extend it to the full 90 %, about 7,530 images and 35,700 questions.

---

## 8 · Datasets: what the system sees

### Visual content
- 4 real reference / defect pairs:
  - MVTec AD: bottle, **large break on the rim**
  - VisA: candles, **small piece missing**, top left
  - MVTec LOCO: **logical** defect, the label is missing
  - GoodsAD: can, **scratch at the bottom**
- Box: the single reference (`random_templates[0]`) can be **another product variant** (a different juice, a different can brand). The system must tell variation from defect.

### Script — Peter (~55 s, ~120 words)
These are real MMAD pairs, exactly as the strict protocol gives them: the reference is always the first normal template of the category. Defects range from obvious, a broken bottle, to tiny, a chip on one of four candles, to logical: in MVTec LOCO, the label that should be there is simply missing, which does not look like a stain at all. And here is a catch we found while preparing these slides: in LOCO and GoodsAD, the single reference can be another product variant, a different juice, a different can brand. A naive patch comparison then flags the whole object. That is exactly why we need readers and a verifier, and why we report results per source.

### To add / improve
- Circle the defect in red on each "Defect" image.
- Tag each pair with its defect type: structural / tiny / logical / variant.
- Mark the LOCO and GoodsAD references with **"different variant!"**.

### Jury question
**"If the reference is a different variant, isn't the 1-shot setting unfair on GoodsAD?"**
It is the benchmark's standard policy, and it is realistic: on a line, the reference photo is not always the exact same product version. Choosing a closer reference would be the "similar template" setting, a different protocol. We expect smaller gains there and report results per source. This is where the readers' semantic understanding should override the patch map, and the error analysis will check whether it does.

---

## 9 · Datasets: the questions

### Visual content
- Figure: one example question per subtask (MMAD, Fig. 2)
- Table (questions / options): Anomaly discrimination **8,297** yes/no · Defect classification **4,688** · Defect localization **4,878** · Defect description **4,710** · Defect analysis **4,782** · Object classification **3,155** · Object analysis **9,160** (all 4 options except discrimination)
- Score = accuracy per subtask, then the **plain mean of the 7** (balanced accuracy for discrimination)

### Script — Peter (~45 s, ~95 words)
Each image comes with several of these questions, already answered. There are seven subtasks. Discrimination, is there a defect, is yes or no. All the others have four options, so chance is 50 and 25 percent. Models are strong on object questions, above 90 percent for object classification, and weak on defect questions: our baseline gets only 61 percent on localization. That is where a visual expert should help most. The score is accuracy per subtask, balanced for discrimination, then the plain mean of the seven. Charles will now explain how we will judge all this.

### To add / improve
- Add a **B1 (Qwen3-VL-8B)** column: 68.7 · 63.3 · **61.0** · 68.2 · 80.0 · 92.3 · 84.0, with the weakest highlighted.
- Add a chance column: 50 / 25.
- Footnote: "Object analysis = Object Structure + Object Details (official script)".

### Jury question
**"Discrimination has 8,297 questions but counts for only 1/7 of the score. Isn't the mean misleading?"**
It is the official MMAD metric, so we keep it for comparability. We also report every subtask separately, plus the two discrimination error types that matter in a factory: good parts rejected and defects missed. Those have very different costs, and no single average captures them.

---

## 10 · Metrics: what the full system must beat

### Visual content
- Baseline ladder:
  - **B0** random guess (floor)
  - **B1** Qwen3-VL-8B, 1 call (best single)
  - **B2** InternVL3-8B · Qwen3-VL-4B (diversity, size)
  - **B3** B1 × k samples, vote (**same compute**)
  - **B4** B1 + B2 vote (no expert)
  - **S** full system + ablations
- Success: **S beats B1 and B3** with a **McNemar** test (paired, same questions). A well-measured negative result is also valid.
- Cards: Accuracy (per subtask, mean of 7, per source) · Inspection errors (good parts rejected, defects missed) · Significance (McNemar + **95 % bootstrap**) · Cost (calls, tokens, s/image)
- Error analysis: **100 failures** of S labeled perception / reasoning / orchestration (**MAST**) / annotation error

### Script — Charles (~90 s, ~195 words)
Thank you, Peter. Our baselines are designed to rule out the easy explanations of any gain. B0 is random guessing, the floor. B1 is the best single open model, Qwen3-VL-8B, one call per question. B2 is InternVL3-8B, our second reader, plus the 4B model to see the effect of size. B3 is the key one: we give B1 exactly the same number of calls as our full system, and take a majority vote. If our system only beats B1, it might just be more compute; it has to beat B3. B4 votes the two different readers, without expert or verifier, to isolate the value of model diversity. Then ablations remove one agent at a time.

Success means S beats B1 and B3 with a McNemar test, a paired test that compares, question by question, where one system is right and the other is wrong. We add 95 percent bootstrap intervals. Beyond accuracy, we report the errors that matter in a factory, good parts rejected and defects missed, plus the cost per image. Finally, we will label 100 failures as perception, reasoning, orchestration following MAST, or annotation errors. A well-measured negative result is also a result.

### To add / improve
- Give B3's **k** as a number: S's MLLM calls per question (2 readers + verifier + reopens, about 3–4), measured on dev.
- Add a **Holm correction** on the two primary comparisons (S vs B1, S vs B3). This is a new commitment; settle it with Peter before the talk.
- Show the minimum detectable effect, **about 1 point** (see Backup Q&A 3), and say that the bootstrap resamples **images**.

### Jury question
**"With five baselines and five ablations, you run many tests. What about multiple comparisons?"**
The primary hypothesis is fixed in advance and covers two comparisons, S vs B1 and S vs B3, with a Holm correction. B2, B4 and the ablations are explanatory: we report them with intervals, not as confirmatory claims. And because the configuration is frozen before the single test run, we cannot go fishing on the test set.

---

## 11 · Results: the goal is to beat 73.9 % with significance

### Visual content
- MMAD mean accuracy, 1-shot (axis from 50 %): Human expert **86.7** · SiGMA (other protocol, dashed) **85.2** · AD-Copilot 7B (trained) **82.3** · GPT-4o **74.9** · **Qwen3-VL-8B, our B1, 73.9** · Qwen2.5-VL-7B 72.2 · InternVL3.5-8B 65.3 · Our system S: *Nov 26*
- Vertical line at 73.9 = the bar to clear
- Preliminary, Nov 12: B1, B2 and the 4B model on dev; visual expert with maps; Colab throughput
- Final, Nov 26: S vs every baseline on test; largest gain expected in discrimination and localization

### Script — Charles (~60 s, ~130 words)
This is where we stand today: no results of our own yet, and the bar we must clear. Our baseline, Qwen3-VL-8B, is at 73.9 percent, about one point below GPT-4o. SiGMA is drawn dashed because of its different protocol. AD-Copilot, at 82.3, is trained on other industrial data, which we do not do. The empty row is our system: it will be filled once, on November 26, from the test split. For Delivery II, on November 12, we will show B1 and B2 measured on our dev split, and the visual expert producing maps. Note that 73.9 is a reported number; SiGMA measured the same model at 74.5. So we re-measure B1 ourselves, and that number, not the leaderboard, is our reference. Peter will close with the references and the schedule.

### To add / improve
- Footnote: "73.9 is reported by the AD-Copilot paper; SiGMA measures 74.5 → we re-measure B1 on our split."
- Add a dotted line for **ordinary humans, 78.7**.
- Color the bars by protocol: strict 1-shot vs other protocol vs trained.
- For Delivery II: replace the empty row with B1 measured on dev.

### Jury question
**"What gain would you call a success, realistically?"**
A statistically significant gain over B3; with about 9,500 paired questions, roughly one point is detectable. We expect the gain mostly on discrimination and localization, where B1 is weakest (68.7 and 61.0) and where the MMAD oracle showed +25 to +28 points of headroom. We expect little on object questions, which are already above 84 %. We do not promise SiGMA's 85.2: it uses up to 256 references.

---

## 12 · References

### Visual content
- Orchestration: Adimulam et al. 2026 · Guo et al. 2024 · Anthropic 2024 · Du et al. ICML 2024 · Huang et al. (AgentCoder) · Cemri et al. NeurIPS 2025 (MAST)
- Benchmark and methods: Jiang et al. ICLR 2025 (MMAD) · Zhang et al. 2026 (SiGMA) · Chen et al. 2025 (Echo) · Jiang et al. 2026 (AD-Copilot)
- Data and encoders: MVTec AD, MVTec LOCO, VisA, GoodsAD · DINOv3 (Siméoni et al. 2025) · Real-IAD

### Script — Peter (~15 s, ~30 words)
Our references are here. Two to remember: MMAD for the benchmark, and MAST for why multi-agent systems fail. We cite original papers directly, since a few attributions in the attached paper are wrong.

### To add / improve
- Group the references under the three headings above.
- Footnote listing the corrected attributions (see the jury question).

### Jury question
**"You say the attached paper has wrong references. Which ones?"**
Checked against arXiv: [5] Agent Laboratory is by Schmidgall et al., not Deng et al.; [6] AgentCoder is by Huang et al., not Luo et al.; [8] KoMA is by Jiang et al., not Zhao et al.; and [9], arXiv 2402.01680, is Guo et al.'s survey under a different title. That is why the report cites the originals.

---

## 13 · Schedule: four weeks, one frozen config

### Visual content
- **Oct 29 · now**: I · Plan (state of the art, data, models, workflow, metrics)
- **Nov 12**: II · Preliminary (harness, splits, B1 and B2 on dev, DINOv3 expert with maps, Colab throughput)
- **Nov 19 · freeze**: config frozen (readers, verifier, aggregator, weights fixed on dev; git tag `v-frozen`)
- **Nov 26**: III · Final (test run once, B3 and ablations, error analysis, report)
- Peter (local GPU): visual expert, evaluation harness, error analysis · Charles (Colab): readers, verifier, aggregator, baseline runs · Both: prompts on dev, report and slides

### Script — Peter (~45 s, ~105 words)
Four weeks, one frozen configuration. Delivery II, on November 12, is about infrastructure: the harness that reproduces the official MMAD score, the fixed splits, B1 and B2 on dev, and the visual expert producing maps. On November 19 we freeze every prompt, weight and threshold, and tag the commit. Only then do we run the test, once, along with the same-compute vote and the ablations, and finish with the error analysis on November 26. The work follows the hardware: I take the visual expert, the harness and the error analysis on my local GPU; Charles takes the readers, verifier, aggregator and baselines on Colab. Charles?

### To add / improve
- Show the test week (**Nov 19–23**) as the critical window.
- Add one risk per milestone with its mitigation, for example Colab throughput → stratified sample.

### Jury question
**"The test window is only Nov 19–23. What if the full system isn't ready on Nov 19?"**
The freeze date does not move, because it protects the test. A component that is not ready is left out of S and reported as future work; thanks to the ablation design, S without it is still a valid, measurable system. Throughput measured in the first weeks sets the test size (1,000–2,000 images), so the run fits in the window.

---

## 14 · Thank you: questions?

### Visual content
- "Thank you! Questions?"
- Code, plan and slides: github.com/RVilchezzz/MULTIAGENT_INSPECTOR
- Peter Roque Vilchez · Charles Maxime Doudy · anomaly map of the broken bottle

### Script — Charles (~15 s, ~40 words)
To sum up: one reference photo, a team of open models, no training, and a fair test against the same compute. The plan is public on GitHub, and the code will follow. Thank you; we are happy to take your questions.

### To add / improve
- Check that the repo link works (the name is different in `slides_content.md`).
- Add a QR code to the repo.
- Keep 2 backup slides ready for questions: B1 per subtask, and the Colab budget.

### Jury question
**"What would you do with more compute?"**
Run the full 90 % test, then full MMAD for a direct leaderboard comparison. Use MMAD-BBox to measure the expert's localization with IoU. And try the multi-view extension on Real-IAD (5 views per part), which addresses our single-view limitation.

---

## Backup Q&A

**1. "Isn't tuning on a 10 % MMAD dev split exactly what you criticize SiGMA for?"**
Partly yes, and we say so. The differences: one reference instead of up to 256, no added domain text, disjoint dev and test, and a configuration frozen before a single test run. We are not eligible for the strict main track, and we present our results as an internally fair comparison: every baseline runs on the same split under the same rules. Strict eligibility would require developing on external data such as MVTec AD 2; we declare this as a limitation.

**2. "How exactly do you count 'same compute'?"**
B3 gets the same number of MLLM calls per question as S: 2 readers, the verifier and any reopens, measured on dev. We also report tokens, seconds per image and GPU memory. A DINOv3 forward pass is negligible next to an 8B model generating text. If S uses more tokens per call because of the crop, the cost table shows it.

**3. "Do you have the statistical power to detect a gain?"**
The test has about 9,500 paired questions. If about 15 % are discordant (an assumption we will measure on dev), that is about 1,425 pairs, and a one-point gain gives McNemar z ≈ 95 / √1,425 ≈ 2.5, so it is detectable at p < 0.05. Questions about the same image are correlated, so the bootstrap resamples images, 1,000 times.

**4. "The verifier is Qwen3-VL-8B, like Reader A. What about self-preference bias?"**
Most of its checks are deterministic: the Pydantic schema, consistency rules between questions, and agreement with the expert's region. The LLM critic only decides whether to reopen, at most once. The "no verifier" ablation and the MAST category "incorrect verification" measure the risk. If a bias shows up, InternVL3 can be the critic.

**5. "Why a fixed workflow and not an LLM planner?"**
Anthropic advises starting with the simplest workflow. MAST shows that loops and poorly specified coordination are frequent failure modes. And a fixed plan per subtask makes the ablations interpretable and the runs reproducible. The only dynamic step is a single reopen.

**6. "Why training-free? A small fine-tune would surely help."**
The target scenario is a product changeover, with no time and no data to train. The trained models at the top (AD-Copilot, 82.3) need large industrial datasets. Staying training-free isolates the effect of orchestration. An optional small comparator head would be trained only on external data and reported separately.

**7. "The MLLMs may have seen MVTec images during pretraining."**
Web-scale pretraining cannot be audited; the leaderboard says the same. It affects all baselines equally, because S uses the same models. Our claims are relative: S vs B1 and B3.

**8. "A real part is 3D. One photo is not enough."**
Agreed. MMAD evaluates one image per question, so we inspect what one camera station sees. On a real line, several cameras or a 3D scan compared with the CAD model solve this. Multi-view with Real-IAD (5 views) is our optional extension.
