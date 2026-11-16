# Speaker script: Delivery II (10 min)

OneRef · IA025 · Unicamp 2026 · Peter Roque Vilchez & Charles Maxime Doudy

Deck: `OneRef – Preliminary Results (Delivery II).pdf` (11 slides). French version: `speaker_script_delivery_II.fr.md`. Spanish version: `speaker_script_delivery_II.es.md`.

## Timing and speakers

About 1,300 spoken words, roughly 10 min at 130 words/min.

| # | Slide | Speaker | Time |
|---|---|---|---|
| 1 | Cover | Charles | 25 s |
| 2 | Context | Charles | 60 s |
| 3 | What we built | Charles | 60 s |
| 4 | First real numbers | Peter | 75 s |
| 5 | What worked, what did not | Peter | 50 s |
| 6 | What went wrong | Peter | 60 s |
| 7 | First end-to-end run | Charles | 70 s |
| 8 | Examples from the run | Charles | 55 s |
| 9 | Planned vs. done | Peter | 40 s |
| 10 | Next steps | Peter | 45 s |
| 11 | Thank you | Charles | 10 s |

Charles (about 4.7 min) covers the context, the code of the readers, verifier, aggregator and orchestrator, and the end-to-end run. Peter (about 4.5 min) covers the DINOv3 visual expert, the data and splits, the problems found, the balance against the plan and the next steps. Each of you answers the jury's questions on your own part.

## Know by heart

1. **DINOv3 is not better on AUROC.** Over all dev images, v3 scores **0.59** and v2 **0.61**. We keep v3 because the expert's job is to *localize*: its peak falls on the defect **77 %** of the time on MVTec AD, against **51 %** for v2. GoodsAD (296 of the 835 images) pulls the overall average down.
2. **The 0.49 → 0.61 recalibration is in-sample.** It was fitted and measured on dev, so it is optimistic. Say it before the jury does.
3. **71.2 % on 18 images is not a result.** The 95 % CI is **[54 %, 86 %]**. The run used Qwen3-VL-4B quantized as both readers and a rules-only verifier. Never compare it with 73.9 %.
4. **Discrimination is 52.5 % balanced, close to chance (50 %).** This is the main weakness so far, and it matches the expert's weak AUROC outside MVTec AD.
5. **20/20 vs 21/21.** The lesson on slide 6 says 20 tests passed on fake data; slide 3 says 21. The 21st is the loader test with the 9 real names, added after the crash.
6. **The numbers add up.** 835 + 7,531 = 8,366 images. 184 calls = 2 readers × 88 questions + 4 reopened × 2 readers. "~3.7 s per question" is per reader call: (323 + 322) / 176 ≈ 3.7 s. Dev has 3,937 questions.
7. **73.9 % for Qwen3-VL-8B is a reported number** (AD-Copilot paper; SiGMA measures 74.5 %). B1 has not been re-measured on our split yet: that is the first task of this week.

## Handovers

- 3 → 4, Charles: "Peter will now show what the expert does on real data."
- 6 → 7, Peter: "Charles will show that first end-to-end run."
- 8 → 9, Charles: "Peter, where are we against the plan?"
- 10 → 11, Peter: "Charles?"

---

## 1 · Cover: OneRef

### Visual content
- Kicker: IA025 · Deep Learning · Unicamp 2026
- Title: **OneRef**
- Subtitle: preliminary results: the pipeline runs end to end, and the visual expert is measured on real MMAD data
- Team: Peter Roque Vilchez · Charles Maxime Doudy · Delivery II · November 12, 2026
- Images: a metal grid from MVTec AD ("Part") and our expert's anomaly map, with the red peak on the defect

### Script — Charles (~25 s, ~60 words)
Good morning. I am Charles Maxime Doudy, and with Peter Roque Vilchez we present Delivery II of OneRef, our IA025 project. Two weeks ago we showed the plan. Today we show where we stand: the pipeline runs end to end, and the visual expert is measured on real MMAD data. On the right, a real dev image and our expert's map: the red spot is the defect.

### To add / improve
- Add the research question in one line: "How much of the multi-agent gain survives with one reference and no tuning?"
- Label the top image "Part (MVTec AD grid, dev)" so the jury knows it is real data, not an illustration.

### Jury question
**"What changed since Delivery I?"**
Three things: the whole workflow is now code, the visual expert has run on all 835 dev images with measured scores per source, and all five agents have run together once on real images. What has not changed is the protocol: one reference, no training, test run once.

---

## 2 · Context: one good photo, one part, a question

### Visual content
- Reference (good bottle) and part (bottle with a broken rim) → **1 · Visual expert** → **2 · Two readers** → **3 · Verifier** → **4 · Vote**
- Benchmark: MMAD, **8,366 images**, **39,670** multiple-choice questions, **7 subtasks**, strict 1-shot
- Baseline to beat: Qwen3-VL-8B alone, **73.9 %**, and a same-cost majority vote
- Rules: no training, models **≤ 8B**, tune on dev only, test run once
- Footer: hypothesis = beats both baselines with significance (McNemar); a well-measured negative result is also valid

### Script — Charles (~60 s, ~130 words)
A quick reminder of the problem. The system gets one photo of a good part, the photo of the part to inspect, and a multiple-choice question. It answers in four stages: a visual expert compares the two photos, two readers, which are multimodal language models, answer the question, a verifier checks them, and a vote decides. We evaluate on MMAD: 8,366 images, 39,670 questions, seven subtasks, and we stay strictly one-shot, with one reference per question. The baseline to beat is Qwen3-VL-8B alone, reported at 73.9 %, and a majority vote with the same compute budget. The rules have not moved: no training, models of at most 8 billion parameters, every choice made on dev, and the test run only once. A well-measured negative result is also valid.

### To add / improve
- Write "73.9 % (reported)" and add "re-measured on our split" so nobody reads it as our own measurement.
- Replace "strict 1-shot" with "1 reference image per question": "1-shot" can be read as one training example.

### Jury question
**"Why should a negative result count?"**
Because the question is how much of SiGMA's 85.2 % survives under a strict protocol. SiGMA uses up to 256 normal images, MMAD domain text and a team chosen on MMAD answers. If the gain disappears with one reference and no tuning, that is a useful measurement, provided it comes with a confidence interval and an error analysis.

---

## 3 · What we built: the full workflow exists as code

### Visual content
- Five cards: **Visual expert** (DINOv3, patch distance, *ran on all 835 dev*) · **Readers A · B** (Qwen3-VL · InternVL3 via vLLM) · **Verifier** (rules + LLM critic, reopen ≤ 1) · **Aggregator** (weighted vote per subtask) · **Orchestrator** (one model per pass, cache, resume); the last four: *ran on real data*
- **21 / 21** GPU-free tests pass (schemas, critic, vote, metrics, resume)
- **835 / 7,531** dev / test images, stratified, seed fixed and committed
- **2.1 GB** to fetch dev instead of 28 GB
- Footer: Pydantic-validated JSON between agents, one retry then error; code on github.com/RVilchezzz/OneRef

### Script — Charles (~60 s, ~135 words)
First message: everything in the plan now exists as code. Five components. The DINOv3 visual expert. Readers A and B, Qwen3-VL and InternVL3, served through vLLM. The verifier, which combines fixed rules with an LLM critic and can reopen a conflict once at most. The aggregator, a weighted vote per subtask. And the orchestrator, which loads one model per pass, saves every output to disk and can resume after a crash. Agents only exchange JSON validated with Pydantic: an invalid answer is retried once, then counted as an error. Three numbers: 21 tests pass without a GPU; dev and test are fixed at 835 and 7,531 images, stratified, with a committed seed; and we download 2.1 gigabytes for dev instead of 28, by reading only the needed files from the remote zips. Peter will now show what the expert does on real data.

### To add / improve
- "Ran on real data" on the readers and verifier is true but needs a qualifier: write "ran on real data (4B, 18 images)". Otherwise the jury will think the 8B readers have run.
- Make the test count match slide 6 (21 here, 20 there): see "Know by heart" point 5.

### Jury question
**"Why one model per pass instead of a real-time pipeline?"**
Because the GPU only holds one 8B model: 6 GB locally, and an L4 or T4 on Colab. Each agent runs over the whole split, writes its outputs to disk and frees the GPU before the next one. It also lets every baseline reuse the same cached reader outputs, so B1 to B4 cost almost nothing extra.

---

## 4 · First real numbers: the expert works on aligned photos only

### Visual content
| Source (dev) | Images | AUROC v2 | AUROC v3 | In mask v2 | In mask v3 |
|---|---|---|---|---|---|
| MVTec AD | 166 | 0.88 | 0.84 | 51 % | **77 %** |
| VisA | 215 | 0.77 | 0.78 | 42 % | 41 % |
| MVTec LOCO | 156 | 0.64 | 0.62 | 37 % | 43 % |
| GoodsAD | 296 | 0.55 | 0.51 | 12 % | 10 % |
| All | 835 | 0.61 | 0.59 | 33 % | 40 % |
- Side cards: "DINOv3 localizes better: 77 % vs 51 % on MVTec AD" and "GoodsAD ≈ chance: the one reference is often another product"
- Footer: frozen DINOv2-base and DINOv3 ViT-B/16, query vs `random_templates[0]`, dev only, one run, **0.35 s** per image on the RTX 3060 Laptop; masks used only to score

### Script — Peter (~75 s, ~165 words)
Thank you, Charles. The visual expert compares the part with its single reference using frozen DINO features. For each patch of the part, the distance to the closest patch of the reference gives an anomaly map. We tested two backbones, DINOv2 and DINOv3, on all 835 dev images, with two measures. AUROC is the chance that a defective part scores higher than a normal one: 0.5 is chance. "In mask" is how often the hottest point of the map falls on the true defect. On MVTec AD, where the photos are aligned, the expert works: AUROC above 0.84, and DINOv3 puts its peak on the defect 77 % of the time, against 51 % for DINOv2. That is why we keep DINOv3: localizing is the expert's job, even if its overall AUROC is slightly lower, 0.59 against 0.61. Outside MVTec AD it degrades: VisA around 0.78, LOCO around 0.63, and GoodsAD is at chance, because the single reference is often another product. Cost: 0.35 seconds per image on our laptop.

### To add / improve
- The "Images" header wraps as "Image / s": widen the column.
- Bold the better value in each pair so the eye finds it at once.
- Add a note under "All": the overall row is weighted by images, and GoodsAD is 35 % of dev (296 / 835).

### Jury question
**"Why keep DINOv3 if its AUROC is lower?"**
The gap in AUROC is small (0.59 vs 0.61, one run) and it comes mostly from GoodsAD, where both are at chance. On localization, which is what the readers receive as a hint, DINOv3 is clearly better on MVTec AD (77 % vs 51 %) and LOCO (43 % vs 37 %). Detection itself is left to the readers and the vote, not to the expert's score alone.

---

## 5 · What worked, what did not: reference · part · our DINOv3 map

### Visual content
- Left, **✓ Worked: defect found**: MVTec AD grid; VisA chewing gum (reference, part, map)
- Right, **× Failed: reference is another product**: GoodsAD food box (green reference, pink part); MVTec LOCO juice (orange label vs banana label)
- Footer: real dev images; MMAD licences (CC BY-NC-SA 4.0, CC BY 4.0, research use)

### Script — Peter (~50 s, ~115 words)
Here is what this looks like. Each row shows the reference, the part, and our DINOv3 map. On the left, two successes: a metal grid from MVTec AD and a chewing gum from VisA. The red spot sits right on the defect. On the right, two failures with the same cause. On GoodsAD, the reference is a green box and the part is a pink box: the map lights up the whole product, because everything is different. On MVTec LOCO, the reference juice has an orange label and the part a banana label: the expert sees a difference, but it cannot tell a product variant from a defect. The lesson: with one reference, the expert is only as good as the match between the two photos.

### To add / improve
- Put a small label over each column: "Reference · Part · Map", repeated on both sides.
- Under each failed case, add the expert's score next to the decision threshold, so the jury sees the false alarm in numbers.

### Jury question
**"Why not pick a better reference, for example the most similar normal image?"**
That would be `similar_templates` or a gallery of normal images, which is exactly what SiGMA does and what our strict protocol forbids. We use only `random_templates[0]`. The point is to measure what survives with one reference that the system did not choose; a better reference can be a separate, clearly labelled variant.

---

## 6 · What went wrong: four problems found, and what we do about them

### Visual content
| Problem | Evidence | Fix / plan |
|---|---|---|
| Loader never tried on real data | crashed on "Anomaly Detection", the name `mmad.json` uses | fixed + test with the 9 real names |
| Expert threshold off-scale for DINOv3 | distances **0.11–0.31** vs range 0.2–0.8: flag never fires (bal. acc. **0.49**) | recalibrated on dev: **0.49 → 0.61** (in-sample) |
| Single reference ≠ same product | GoodsAD AUROC **0.51**, peak on defect **10 %** | trust the expert less there: weights fixed on dev |
| 8B readers not run yet | Colab not executed; local 4B run: **18 images** only | 50-question cost test, then full dev |
- Lesson: unit tests with fake data passed 20/20 while the real file broke the first line. Run on real data early.

### Script — Peter (~60 s, ~135 words)
Four problems came up, and here is what we did about each. One: our data loader had never seen the real file. It crashed on the category name "Anomaly Detection", which is what mmad.json actually uses. It is fixed, with a test on the nine real names. Two: the expert's threshold was set for distances between 0.2 and 0.8, but DINOv3 distances fall between 0.11 and 0.31, so the flag never fired: balanced accuracy 0.49. We recalibrated it on dev and reached 0.61, but that is measured on the same images, so it is optimistic. Three: the single reference is often not the same product, as on GoodsAD, so the vote will trust the expert less there. Four: the 8B readers have not run yet. The lesson: our unit tests passed on fake data while the real file broke on line one. Charles will show that first end-to-end run.

### To add / improve
- Write "20/20 at the time" in the lesson, or update it to 21/21, so it matches slide 3.
- Move "(in-sample)" next to the 0.61 in orange: it is the honest qualifier the jury will look for.

### Jury question
**"Isn't recalibrating the threshold on dev overfitting?"**
It is fitting on dev, which is what dev is for: the protocol forbids tuning on test, not on dev. The 0.61 is in-sample, so we present it as an upper estimate. The real check is the test run on Nov 19, with the threshold frozen, and we will report the expert's balanced accuracy there too.

---

## 7 · First end-to-end run: all five agents ran on real MMAD images

### Visual content
- Flow: Expert **18 s** → Reader A **323 s** → Reader B **322 s** → Verifier: **4 reopened** → Vote
- Accuracy over **88 questions**: discrimination (balanced) **52.5 %** · classification **62.5 %** · description **62.5 %** · localization **50.0 %** · analysis **87.5 %** · object classification **100 %** · object analysis **83.3 %** · **mean of 7: 71.2 %**
- Cards: **0 errors in 184 calls** (all valid JSON, ~3.7 s per question) · **Readers agree 87/88** (same model twice) · **95 % CI [54 %, 86 %]** (18 images: a working pipeline, not a result)
- Footer: laptop smoke test, RTX 3060 6 GB, 18 dev images (2–4 per source), both readers = Qwen3-VL-4B quantized (Ollama), rules-only verifier; the 8B readers run on Colab

### Script — Charles (~70 s, ~155 words)
Thanks, Peter. This is our first end-to-end run: all five agents, on real MMAD images, on the laptop's 6-gigabyte GPU. Eighteen dev images, two to four per source, 88 questions. To fit in memory, both readers were the same small model, Qwen3-VL-4B quantized, and the verifier used rules only. At the top: 18 seconds for the expert, about 320 seconds per reader, and the verifier reopened four conflicts. On the left, accuracy per subtask, with a mean of 71.2 %. Discrimination is 52.5 % balanced, close to chance, which matches the expert's weak AUROC. On the right is what we actually learned. Zero errors in 184 calls: every answer was valid JSON. The two readers agreed on 87 of 88 questions, because it was the same model twice, which is exactly why the real system uses two different families. And the confidence interval goes from 54 to 86 %: this is a working pipeline, not a result.

### To add / improve
- Add the number of questions per subtask next to each score: with 88 questions, some cells hold fewer than ten questions, and 100 % on object classification means little.
- Write "~3.7 s per reader call" instead of "per question" (323 + 322 s over 176 calls).
- Put a dashed 50 % line next to the discrimination score.

### Jury question
**"With 52.5 % on discrimination, isn't the expert making things worse?"**
We cannot tell from 18 images, and that is precisely what the "no expert" ablation measures. What we know is that the expert is reliable on aligned photos and at chance on GoodsAD, so its vote will be weighted per subtask and per its dev reliability. If it still hurts, the ablation will show it and we will report it.

---

## 8 · Examples from the run: four real cases, four behaviours

### Visual content
- Each card: reference · part · the zoom the expert sent to the readers (orange)
- **✓ Cable** (MVTec AD): 5/5 right, "Missing component, in the central part"
- **↻ Tile** (MVTec AD): verifier reopened; readers said bottom-left, expert bottom-center; still wrong (center-left)
- **× Can** (GoodsAD): false alarm, good part called defective; the reference is another brand
- **× Pushpins** (MVTec LOCO): followed the hint, "the expert hint indicates the top right": wrong place
- Takeaway: the expert's hint is powerful both ways, so its weight must depend on how reliable it was on dev

### Script — Charles (~55 s, ~130 words)
Four real cases from that run, and four different behaviours. Each card shows the reference, the part, and in orange the zoom that the expert sent to the readers. Top left, a cable from MVTec AD: five questions out of five right, with the answer "missing component, in the central part". Top right, a tile: the readers said bottom-left, the expert said bottom-center. The verifier caught the conflict and reopened it, but the final answer was still wrong: the defect is center-left. Bottom left, a can from GoodsAD: a false alarm, because the reference is another brand. Bottom right, pushpins from LOCO: the reader wrote "the expert hint indicates the top right" and followed it to the wrong place. The takeaway: the expert's hint is powerful in both directions, so its weight must depend on how reliable it was on dev. Peter?

### To add / improve
- On the tile card, add the true location ("truth: center-left") in a different colour, so the reopen story reads at a glance.
- On the pushpins card, remind that LOCO defects are logical (wrong count, missing part): a patch distance cannot see them.

### Jury question
**"The verifier reopened the tile and the answer stayed wrong. What is the verifier for?"**
Its job is to detect disagreement and force a second look, not to know the answer. In this run it was rules-only and both readers were the same 4B model, so a second look had little new to say. With two model families and the LLM critic, the "no verifier" ablation will measure what it adds; four reopens out of 88 questions also show it is not looping.

---

## 9 · Planned vs. done: what the plan promised for Nov 12

### Visual content
| Planned (Delivery I) | Status | Detail |
|---|---|---|
| MMAD download | Done | dev subset, **2.1 GB**, **2,084 files** |
| Dev and test splits fixed | Done | **835 / 7,531** images, committed |
| DINOv3 expert with anomaly maps | Done | all 835 dev images, per-source scores |
| Evaluation harness | Partial | metrics, McNemar, bootstrap coded; check vs official `summary.py` pending |
| B1 and B2 on dev | Pending | readers coded and tested, not run |
| Colab throughput measured | Pending | first task this week |
- **3 / 6** items done · **+ extra**: full multi-agent workflow coded and already run end to end on 18 dev images (planned for Nov 12–19)

### Script — Peter (~40 s, ~95 words)
Against what we promised for today: three items out of six are done. The MMAD dev download, 2.1 gigabytes and 2,084 files. The splits, fixed and committed. And the DINOv3 expert on every dev image, with scores per source. The evaluation harness is partial: metrics, McNemar and the bootstrap are coded, but not yet checked against MMAD's official summary script. B1 and B2 on dev, and the Colab throughput, are still pending. On the other hand, the full multi-agent workflow, planned for next week, is already coded and has run end to end.

### To add / improve
- Write "3 done · 1 partial · 2 pending" instead of "3 / 6": it is more honest and reads faster.
- Say why B1 and B2 slipped (Colab not yet executed) in the Detail column.

### Jury question
**"You are behind on the baselines. Is the Nov 19 freeze realistic?"**
The readers are coded and tested; what is missing is GPU time, not code. The 50-question cost test fixes the test size: 1,000 to 2,000 images depending on the measured units. Every baseline reuses the same cached reader outputs, so B3 and B4 come almost free once B1 and B2 have run.

---

## 10 · Next steps: same goal, adjusted path to Nov 26

### Visual content
- **Nov 12–15 · Readers on dev**: 50-question cost test on Colab L4, then B1 and B2 on the **3,937** dev questions
- **Nov 15–19 · Calibrate on dev**: expert threshold, vote weights per reader and subtask, prompts; official `summary.py` check
- **Nov 19 · Freeze · Test, run once**: S vs B1–B4, ablations; **~1,000–2,000** test images depending on measured cost
- **Nov 26 · Final**: McNemar + bootstrap, error analysis per source, report
- Adjustment: the expert's vote will depend on its dev reliability for each subtask; results always reported per source (GoodsAD and LOCO apart)

### Script — Peter (~45 s, ~110 words)
The goal does not change, but the path is adjusted. Until the fifteenth: the 50-question cost test on a Colab L4, then B1 and B2 on the 3,937 dev questions. From the fifteenth to the nineteenth: calibration on dev, meaning the expert's threshold, the vote weights per reader and subtask, the prompts, and the check against the official summary script. On the nineteenth we freeze everything and run the test once: the full system against B1 to B4, plus the ablations, on 1,000 to 2,000 images depending on the measured cost. On the twenty-sixth: McNemar, bootstrap, error analysis per source, and the report. One adjustment: the expert's vote will depend on its reliability on dev. Charles?

### To add / improve
- Add the Colab budget (100 units, never A100) and the plan B (+100 units) under the first box.
- Mark the freeze as a hard line on the timeline, e.g. "no change after this point".

### Jury question
**"What if the 50-question test shows Colab is too expensive?"**
Then the test shrinks towards 1,000 images, which still gives enough paired questions for McNemar, and we switch to the T4 in 4-bit if the L4 burns too many units. The cost is measured before the test size is fixed, precisely so that it never forces us to change the protocol.

---

## 11 · Thank you!

### Visual content
- Kicker: OneRef · Delivery II
- **Thank you! Questions?**
- Code and results: github.com/RVilchezzz/OneRef
- Image: DINOv3 map on the VisA chewing gum

### Script — Charles (~10 s, ~25 words)
Thank you for your attention. The code and the results are public at github.com/RVilchezzz/OneRef. We are happy to take your questions.

### To add / improve
- Nothing to change. Keep slide 4 ready to go back to: most questions will be about it.

### Jury question
**"What is the one number we should remember today?"**
77 %: on aligned photos, our expert's peak falls on the true defect three times out of four, from a single reference. Everything else, including whether the full team beats Qwen3-VL-8B, will be measured on the test after Nov 19.

---

## Backup Q&A

**"Why not fine-tune the expert or the readers on MMAD?"**
The protocol is training-free by design: MMAD is our test bed, and training on it would make the comparison with the leaderboard meaningless. Anything trained on MMAD belongs to a different track.

**"Is the AUROC of 0.59 good enough to be useful?"**
Not as a detector on its own, and we do not use it that way. On MVTec AD it reaches 0.84 to 0.88; overall it is dragged down by GoodsAD. Its main value is localization, passed to the readers as a zoomed crop and a 3×3 region.

**"Why is GoodsAD so hard?"**
The photos are not aligned and the single reference is often another product or brand, so a patch distance flags the whole object. We keep it and report it apart rather than drop it: dropping it would inflate our numbers.

**"What exactly does the expert send to the readers?"**
A 3×3 region, a zoomed crop around the peak and a score. In the ablations we also compare a raw map, a calibrated map and a text-only hint.

**"How will you prove the readers are not simply copying the expert?"**
The pushpins case shows they sometimes do. The "no expert" ablation measures the net effect, and the error analysis will count how often a wrong hint led to a wrong answer, with MAST categories for orchestration failures.

**"Why Ollama locally and vLLM on Colab?"**
vLLM does not run on Windows and 6 GB cannot hold an 8B model in bf16. Ollama serves a quantized 4B model through the same OpenAI-compatible API, so the code is identical; only the endpoint changes.

**"Is 18 images enough to say anything?"**
Only that the pipeline works: zero format errors in 184 calls, the verifier reopens without looping, and logs are complete. The accuracy figures are not a result, which is why we show the confidence interval [54 %, 86 %] on the slide.

**"Is the test set really untouched?"**
Yes. It is fixed by a committed seed, disjoint from dev, and the CLI refuses `--split test` without a frozen configuration. It will run once, after Nov 19.
