# Le pipeline lancé pour la Delivery II, expliqué pas à pas (avec les vraies traces)

Pour Charles, à lire avant l'oral. Ce fichier explique **ce qui a réellement tourné** pour produire les slides 3 à 8, à partir des journaux du run, et relie chaque chiffre des slides au fichier qui le prouve.

**Sources** (branche `origin/peter/expert-dev`, commit `bbe4853` du 12 nov.) :
- `docs/reports/2026-11-11_first_end_to_end_run.md` : le rapport de Peter (environnement, bugs, trois runs, cas détaillés) ;
- `results/runs/<run>/telemetry.jsonl` : une ligne par appel d'agent et par passe (prompts, réponses brutes, jetons, temps) ;
- `results/runs/<run>/1_expert/` … `7_aggregator/` : l'état complet de chaque image après chaque passe ;
- `results/pipeline_smoke/<run>/per_question.csv` et `summary.json` : le tableau des 88 questions et le résumé ;
- `results/expert_eval/` : l'expert seul sur les 835 images ;
- le code (`src/inspector/…`) et `configs/2026-11-12_local_ollama_smoke.env`.

**Convention :** tout ce qui suit est lu dans ces fichiers. Ce qui est **mon analyse** des traces (et non un chiffre du rapport) est marqué ainsi.

---

## 0. Ce que les traces ont changé par rapport à ce qu'on croyait

1. **Le run de 18 images a bien utilisé `--per-source 4`** (`--run-name smoke_ollama_qwen4b_ps4`). La commande avec `--per-source 2` est celle du run de 9 images.
2. **« 4 reopened » = 4 questions, sur 4 images différentes.** Les deux lectures sont vraies.
3. **Le résultat le plus important n'est pas sur la slide (mon analyse) :** le oui ou non final de la discrimination **suit le drapeau de l'expert sur 17 images sur 18**. Ce n'est pas le vote qui le force : ce sont les lecteurs qui reprennent l'indice du prompt, et la règle de réouverture les pousse dans le même sens. Conséquence : **7 pièces saines sur 10 rejetées**.
4. **Les 4 réouvertures ont un bilan nul :** une réponse corrigée, une cassée, deux restées fausses.
5. **L'intervalle de confiance** est [52,4 ; 85,2] dans le rapport et [53,8 ; 85,6] dans la sortie de la CLI (arrondi en [54, 86] sur la slide). Même moyenne : seul l'ordre de tirage du bootstrap change.

---

## 1. Vue d'ensemble

```
python -m inspector.main --split dev --per-source 4 --run-name smoke_ollama_qwen4b_ps4   (+ le .env du run)
        │
        ▼
  18 images dev (4 par source, moitié normales / moitié défectueuses ; 2 pour la petite étiquette MVTec-AD) → 88 questions
        │
        ▼   7 passes, chacune sur TOUTES les images, un seul modèle chargé à la fois
  1_expert ─▶ 2_reader_a ─▶ 3_reader_b ─▶ 4_critic ─▶ 5_reader_a_reopen ─▶ 6_reader_b_reopen ─▶ 7_aggregator
   18,0 s      322,9 s        322,0 s       0,4 s         15,7 s (4 q.)         5,7 s (4 q.)          0,4 s
        │
        ▼
  réponses finales → 88 lignes « juste / faux » → exactitude par sous-tâche, moyenne de 7, IC bootstrap   (~11 min au total)
```

**Comment les passes s'enchaînent** (`orchestrator/pass_runner.py`, `orchestrator/graph.py`) :
- L'ordre est fixé en Python (`build_steps`), pas décidé par un LLM : l'orchestrateur n'est pas un agent.
- Chaque passe charge **un seul** modèle, traite toutes les images, le décharge et libère le GPU. La télémétrie le montre : chaque `pass_start` est suivi de son `pass_end` avant la passe suivante.
- Après chaque passe, l'état complet de chaque image est écrit dans `<run>/<passe>/<image>.json`. Si on relance, tout ce qui est en cache est relu sans recharger le modèle.
- Les passes 5 et 6 ne traitent que les images marquées par le vérificateur : `to_run: 4` dans la télémétrie, contre 18 pour les autres passes.

---

## 2. Les trois runs (dans l'ordre)

| | Run 1 `smoke_ollama_qwen4b_ps2` | Run 2 `…_ps2_v2` | **Run 3 `…_ps4` (slides 7-8)** |
|---|---|---|---|
| Images (défectueuses) | 9 prévues | 9 (4) | **18 (8)** |
| Par source | — | DS-MVTec 2, GoodsAD 2, MVTec-AD 1, LOCO 2, VisA 2 | DS-MVTec 4, GoodsAD 4, MVTec-AD 2, LOCO 4, VisA 4 |
| Issue | **Arrêté** après 3 images : une photo GoodsAD envoyée en pleine résolution fait 12 492 jetons pour un contexte de 8 192, et l'exception arrête la passe | Complet | Complet |
| Questions | — | 45 | **88** |
| Moyenne des 7 | — | 64,4 % [37,0 ; 81,9] | **71,2 %** [52,4 ; 85,2] |
| Pièces saines rejetées / défauts manqués | — | 60 % / 25 % | **70 % / 25 %** |
| Appels lecteurs / invalides | — | 94 / 0 | **184 / 0** |
| Questions rouvertes | — | 2 (2 images) | **4 (4 images)** |
| Accord A = B | — | 45/45 | **87/88** |

Le commit `353647e` mentionne « 9 images, 2 reopens » : c'est le run 2. La slide présente le run 3.

**Configuration commune aux runs 2 et 3** (`.env` du run, rapport §2 et §7.1) :

| Élément | Valeur |
|---|---|
| Machine | RTX 3060 Laptop 6 Go, Python 3.14, torch 2.11, pydantic-ai-slim 2.66 |
| Lecteurs A et B | `qwen3-vl:4b-instruct`, quantifié Q4 (3,3 Go), servi par **Ollama 0.40.7** avec `OLLAMA_CONTEXT_LENGTH=8192` |
| Vérificateur | **règles seules** (`ONEREF_USE_LLM_CRITIC=false`) |
| Expert | DINOv3 ViT-B/16 gelé, 448 px |
| Calibration | `ANOMALY_NORM_LOW=0.065`, `HIGH=0.265` (pic 0,165 → score 0,5 = drapeau) |
| Lecteurs | température 0, 256 jetons max, images ≤ 768 px, 1 relance sur JSON invalide |
| Vote | tous les poids à 1,0 (non réglés) ; l'expert vote sur la discrimination avec un poids de 1,0 |

**Pourquoi Ollama et un 4B** (rapport §8) : le téléchargement des poids bf16 de Qwen3-VL-4B depuis Hugging Face (8,9 Go) avançait à ~110 Ko/s, puis ~2 Mo/s sans Xet (environ 1 h). Peter l'a remplacé par le modèle quantifié d'Ollama, 3,3 Go.

---

## 3. Passe 1 : l'expert visuel (`agents/visual_expert.py`)

**Ce qu'il calcule :**
1. Pièce et référence (`random_templates[0]`, la seule) redimensionnées en 448×448 et passées dans DINOv3 : 28×28 = 784 patchs de 16 px, un vecteur chacun.
2. Pour chaque patch de la pièce : `1 − similarité cosinus avec le patch le plus proche de la référence`. Prendre le plus proche tolère les petits décalages.
3. La carte est agrandie à la taille de l'image. Le **pic** donne la case de la grille 3×3 et un zoom (un tiers de l'image, agrandi ×2).
4. Score : `(pic − 0,065) / 0,2`, borné entre 0 et 1. Drapeau « défaut » si score ≥ 0,5.

**Ce qu'il a produit sur les 18 images** (rapport §7.4) :

| Image | Source | Vérité | Région de l'expert | Score | Drapeau |
|---|---|---|---|---|---|
| cable missing_cable/009 | MVTec AD | défaut | center | 1,00 | oui |
| carpet good/019 | MVTec AD | normal | bottom left | 0,73 | **oui (faux)** |
| tile oil/010 | MVTec AD | défaut | bottom center | 1,00 | oui |
| transistor good/003 | MVTec AD | normal | center | 0,28 | non |
| cigarette_box good/039_003 | GoodsAD | normal | center | 1,00 | **oui (faux)** |
| drink_can good/019_002 | GoodsAD | normal | bottom center | 1,00 | **oui (faux)** |
| food_box deformation/027_008 | GoodsAD | défaut | center | 1,00 | oui |
| food_package broken/069_009 | GoodsAD | défaut | center | 1,00 | oui |
| pill good/010 | MVTec-AD | normal | middle left | 0,27 | non |
| pill good/011 | MVTec-AD | normal | bottom center | 0,40 | non |
| juice_bottle good/010 | LOCO | normal | center | 0,81 | **oui (faux)** |
| pushpins logical/071 | LOCO | défaut | top left | 0,55 | oui |
| pushpins logical/080 | LOCO | défaut | top right | 0,40 | **non (raté)** |
| screw_bag good/118 | LOCO | normal | middle left | 0,73 | **oui (faux)** |
| candle good/0730 | VisA | normal | middle left | 0,32 | non |
| macaroni2 bad/058 | VisA | défaut | center | 0,85 | oui |
| macaroni2 good/0294 | VisA | normal | top left | 0,85 | **oui (faux)** |
| pcb3 bad/068 | VisA | défaut | middle left | 0,32 | **non (raté)** |

- **Les 4 images GoodsAD ont toutes le score maximal de 1,00**, y compris les deux saines : c'est le problème de la référence qui est un autre produit (slide 6, problème 3).
- **Mon analyse :** sur ces 18 images, le drapeau de l'expert seul aurait 40 % de réussite sur les pièces saines (4 sur 10) et 75 % sur les défauts (6 sur 8), soit **57,5 %** d'exactitude équilibrée.

**Le texte de l'indice** (`as_text()`, copié de la télémétrie, transistor) :
> « Visual expert (automatic patch comparison with the reference): most different region = center, difference level = low (0.28). This is a hint, not ground truth. »

Le niveau est « low » sous 0,33, « medium » sous 0,66, « high » au-dessus.

---

## 4. Passes 2 et 3 : les deux lecteurs (`agents/reader_mllm.py`)

**Ce que reçoit un lecteur, pour chaque question** (prompt réel, télémétrie) :
- 3 images : la pièce, la référence, le zoom de l'expert, réduites à 768 px au maximum ;
- un message système d'inspecteur (« Image 1 est la pièce, image 2 une référence normale, image 3 un zoom sur la région la plus différente… réponds par une lettre, une confiance, une phrase ») ;
- le texte de l'indice ci-dessus ;
- la question et ses options, par exemple « Question: Is there any defect in the object? Options: A. Yes. B. No. »

**Ce qu'il renvoie** (réponse brute, transistor) :
`{"selected_option": "B", "confidence": 0.95, "brief_rationale": "The component appears structurally sound with no visible defects or anomalies compared to the reference."}`

**Les chiffres :**
- **~3 476 jetons en entrée par appel** en moyenne : les trois images en représentent l'essentiel. Le transistor (5 questions) consomme 17 099 jetons d'entrée et 218 de sortie, en 18,7 s.
- **3,67 s par question et par lecteur** (`summary.json`) : 322,9 s pour 88 questions. C'est le « ~3.7 s » de la slide, qui vaut **par appel d'un lecteur**.
- **0 réponse invalide sur 184 appels.** La protection (lettre absente des options → une relance ; toujours invalide ou requête refusée → réponse vide, comptée fausse) n'a jamais eu à servir dans le run 3. Elle a servi à rendre le run possible après le run 1.

**Les deux lecteurs sont le même modèle**, à température 0, avec les mêmes entrées : ils ne diffèrent que sur 1 question sur 88 (étiquette du jus de fruits, voir section 7).

---

## 5. Passe 4 : le vérificateur, règles seules (`agents/critic_verifier.py`)

**Les règles actives** (pour chaque lecteur, chaque question) :

| Règle | Déclenchement | Effet |
|---|---|---|
| 1. Réponse manquante | réponse vide | image invalide, question rouverte |
| 2a. « Pas de défaut » contredit | discrimination « No » **et** score de l'expert ≥ 0,5 | question rouverte |
| 2b. Localisation contredite | la case citée dans l'option choisie ≠ région de l'expert **et** score ≥ 0,5 | question rouverte |

La règle 2c (« No » alors qu'une question de localisation existe) est **désactivée** : en MMAD, la présence de cette question révèle l'étiquette. Un désaccord entre lecteurs n'est **pas** un conflit : c'est le vote qui tranche. Une seule réouverture par image.

**Les 4 verdicts du run 3** (copiés de la télémétrie) :

| Image | Règle | Message du vérificateur |
|---|---|---|
| carpet good/019 | 2a | « reader_a: said no defect but the visual expert found a strong difference in the bottom left. reader_b: … » |
| tile oil/010 | 2b | « reader_a: chose 'bottom left' but the expert's strongest region is 'bottom center'. reader_b: … » |
| pushpins logical/071 | 2a | « reader_a: said no defect but the visual expert found a strong difference in the top left. reader_b: … » |
| macaroni2 bad/058 | 2b | « reader_a: chose 'top right' but the expert's strongest region is 'center'. reader_b: … » |

**Deux défauts de conception visibles dans ces traces (mon analyse) :**
- **Macaronis :** les options sont « Bottom left / Top left / Bottom right / Top right ». La région de l'expert, « center », **n'est pas une option**. La règle 2b se déclenche donc quel que soit le choix du lecteur.
- **Règle 2a :** elle ne rouvre que les « No » contredits par l'expert, jamais les « Yes » sur une pièce que l'expert juge saine. Elle ne peut donc pousser les lecteurs **que vers « défaut »**.

---

## 6. Passes 5 et 6 : la réouverture

Seules les 4 questions marquées sont reposées, aux deux lecteurs, avec une phrase en plus (prompt réel, tapis) :
> « A reviewer flagged a possible inconsistency in your previous answers: reader_a: said no defect but the visual expert found a strong difference in the bottom left. reader_b: … Reconsider carefully. »

**Ce que les réouvertures ont changé** (`per_question.csv`, colonnes `reader_a_first` → `reader_a_final`) :

| Image | Question | Avant | Après | Vérité | Bilan |
|---|---|---|---|---|---|
| pushpins logical/071 | défaut ? | No | **Yes** | Yes | **corrigée** |
| carpet good/019 | défaut ? | No | **Yes** | No | **cassée** |
| tile oil/010 | où ? | Bottom-left | Bottom-left | Center-left | inchangée, fausse |
| macaroni2 bad/058 | où ? | Top right | Top right | Top left | inchangée, fausse |

**Bilan nul.** Sur le tapis, le lecteur justifie son changement par l'indice : *« The visual expert flagged a high-difference region in the bottom left, suggesting a defect… »*.

Calcul des appels : 88 × 2 = 176, plus 4 × 2 = 8 → **184**.

---

## 7. Passe 7 : l'agrégateur (`agents/aggregator.py`)

Pour chaque question : chaque lecteur vote pour sa lettre (poids 1,0). Sur la discrimination, l'expert vote aussi (« Yes » si drapeau, sinon « No », poids 1,0). En cas d'égalité, la plus grande somme de confiances l'emporte, puis l'ordre alphabétique.

**Ce que ça donne dans les fichiers** (`7_aggregator/juice_bottle__test__good__010.json`) :
- discrimination : `{"B": 3.0}`, soit les deux lecteurs **et** l'expert pour « Yes » (faux : la bouteille est saine) ;
- question de l'étiquette : `{"D": 1.0, "A": 1.0}`, la **seule** question où les lecteurs diffèrent. L'égalité est départagée et donne A, la bonne réponse.

**Mon analyse :** avec deux lecteurs identiques, le vote de l'expert ne change jamais le résultat. L'expert pèse **en amont**, par l'indice dans le prompt et par la réouverture.

---

## 8. Le résultat qui compte : la discrimination suit l'expert

**Mon analyse**, à partir de `per_question.csv` et de la table de la section 3 :

| | Drapeau de l'expert : oui | Drapeau de l'expert : non |
|---|---|---|
| **Pièces saines (10)** | 6, **toutes rejetées** | 4, dont 3 acceptées (pill 011 rejetée) |
| **Pièces défectueuses (8)** | 6, toutes détectées (pushpins 071 après réouverture) | 2, **toutes deux ratées** |

- Le oui ou non final = le drapeau de l'expert sur **17 images sur 18**. Seule exception : pill 011 (score 0,40, « medium »), rejetée quand même.
- Avant la réouverture, les lecteurs suivaient déjà le drapeau sur 15 images. Les 2 réouvertures en discrimination (tapis, punaises 071) les ont alignés sur l'expert.
- Les justifications le disent : *« The visual comparison highlights a high difference level (0.81) in the center region, indicating a defect »* (bouteille de jus saine).
- D'où les chiffres du rapport : **70 % de pièces saines rejetées, 25 % de défauts manqués, 52,5 % d'exactitude équilibrée**. C'est moins que le drapeau de l'expert seul (57,5 %) sur ces 18 images, mais l'écart tient à une seule image (pill 011) : ce n'est pas significatif.
- **Ce que ça ne prouve pas :** l'effet net de l'expert. Il n'existe pas encore de run « sans expert » (rapport §9). On voit une **dépendance**, pas une perte mesurée.

C'est la conclusion du rapport (§7.6) : le poids de l'expert, et le fait même de montrer l'indice, doivent dépendre de sa fiabilité sur le dev, par sous-tâche et par source.

---

## 9. Les métriques et le nombre de questions derrière chaque case

| Sous-tâche | Questions | Justes | Exactitude |
|---|---|---|---|
| Discrimination (équilibrée) | 18 | 3/10 saines + 6/8 défauts | **52,5 %** |
| Classification du défaut | 8 | 5 | 62,5 % |
| Localisation du défaut | 6 | 3 | 50,0 % |
| Description du défaut | 8 | 5 | 62,5 % |
| Analyse du défaut | 8 | 7 | 87,5 % |
| Classification de l'objet | 10 | 10 | 100 % |
| Analyse de l'objet | 30 | 25 | 83,3 % |
| **Moyenne des 7** | **88** | | **71,2 %** |

- Dans ces 18 images, les questions sur le défaut ne portent que sur les 8 images défectueuses, et les questions sur l'objet que sur les 10 saines. Deux images défectueuses (food_box, pushpins 071) n'ont pas de question de localisation : d'où 6 et non 8.
- Discrimination en exactitude **équilibrée** = (30 % + 75 %) / 2. Les autres sous-tâches en exactitude simple. Une réponse vide compterait fausse.
- **IC 95 %** : bootstrap par image, 1 000 tirages (`utils/metrics.py`). [52,4 ; 85,2] dans le rapport, [53,8 ; 85,6] dans la CLI : même moyenne.

---

## 10. Slide par slide : d'où vient chaque chiffre

### Slide 3 (Charles) « The full workflow exists as code »
| Sur la slide | Preuve | À dire |
|---|---|---|
| Visual expert « Ran on all 835 dev » | `results/expert_eval/dev_dinov3-vitb16_v1/` | Vrai |
| Readers « Qwen3-VL · InternVL3 via vLLM », « Ran on real data » | `.env` du run | **À nuancer** : c'est Qwen3-VL-**4B** via **Ollama**, deux fois ; InternVL3 et vLLM n'ont pas tourné |
| Verifier « rules + LLM critic », « Ran on real data » | `.env` : `USE_LLM_CRITIC=false` | **À nuancer** : règles seules |
| Aggregator, Orchestrator « Ran on real data » | `7_aggregator/`, télémétrie | Vrai |
| 21 / 21 tests | rapport §5 | 20 → 21 avec le test du chargeur ; **22** aujourd'hui avec le test de `--per-source` |
| 835 / 7 531 | `data/splits/*.json`, rapport §3.2 | Vrai ; 3 937 / 35 733 questions ; 742 références distinctes dans le dev |
| 2,1 Go au lieu de 28 | rapport §3.3 | 2 084 fichiers, 0 manquant |

### Slides 4-5 (Peter) : l'expert seul sur 835 images
Tous les chiffres sont dans `results/expert_eval/*/summary.json` et le rapport §6. Ce qu'il faut savoir en plus :
- les variantes de score essayées (pic − moyenne, pic / moyenne) n'améliorent pas l'AUROC : on garde le pic brut ;
- exactitude équilibrée par source au seuil 0,165 : MVTec AD 0,727, VisA 0,695, LOCO 0,588, GoodsAD 0,500 ;
- 0,35 s par image, environ 5 min pour tout le dev.

### Slide 6 (Peter) : les quatre problèmes
| Problème sur la slide | Preuve exacte (rapport §4) |
|---|---|
| Le chargeur plante sur « Anomaly Detection » | `ValueError: Unknown MMAD subtask type: 'Anomaly Detection'` ; corrigé en `711294a` |
| Seuil hors échelle (0,49 → 0,61) | distances DINOv3 0,11-0,31 pour une échelle de 0,2-0,8 ; corrigé en `c15bc0b` |
| GoodsAD : AUROC 0,51, pic sur le défaut 10 % | `summary.json` ; dans le run, score 1,00 sur les 4 images GoodsAD |
| Lecteurs 8B pas lancés | rapport §9 |
| (pas sur la slide) Contexte dépassé, passe interrompue | run 1 : 12 492 jetons pour 8 192 ; corrigé en `353647e` (768 px, requête refusée = erreur) |

### Slide 7 (Charles) : le run de 18 images
| Sur la slide | Preuve | Précision |
|---|---|---|
| Expert 18 s · Reader A 323 s · Reader B 322 s | `summary.json` : 18,0 / 322,9 / 322,0 | Les passes de réouverture (15,7 + 5,7 s) ne sont pas affichées ; ~11 min au total |
| Verifier : 4 reopened | 4 verdicts de la télémétrie | 4 questions sur 4 images ; bilan nul (section 6) |
| Tableau des 7 sous-tâches | `per_question.csv` | Voir le nombre de questions, section 9 |
| 0 errors in 184 calls, ~3,7 s | `reader_calls: 184`, `reader_call_errors: 0`, 3,67 s | Par question **et par lecteur** |
| Readers agree 87/88 | `reader_agreement: [87, 88]` | La seule différence : l'étiquette du jus |
| 95 % CI [54 %, 86 %] | sortie CLI [53,8 ; 85,6] | Le rapport recalcule [52,4 ; 85,2] |
| Pied de page « 2–4 per source » | `by_source_images` | 4 par source, 2 pour la petite étiquette MVTec-AD |
| (absent) | `discrimination_fpr: 0.7`, `miss_rate: 0.25` | **70 % de pièces saines rejetées** |

### Slide 8 (Charles) : les quatre cas, relus dans les traces
| Carte | Ce que montrent les traces |
|---|---|
| **✓ Cable** | Expert « center », 1,00. 5/5 justes : « Missing component », « In the central part of the cable », « Two large voids where wires should be », « Missing cable components, critical for functionality ». Confiance 0,95 partout. |
| **↻ Tile** | Expert « bottom center », 1,00. Les deux lecteurs : « Bottom-left ». Règle 2b → réouverture → toujours « Bottom-left » ; vérité « Center-left ». L'expert se trompait aussi de case. Le type (fissure au lieu de décoloration) est faux aussi : 3/5. |
| **✗ Can** | Canette **saine**, expert 1,00 (« bottom center »), la référence est une autre marque. Réponse « Yes » (0,95) : *« the can's label is misaligned and distorted »*. Les 4 questions sur l'objet sont justes, y compris « beer can », lu sur l'inscription chinoise. 4/5. |
| **✗ Pushpins** | Image **080** : expert « top right », **0,40, sous le seuil** → pas de drapeau, donc pas de réouverture. Discrimination « No » (faux). Localisation : *« The visual expert hint indicates the top right region as the most different, and the zoomed crop confirms a defect there »* ; vérité « Bottom center ». Défaut réel : un séparateur manquant (logique). **0/5.** |

**Deux autres cas utiles, absents de la slide :**
- **Tapis sain (carpet good/019)** : la réouverture a **cassé** une bonne réponse (« No » → « Yes »).
- **Punaises 071** : la réouverture a **corrigé** la discrimination (« No » → « Yes »). La même image de punaises sert donc d'exemple dans les deux sens.

---

## 11. Ce qui a été corrigé en route pour que le run passe

| # | Problème | Correction | Commit |
|---|---|---|---|
| 1 | Le chargeur plantait sur « Anomaly Detection » (les tests sur données factices passaient 20/20) | Correspondance ajoutée + test sur les 9 noms réels | `711294a` |
| 2 | Photos GoodsAD trop grandes : 12 492 jetons pour 8 192 | Images > 768 px envoyées en JPEG réduit | `353647e` |
| 3 | Une requête refusée arrêtait toute la passe (run 1 arrêté après 3 images) | Requête refusée = réponse vide, comptée comme erreur | `353647e` |
| 4 | Seuil de l'expert hors échelle | Recalibré sur le dev à 0,165 (in-sample) | `c15bc0b` |
| — | Tirage d'un petit sous-ensemble équilibré | Option `--per-source K` | `c15bc0b` |

Le « 0 erreur sur 184 appels » est donc valable **après** ces corrections.

---

## 12. Écarts d'implémentation par rapport à la Delivery I

Le tableau complet est dans `delivery_I_vs_II_comparison.fr.md`. En bref :
- **Lecteurs :** pydantic-ai et un serveur compatible OpenAI (Ollama pour ce run), et non `transformers` ; **pas deux familles**, le même Qwen3-VL-4B deux fois.
- **Vérificateur :** règles + critique LLM optionnelle ; le run était en **règles seules**, et la règle 2c est désactivée.
- **Expert :** ViT-B/16 à 448 px, **un seul seuil global** ; ni filtre de régions connexes ni tête de comparaison.
- **Agrégateur :** poids tous à 1,0 ; pas de pondération par l'accord avec l'expert.
- **Orchestrateur :** séquence fixe de 7 passes, sans routage par sous-tâche.
- **Ajouts non annoncés :** réduction à 768 px, requête refusée = erreur, `--per-source`, sous-ensemble de 2,1 Go, copie des sorties dans `results/runs/`.

---

## 13. Ce qui n'est PAS dans ce run, et l'état du dépôt

**Pas dans ce run** (rapport §9) : lecteurs 8B, InternVL3, critique LLM, poids réglés, run « sans expert », baselines B1 à B4, comparaison avec le `summary.py` officiel, plusieurs graines, test (7 531 images). `scripts/quick_reader_baseline.py` (avec ou sans indice) est écrit mais **jamais lancé**.

**État du dépôt :**
- Les sorties du run sont maintenant dans le dépôt, **sur la branche de Peter** (`results/runs/`, images en JPEG 800 px). Les fichiers JSON citent encore les `.png` d'origine.
- `main` ne contient toujours rien de tout cela (dernier commit le 10 nov.) : la PR n° 3 de `peter/expert-dev` est à fusionner avant que le jury ouvre le dépôt.

---

## 14. Chiffres à garder en tête

| Chiffre | Origine |
|---|---|
| 18 images, 88 questions, `--per-source 4` | `summary.json`, rapport §7.1 |
| 18,0 / 322,9 / 322,0 / 0,4 / 15,7 / 5,7 / 0,4 s | les 7 passes, ~11 min |
| 184 appels, 0 invalide, ~3 476 jetons par appel | 2 × 88 + 4 × 2 |
| 4 questions rouvertes, 4 images, bilan nul | 1 corrigée, 1 cassée, 2 inchangées |
| 17 / 18 | oui/non final = drapeau de l'expert |
| 70 % / 25 % | pièces saines rejetées / défauts manqués |
| 71,2 %, IC [52,4 ; 85,2] (CLI : [53,8 ; 85,6]) | « a working pipeline, not a result » |
| 22 tests | rapport §5 (21 sur la slide) |
