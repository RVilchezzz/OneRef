# Delivery II · Ma partie (Charles) en détail : ce que le pipeline a réellement testé et exécuté

Pour Charles, à lire avant l'oral et à garder sous la main pour les questions du jury. Ce n'est **pas** un texte à dire en entier : le script de 10 min est dans `speaker_script_delivery_II.fr.md`. Ici, on développe les slides **3, 7 et 8** (et un peu 2 et 11) pour savoir répondre à « mais concrètement, qu'est-ce qui a tourné ? ».

**Fichiers liés :**
- `pipeline_run_explained_delivery_II.fr.md` : le pipeline passe par passe, avec les vraies traces, et d'où vient chaque chiffre des slides (section 10) ;
- `delivery_I_vs_II_comparison.fr.md` : ce qui était annoncé en Delivery I contre ce qui existe.

**Sources :** le PDF du deck ; sur la branche `peter/expert-dev` (commit `bbe4853`), le rapport de Peter `docs/reports/2026-11-11_first_end_to_end_run.md`, les sorties du run dans `results/runs/smoke_ollama_qwen4b_ps4/` et le tableau `results/pipeline_smoke/smoke_ollama_qwen4b_ps4/per_question.csv` ; `tests/test_workflow.py`. Les citations des slides restent en anglais. Ce qui est **mon analyse** des traces (et pas un chiffre du rapport) est marqué ainsi.

---

## 0. L'idée à retenir en 20 secondes

On a **trois niveaux de preuve**, du plus faible au plus fort. Il faut toujours dire de quel niveau on parle :

| Niveau | Quoi | Modèles | Ce que ça prouve | Ce que ça ne prouve pas |
|---|---|---|---|---|
| **1. Tests logiciels** (slide 3) | `pytest`, sans GPU | **Simulés** (FunctionModel / mock) | Le code est correct : schémas, règles, vote, métriques, reprise | Rien sur la qualité des réponses |
| **2. Run de bout en bout** (slides 7-8) | 18 images dev, 88 questions | **Réels** mais petits : Qwen3-VL-4B quantifié ×2, vérificateur à règles | La chaîne complète tourne sur de vraies images MMAD, sans erreur de format | Aucun résultat statistique (IC 95 % ≈ [52, 86]) |
| **3. Mesures de l'expert** (slides 4-5, Peter) | 835 images dev | DINOv3 contre DINOv2 | L'expert localise (77 % contre 51 % dans le masque sur MVTec AD) | Rien sur les lecteurs 8B |

Phrase passe-partout : « Le niveau 1 prouve que le code est juste, le niveau 2 que la chaîne tient sur de vraies images, mais aucun des deux n'est encore un résultat de précision. Le résultat viendra de Colab avec les lecteurs 8B. »

**Ce que les traces du run ont appris de plus (à garder en tête)** :
1. Le oui ou non final de la discrimination **suit le drapeau de l'expert sur 17 images sur 18** : 7 pièces saines sur 10 rejetées (mon analyse, section 2).
2. Les 4 réouvertures ont un **bilan nul** : 1 réponse corrigée, 1 cassée, 2 restées fausses.
3. La slide 3 **surestime** ce qui a tourné : ni InternVL3, ni vLLM, ni la critique LLM.

---

## 1. Slide 3, « The full workflow exists as code » : les tests et ce qui a tourné

### Ce que dit la slide, et ce qu'il faut corriger à l'oral
| Carte ou chiffre | Vrai ? | À dire |
|---|---|---|
| Visual expert « Ran on all 835 dev » | Oui | `results/expert_eval/dev_dinov3-vitb16_v1/` |
| Readers « Qwen3-VL · InternVL3 via vLLM », « Ran on real data » | **En partie** | Seul **Qwen3-VL-4B quantifié**, servi par **Ollama**, a tourné, deux fois. InternVL3 et vLLM : pas encore |
| Verifier « rules + LLM critic », « Ran on real data » | **En partie** | Les **règles seules** ont tourné (`USE_LLM_CRITIC=false`) ; la critique LLM n'a tourné qu'en test simulé |
| Aggregator, Orchestrator « Ran on real data » | Oui | `7_aggregator/`, télémétrie des 7 passes |
| **21 / 21** tests | Oui au moment du deck | **22** aujourd'hui (rapport §5) |
| **835 / 7,531** | Oui | 3 937 / 35 733 questions ; 742 références distinctes dans le dev |
| **2.1 GB** au lieu de 28 | Oui | 2 084 fichiers lus dans les zips distants par requêtes HTTP *range*, 0 manquant |

Le script de la slide 3 a été corrigé : « prévus en Qwen3-VL et InternVL3 ; pour l'instant, c'est un Qwen 4B qui a tourné, via Ollama ».

### Comment est fait un test « sans GPU »
Aucun vrai modèle n'est appelé. Le fichier désactive même les requêtes réelles (`models.ALLOW_MODEL_REQUESTS = False`). Quand un agent a besoin d'un modèle, on lui donne un `FunctionModel` **scripté** : une fonction Python qui répond ce qu'on a décidé (par exemple d'abord une mauvaise réponse, puis une bonne). On vérifie le **comportement du code autour du modèle**, pas le modèle.

### Groupe par groupe (noms réels dans `tests/test_workflow.py`)

**A. Schémas Pydantic : les mauvaises données sont rejetées**
- `test_schema_rejects_bad_values` : une option « E », une confiance de 1,5, une région « nowhere », une réponse « C » absente des options d'une question oui/non : tout lève une `ValidationError`.
- *Pourquoi :* les agents s'échangent un JSON propre ; une réponse mal formée ne peut pas contaminer la suite.

**B. Protocole strict : une seule référence, pas de texte de domaine**
- `test_strict_one_shot_reference` : la référence est toujours `random_templates[0]`.
- `test_load_mmad_ignores_similar_templates` : le chargeur ignore les `similar_templates`.
- *Pourquoi :* c'est la règle n° 1 du protocole, celle qui nous distingue de SiGMA.

**C. Lecteurs : la relance sur format invalide**
- `test_reader_agent_retries_on_option_not_offered` : réponse « C » sur une question oui/non → `ModelRetry` → la 2ᵉ réponse est gardée.
- `test_reader_agent_gives_none_after_retries` : jamais de JSON → réponse `None`, chaque appel porte un champ `error`.
- *Lien avec la slide 7 :* le « 0 errors in 184 calls » est mesuré, et le mécanisme qui gère les erreurs est testé.

**D. Vérificateur : règles, réouverture, LLM**
- `test_critic_conflict_reopens_once` : conflit → réouverture, `reopen_count = 1` ; un 2ᵉ passage n'en donne **pas** d'autre.
- `test_critic_flags_missing_answer_as_invalid` : une réponse manquante rend le verdict invalide.
- `test_llm_critic_calls_rule_tool_rejects_unknown_ids_and_merges_with_rules` : le critique LLM appelle l'outil `run_consistency_rules`, un identifiant inventé est rejeté, le résultat est l'**union règles ∪ LLM**.
- `test_llm_critic_failure_keeps_rules_verdict` : si le LLM déraille, le verdict des règles reste.
- `test_cell_in_text_longest_match` : « top-center area » → « top center » (correspondance la plus longue). C'est cette fonction qui lit la case dans l'option choisie (règle 2b, voir la tuile).
- **Attention :** la critique LLM n'a tourné qu'ici, en simulé.

**E. Agrégateur : le vote pondéré**
- `test_aggregator_weighted_vote_and_tiebreak` : un poids de 2 bat un poids de 1 ; à égalité, la **confiance** décide.
- `test_aggregator_expert_votes_on_discrimination_and_none_if_no_votes` : l'expert vote sur la discrimination ; sans vote, la réponse est `None`.
- *Lien avec la slide 8 :* « son poids doit dépendre de sa fiabilité sur dev » est déjà un paramètre (`weights`, `expert_weight_discrimination`) ; il reste à le **régler**.

**F. Métriques**
- `test_balanced_accuracy_and_fpr` : exactitude équilibrée, taux de fausses alarmes, défauts manqués.
- `test_mcnemar_and_bootstrap` : McNemar exact et IC bootstrap.

**G. Découpage** : `test_split_is_stratified_and_deterministic` (même graine → mêmes images, pas de recouvrement).

**H. Orchestrateur**
- `test_pass_runner_unloads_each_pass_and_resumes_from_cache` : `load, unload, load, unload` (jamais deux modèles ensemble) ; relancé, tout vient du cache.
- `test_full_mock_pipeline_end_to_end` : pipeline complet simulé, `reopen_count ≤ max_reopen`.

**I. Expert, fonctions pures** : `test_expert_anomaly_map_and_grid`. Le chargement de DINOv3 n'est pas testé ici, il est validé par la mesure sur 835 images.

**J. Connexion** : `test_reader_fails_fast_without_llm_server`.

**K. Les deux tests ajoutés par Peter** (rapport §5) :
- `test_parse_subtask_accepts_every_type_in_mmad_json` : les 9 noms de sous-tâches du vrai fichier, dont « Anomaly Detection ». C'est la **réponse au bug de la slide 6**.
- `test_sample_per_source_is_balanced_and_deterministic` : l'option `--per-source K` qui a tiré les 18 images.

### Le compte des tests
- `main` : **20** → le « 20/20 » de la slide 6 (les tests passaient alors que le vrai fichier cassait le chargeur).
- `711294a` : + test du chargeur → **21** → le « 21/21 » de la slide 3.
- `c15bc0b` : + test de `--per-source` → **22 passés** (rapport §5, dans `src/` et dans la copie `Colab/`).
- Ta branche `Test_implementation` en a 27 (endpoints distants, repli, rotation de clés) : pas comparable.
- **Phrase honnête :** « 21 sur 21 au moment du deck, 22 aujourd'hui ; le 21ᵉ répond au bug du chargeur. »

---

## 2. Slide 7, « All five agents ran on real MMAD images » : ce qui a été lancé

### La commande et la configuration exactes (rapport §7.1)
```bash
set -a; source configs/2026-11-12_local_ollama_smoke.env; set +a
PYTHONPATH=src .venv/bin/python -m inspector.main --split dev --per-source 4 --run-name smoke_ollama_qwen4b_ps4
```
- **Machine :** le portable de **Peter**, RTX 3060 6 Go. (À l'oral, ne dis pas « mon portable ».)
- **Lecteurs :** `qwen3-vl:4b-instruct`, Q4, 3,3 Go, servi par Ollama 0.40.7, contexte 8 192 jetons. Le même modèle pour A et B.
- **Vérificateur :** règles seules. **Expert :** DINOv3, seuil recalibré à 0,165.
- **Images :** 4 par source (2 saines, 2 défectueuses), et 2 saines pour la petite étiquette `MVTec-AD` → **18 images, 8 défectueuses, 88 questions**. C'est le « 2–4 per source » du pied de page.

### C'était le troisième essai
| Run | Images | Ce qui s'est passé |
|---|---|---|
| 1 `…_ps2` | 9 prévues | **Arrêté** après 3 images : une photo GoodsAD faisait 12 492 jetons pour 8 192 de contexte, et l'erreur arrêtait la passe |
| 2 `…_ps2_v2` | 9 | Complet après correction : 64,4 %, 2 réouvertures (c'est le « 9 images » du commit `353647e`) |
| **3 `…_ps4`** | **18** | **Le run de la slide** |

### Les sept passes, avec les vrais temps (`summary.json`)
| Passe | Temps | Ce qu'elle a fait |
|---|---|---|
| 1 Expert | **18,0 s** | 18 cartes, 18 zooms, 18 scores (≈ 1 s par image avec le chargement du modèle) |
| 2 Lecteur A | **322,9 s** | 88 réponses en JSON, ~3 476 jetons d'entrée par appel |
| 3 Lecteur B | **322,0 s** | idem, même modèle |
| 4 Vérificateur | 0,4 s | 4 questions marquées, sur 4 images |
| 5 Lecteur A, réouverture | 15,7 s | les 4 questions reposées avec le message du vérificateur |
| 6 Lecteur B, réouverture | 5,7 s | idem |
| 7 Vote | 0,4 s | 88 réponses finales |

Total ≈ 11 min. La slide n'affiche pas les passes 5 et 6.

### Le compte des appels
- 2 lecteurs × 88 questions = **176**, plus 4 questions rouvertes × 2 lecteurs = **8** → **184 appels, 0 invalide**.
- « ~3,7 s » = **3,67 s par question et par lecteur** (322,9 s / 88). Ce n'est pas « par question » au sens du système : il y a deux lecteurs.

### Le tableau des résultats, avec le nombre de questions derrière chaque case
| Sous-tâche | Questions | Exactitude | Lecture honnête |
|---|---|---|---|
| Discrimination (équilibrée) | 18 | **52,5 %** | 3 saines sur 10 acceptées, 6 défauts sur 8 trouvés. Voir l'encadré ci-dessous |
| Classification du défaut | 8 | 62,5 % | 5 sur 8 |
| Description du défaut | 8 | 62,5 % | 5 sur 8 |
| Localisation du défaut | 6 | **50,0 %** | 3 sur 6 : c'est là que l'indice aide (câble) ou entraîne (punaises 080) |
| Analyse du défaut | 8 | 87,5 % | 7 sur 8 |
| Classification de l'objet | 10 | 100 % | 10 sur 10 : l'objet est facile à reconnaître |
| Analyse de l'objet | 30 | 83,3 % | 25 sur 30 |
| **Moyenne des 7** | **88** | **71,2 %** | moyenne simple des 7 cases |

- **Erreurs d'inspection** (absentes de la slide) : **70 % de pièces saines rejetées**, **25 % de défauts manqués**.
- **IC 95 %** : la CLI a affiché [53,8 ; 85,6], arrondi en [54, 86] sur la slide ; le rapport recalcule [52,4 ; 85,2]. Même moyenne : seul l'ordre de tirage du bootstrap change. Dire « environ 52 à 86 % ».

### Encadré : pourquoi la discrimination est à 52,5 % (mon analyse des traces)
| | Drapeau de l'expert : oui (score ≥ 0,5) | Drapeau : non |
|---|---|---|
| **10 pièces saines** | 6 → **toutes rejetées** | 4 → 3 acceptées (pill 011 rejetée) |
| **8 pièces défectueuses** | 6 → toutes trouvées | 2 → **toutes deux ratées** (punaises 080, pcb3) |

- Le oui ou non final = le drapeau de l'expert sur **17 images sur 18**.
- Ce n'est **pas le vote** qui l'impose : ce sont les lecteurs qui suivent l'indice du prompt. Leurs justifications le disent : *« The visual comparison highlights a high difference level (0.81) in the center region, indicating a defect »* (une bouteille de jus **saine**).
- La règle de réouverture renforce l'effet : elle ne rouvre que les « non » contredits par l'expert, jamais les « oui ». Elle ne peut donc pousser que vers « défaut ».
- Les 4 images GoodsAD ont toutes un score de **1,00**, saines comprises : la référence est un autre produit.
- **Ce que ça ne prouve pas :** l'effet net de l'expert. Il n'y a pas encore de run « sans expert ». C'est une **dépendance** observée, pas une perte mesurée.

### Ce que ce run prouve, et ce qu'il ne prouve pas
- **Prouve :** la chaîne complète tourne sur de vraies images, sur 6 Go, un modèle à la fois ; le JSON est toujours valide ; la réouverture se déclenche et s'arrête à une.
- **Ne prouve pas :**
  - une précision : 18 images, IC de plus de 30 points ;
  - l'apport d'un vote : même modèle deux fois, d'accord 87 fois sur 88 (la seule différence : l'étiquette de la bouteille de jus, départagée en faveur de la bonne réponse) ;
  - une comparaison avec le 73,9 % de Qwen3-VL-8B : autre modèle, autre taille d'échantillon, vérificateur sans LLM.

---

## 3. Slide 8, « Four real cases, four behaviours » : les cas dans les traces

Chaque carte : référence · pièce · zoom envoyé aux lecteurs (cadre orange). Citations : justification du lecteur A, telle que journalisée.

**✓ Cable, `missing_cable/009` (MVTec AD) : 5/5.**
- Expert : « center », score **1,00**. L'indice était bon.
- Réponses : « Missing component », « In the central part of the cable », « Two large voids where wires should be », « Missing cable components, critical for functionality ». Confiance 0,95 partout.
- *Montre :* quand l'indice est bon, tout le reste suit.

**↻ Tile, `oil/010` (MVTec AD) : le vérificateur rouvre, sans effet.**
- Expert : « bottom center », 1,00. Les deux lecteurs : « D. Bottom-left ». Vérité : « C. Center-left ».
- Règle 2b (case ≠ région de l'expert). Message réel : *« reader_a: chose 'bottom left' but the expert's strongest region is 'bottom center'. reader_b: … »*
- Après la réouverture : toujours « Bottom-left ». **L'expert se trompait de case lui aussi.**
- Le type est faux aussi (« crack » au lieu de « discoloration ») : 3/5.
- *Montre :* le mécanisme marche, mais rouvrir avec un indice faux ne corrige rien.

**✗ Can, `drink_can good/019_002` (GoodsAD) : fausse alarme.**
- Canette **saine**, expert « bottom center » à **1,00** ; la référence est une canette d'une autre marque.
- Réponse « Yes » (0,95) : *« The close-up shows a clear defect: the can's label is misaligned and distorted… »*
- Les 4 questions sur l'objet sont justes, dont « beer can », lu sur l'inscription chinoise : 4/5.
- *Montre :* la référence unique non alignée fabrique un « défaut », et le lecteur le confirme.

**✗ Pushpins, `logical_anomalies/080` (MVTec LOCO) : le lecteur suit l'indice.**
- Expert : « top right », **0,40, sous le seuil** → pas de drapeau, **donc pas de réouverture**.
- Discrimination : « No » (faux). Localisation : *« The visual expert hint indicates the top right region as the most different, and the zoomed crop confirms a defect there »* ; vérité « Bottom center ».
- Défaut réel : un **séparateur manquant**, défaut logique qu'une distance entre patchs ne voit pas. **0/5.**
- *Montre :* même un indice « medium », sous le seuil, entraîne le lecteur.

**Deux cas utiles absents de la slide** (pour répondre au jury) :
- **Tapis sain, `carpet good/019`** : expert 0,73 (faux drapeau). Les lecteurs disent « No » (juste), la règle 2a rouvre, ils passent à « Yes » (faux) : *« The visual expert flagged a high-difference region in the bottom left, suggesting a defect… »*. **La réouverture a cassé une bonne réponse.**
- **Punaises `071`** : expert 0,55. Les lecteurs disent « No » (faux), réouverture, « Yes » (juste). **La réouverture a corrigé.**
- Le quatrième cas rouvert, **macaronis `bad/058`**, révèle un défaut de règle : l'expert dit « center », qui **n'est pas une des options** (« Bottom left / Top left / Bottom right / Top right ») ; la règle 2b se déclenche donc quelle que soit la réponse.

**Conclusion de la slide (mot pour mot) :** « the expert's hint is powerful both ways, so its weight must depend on how reliable it was on dev. » Les traces le confirment au-delà des 4 cartes : la discrimination suit le drapeau 17 fois sur 18.

Précaution : **quatre cas choisis sur 18 images sont des exemples, pas des statistiques.**

---

## 4. Développements prêts à dire (30 à 60 s chacun)

À piocher selon le temps ou les questions.

**Slide 3, les tests (≈ 30 s).**
« Avant de lancer un modèle réel, on a testé tout le code sans GPU, avec des modèles simulés : les schémas de messages, la règle d'une seule référence, la relance sur réponse invalide, le vérificateur qui ne rouvre qu'une fois, le vote pondéré, les métriques, le découpage fixé, et un seul modèle chargé à la fois avec reprise sur cache. Vingt-deux tests passent aujourd'hui. Ils prouvent que le code est juste, pas que le système est bon. »

**Slide 3, le fonctionnement par passes (≈ 45 s).**
« Le workflow est un enchaînement fixe de sept passes, écrit en Python : l'expert, le lecteur A, le lecteur B, le vérificateur, la réouverture des deux lecteurs, puis le vote. Chaque passe traite toutes les images, puis décharge son modèle : jamais deux modèles en mémoire, ce qui tient sur 6 Go. Après chaque passe, l'état de chaque image est écrit sur disque : si Colab se coupe, on reprend où on s'est arrêté. »

**Slide 7, ce que chaque passe a fait (≈ 60 s).**
« L'expert a comparé chaque pièce à sa seule référence en 18 secondes et en a tiré une case de la grille, un zoom et un score. Chaque lecteur a reçu la pièce, la référence, ce zoom et une phrase d'indice, et a répondu aux 88 questions en un peu plus de cinq minutes, en JSON validé. Le vérificateur a marqué quatre questions : deux "pas de défaut" contredits par l'expert, deux localisations dans une autre case que l'expert. Ces quatre questions ont été reposées aux deux lecteurs. 176 appels plus 8 : 184, aucun invalide. »

**Slide 7, pourquoi la discrimination est à 52,5 % (≈ 45 s).**
« Le point le plus instructif n'est pas sur la slide. Sur ces 18 images, le oui ou non final suit le drapeau de l'expert dix-sept fois sur dix-huit. Les six pièces saines que l'expert a signalées ont toutes été rejetées, et les deux défauts qu'il n'a pas vus ont été ratés. Les lecteurs ne vérifient pas l'indice, ils le suivent. Sur GoodsAD, l'expert est en alerte sur toutes les images : d'où 70 % de pièces saines rejetées. C'est exactement ce que le réglage sur dev doit corriger : le poids de l'expert, et le fait même de montrer l'indice quand il n'est pas fiable. »

**Slide 8, la tuile et le bilan des réouvertures (≈ 40 s).**
« Sur la tuile, les lecteurs placent la tache en bas à gauche, l'expert en bas au centre. Le vérificateur rouvre une fois, la réponse ne change pas, et la vraie case était centre-gauche : l'expert se trompait aussi. Sur l'ensemble du run, les quatre réouvertures ont corrigé une réponse, en ont cassé une sur un tapis sain, et deux sont restées fausses. Le mécanisme fonctionne ; son utilité reste à mesurer. »

**Slide 8, la canette (≈ 30 s).**
« Sur GoodsAD, l'unique référence est souvent un autre produit. Ici, une canette d'une autre marque : l'expert donne le score maximal sur une canette saine, et le lecteur écrit que l'étiquette est "mal alignée". Les questions sur l'objet, elles, sont justes : il lit même l'inscription chinoise. Le problème est la référence, pas la lecture. »

---

## 4 bis. Ce qui a changé depuis la Delivery I (liens avec ce qu'on avait dit)

Détail complet : `delivery_I_vs_II_comparison.fr.md`.

**Slide 3, les choix qui ont évolué (≈ 40 s).**
« Par rapport au plan de la Delivery I, trois choix ont évolué. Les modèles sont servis derrière un serveur compatible OpenAI, vLLM sur Colab et Ollama en local, plutôt qu'appelés directement : le code est le même, seule l'adresse change. Le vérificateur a des règles et une critique LLM en option ; pour ce premier run, nous avons gardé les règles seules. Et l'agrégateur existe avec son vote pondéré, mais les poids sont encore tous à un : les régler sur le dev est une tâche de cette semaine. »

**Slide 7, ce qui diffère du run annoncé (≈ 35 s).**
« En Delivery I, nous annoncions deux lecteurs de familles différentes. Ce run utilise deux fois le même modèle de quatre milliards de paramètres, sur le portable de Peter : c'est l'ablation "lecteurs identiques", pas le système final. Voilà pourquoi ils sont d'accord 87 fois sur 88. Les lecteurs 8B de deux familles tourneront sur Colab. »

**Slide 11, le retour sur le planning (≈ 30 s).**
« Nous avions prévu les baselines et le débit Colab pour aujourd'hui. Nous avons construit le workflow d'abord, et la mesure vient maintenant : c'est le retard à rattraper, en commençant par le test de coût sur cinquante questions. »

⚠ **Pour toi :** ta branche `Test_implementation` ajoute des options Groq / Gemini / Grok. Elles sortent du protocole strict (modèles ouverts ≤ 8B, en local) et ne serviraient qu'à une borne supérieure. Ne les cite pas à côté des résultats.

---

## 5. Ce qui n'est PAS encore testé (à dire avant que le jury le trouve)

- **Aucun lecteur 8B, ni InternVL3, ni vLLM** n'a tourné ; Colab n'a pas été exécuté.
- **La critique LLM** n'a tourné qu'en test simulé.
- **Pas de run « sans expert »** : l'effet de l'indice n'est pas mesuré (`quick_reader_baseline.py` est écrit, jamais lancé).
- **Baselines B1 à B4 non mesurées** ; le 73,9 % reste le chiffre publié.
- **Poids de l'agrégateur non réglés** ; seuil de l'expert réglé **et** mesuré sur les mêmes images.
- **Métriques non comparées** au `summary.py` officiel de MMAD ; une seule graine.
- **Le test (7 531 images) n'a pas été touché** et ne le sera qu'une fois, après le gel du 19 novembre.

---

## 6. Questions probables du jury sur ma partie

**1. « Les tests prouvent quoi, exactement ? »**
Que le code se comporte comme prévu quand le modèle répond ce qu'on a scripté : schémas, une seule réouverture, vote, métriques, découpage, reprise sur cache. Pas que les réponses sont justes : les modèles sont simulés. La preuve : ces tests passaient 20 sur 20 alors que le vrai fichier cassait le chargeur.

**2. « Pourquoi seulement 18 images ? »**
C'est un test de fumée sur un portable de 6 Go avec un modèle 4B quantifié, pour trouver les pannes. Le premier essai s'est arrêté sur une photo trop grande pour le contexte ; on a corrigé, relancé sur 9 puis 18 images. Les mesures sérieuses se feront sur le dev complet, sur Colab.

**3. « Le 71,2 % est-il comparable au 73,9 % ? »**
Non : modèle 4B quantifié, 18 images (IC d'environ 52 à 86 %), vérificateur sans LLM, et le 73,9 % est un chiffre publié, à remesurer sur notre découpage.

**4. « L'expert influence-t-il la réponse finale ? »**
Beaucoup. Sur la discrimination, le oui ou non final suit son drapeau 17 fois sur 18. Ce n'est pas le vote qui l'impose, ce sont les lecteurs qui reprennent l'indice, et la réouverture les pousse dans le même sens. D'où 70 % de pièces saines rejetées. Son effet net n'est pas mesuré tant qu'on n'a pas de run sans expert.

**5. « Le vérificateur sert-il à quelque chose ? »**
Il détecte un conflit et impose un second regard ; il ne connaît pas la réponse. Dans ce run, les quatre réouvertures ont corrigé une réponse, en ont cassé une et laissé deux fausses : bilan nul, avec deux fois le même petit modèle et un indice parfois faux. L'ablation « sans vérificateur » sur le dev dira s'il apporte quelque chose avec deux familles de lecteurs.

**6. « Que voit exactement un lecteur ? »**
Trois images (pièce, référence `random_templates[0]`, zoom de l'expert, chacune réduite à 768 px), un message système d'inspecteur, une phrase d'indice (région, niveau « low / medium / high », « this is a hint, not ground truth ») et la question avec ses options. Environ 3 500 jetons par question. Aucun texte de domaine, aucune seconde référence.

**7. « Pourquoi Ollama et un 4B ? »**
6 Go ne suffisent pas pour un 8B en bf16, et le téléchargement des poids bf16 du 4B depuis Hugging Face avançait à environ 2 Mo/s. Ollama sert le même modèle quantifié en 3,3 Go, derrière la même API compatible OpenAI : le code ne change pas.

**8. « Zéro erreur sur 184 appels, est-ce crédible ? »**
Oui, mais **après** deux corrections : images réduites à 768 px, et requête refusée comptée comme erreur au lieu d'arrêter la passe. Le format est contraint (JSON validé, une relance) et ce mécanisme est testé. Avec plus de questions, on s'attend à quelques erreurs, qu'on comptera fausses.

**9. « Avez-vous touché au test ? »**
Non. Il ne sera lancé qu'une fois, après le gel du 19 novembre ; tout le réglage se fait sur les 835 images dev.

---

## 7. Pense-bête avant l'oral
- [ ] Dire « Qwen 4B via Ollama » sur la slide 3, pas « InternVL3 via vLLM ».
- [ ] Dire « le portable de Peter », pas « mon portable ».
- [ ] Dire « par question et par lecteur » pour les 3,7 s.
- [ ] Dire « 4 questions, sur 4 images » pour les réouvertures.
- [ ] Avoir en tête 17 / 18 et 70 % pour la discrimination.
- [ ] Ne jamais mettre 71,2 % à côté de 73,9 % sans dire que ce n'est pas comparable.
- [ ] Dire « 21 au moment du deck, 22 aujourd'hui » pour les tests.
- [ ] Vérifier avec Peter que la PR n° 3 (`peter/expert-dev` → `main`) est fusionnée : sinon le dépôt public ne montre ni les résultats ni les sorties.
