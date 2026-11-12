# Delivery II · Ma partie (Charles) en détail : ce que le pipeline a réellement testé et exécuté

Pour Charles, à lire avant l'oral et à garder sous la main pour les questions du jury. Ce n'est **pas** un texte à dire en entier : le script de 10 min est dans `speaker_script_delivery_II.fr.md`. Ici, on développe les slides **3, 7 et 8** (et un peu 2) pour savoir répondre à « mais concrètement, qu'est-ce qui a tourné ? ».

Sources : le PDF `OneRef – Preliminary Results (Delivery II).pdf` (chiffres des slides) et `tests/test_workflow.py` (contenu réel des tests). Les citations des slides restent en anglais.

---

## 0. L'idée à retenir en 20 secondes

On a **trois niveaux de preuve**, du plus faible au plus fort. Il faut toujours dire de quel niveau on parle :

| Niveau | Quoi | Modèles | Ce que ça prouve | Ce que ça ne prouve pas |
|---|---|---|---|---|
| **1. Tests logiciels** (slide 3) | `pytest`, sans GPU | **Simulés** (FunctionModel / mock) | Le code est correct : schémas, règles, vote, métriques, reprise | Rien sur la qualité des réponses |
| **2. Run de bout en bout** (slides 7-8) | 18 images dev, 88 questions | **Réels** mais petits : Qwen3-VL-4B quantifié ×2, vérificateur à règles | La chaîne complète tourne sur de vraies images MMAD sans erreur | Aucun résultat statistique (IC 95 % [54, 86]) |
| **3. Mesures de l'expert** (slides 4-5, Peter) | 835 images dev | DINOv3 vs DINOv2 | L'expert localise (77 % vs 51 % dans le masque sur MVTec AD) | Rien sur les lecteurs 8B |

Phrase de passe-partout : « Le niveau 1 prouve que le code est juste, le niveau 2 que la chaîne tient sur de vraies images, mais aucun des deux n'est encore un résultat de précision. Le résultat viendra de Colab avec les lecteurs 8B. »

---

## 1. Slide 3, « The full workflow exists as code » : les 21 / 21 tests

### Ce que dit la slide
- Cinq cartes : Visual expert, Readers A·B, Verifier, Aggregator, Orchestrator.
  - Seul l'expert est marqué **« Ran on all 835 dev »**. Les quatre autres sont marqués **« Ran on real data »** (le run de 18 images).
- **21 / 21** : « GPU-free tests pass (schemas, critic, vote, metrics, resume) ».
- **835 / 7,531** images dev / test, stratifiées, graine fixée et commitée.
- **2.1 GB** au lieu de 28 GB pour récupérer le dev : seuls les fichiers nécessaires, lus directement dans les zips distants.
- Bas de slide : messages JSON validés par Pydantic ; réponse invalide = **une relance**, puis comptée comme erreur.

### Comment est fait un test « sans GPU »
Aucun vrai modèle n'est appelé. Le fichier désactive même les requêtes réelles (`models.ALLOW_MODEL_REQUESTS = False`). Quand un agent a besoin d'un modèle, on lui donne un `FunctionModel` **scripté** : une fonction Python qui répond ce qu'on a décidé (par exemple d'abord une mauvaise réponse, puis une bonne). On vérifie alors le **comportement du code autour du modèle**, pas le modèle.

### Groupe par groupe (noms réels dans `tests/test_workflow.py`)

**A. Schémas Pydantic : les mauvaises données sont rejetées**
- `test_schema_rejects_bad_values` : une option « E » (hors A-D), une confiance de 1,5, une région « nowhere » (hors de la grille 3×3), une réponse « C » absente des options d'une question oui/non : tout lève une `ValidationError`.
- *Pourquoi c'est important :* c'est ce qui garantit que les agents s'échangent un JSON propre. Une réponse mal formée ne peut pas contaminer la suite.

**B. Protocole strict : une seule référence, pas de texte de domaine**
- `test_strict_one_shot_reference` : la référence est toujours `random_templates[0]`.
- `test_load_mmad_ignores_similar_templates` : même si `mmad.json` contient des `similar_templates`, le chargeur les ignore (l'échantillon n'a même pas l'attribut).
- *Pourquoi :* c'est la règle n°1 de notre protocole (1-shot, pas de `similar_templates`). Ce test empêche de la violer par erreur. C'est lui qui nous distingue de SiGMA.

**C. Lecteurs : la relance sur format invalide**
- `test_reader_agent_retries_on_option_not_offered` : si le lecteur répond « C » sur une question oui/non, le validateur lève `ModelRetry` ; la **deuxième** réponse (« B ») est gardée. On vérifie qu'il y a bien 2 appels et que la réponse brute est journalisée.
- `test_reader_agent_gives_none_after_retries` : si le modèle ne renvoie jamais de JSON, la réponse finale est `None` et chaque appel porte un champ `error`.
- *Lien avec la slide :* c'est exactement la règle « relancée une fois, puis comptée comme erreur ». Et c'est pour ça que le run réel affiche « 0 errors in 184 calls » : le mécanisme est testé, pas supposé.

**D. Vérificateur (critique) : règles, réouverture, LLM**
- `test_critic_conflict_reopens_once` : deux lecteurs en désaccord et un lecteur qui contredit l'expert → conflit détecté, les deux questions sont rouvertes, `reopen_count = 1`. **Un deuxième passage ne donne pas une deuxième réouverture** (garde « ≤ 1 »).
- `test_critic_flags_missing_answer_as_invalid` : une réponse manquante rend le verdict invalide.
- `test_llm_critic_calls_rule_tool_rejects_unknown_ids_and_merges_with_rules` : le critique LLM appelle l'outil `run_consistency_rules` (les règles arrivent bien jusqu'au LLM), un identifiant de question inventé (« nope ») est rejeté puis re-demandé, et le résultat final est l'**union règles ∪ LLM**.
- `test_llm_critic_failure_keeps_rules_verdict` : si le LLM répond n'importe quoi, on garde le verdict des règles et on journalise `llm_error`.
- `test_cell_in_text_longest_match` : « top-center area » → « top center », « in the center » → « center » (on prend la correspondance la plus longue).
- *Pourquoi :* le vérificateur est la partie la plus « logique » du système ; ces tests montrent qu'il ne peut ni boucler (une réouverture max) ni planter si le LLM déraille.
- **Attention :** ces tests utilisent un LLM simulé. Dans le run de 18 images, le vérificateur était **à règles seulement** (pas de critique LLM réel).

**E. Agrégateur : le vote pondéré**
- `test_aggregator_weighted_vote_and_tiebreak` : un poids de 2 pour le lecteur A en localisation l'emporte sur un poids de 1 ; à égalité 1-1, c'est la **confiance** la plus haute qui décide.
- `test_aggregator_expert_votes_on_discrimination_and_none_if_no_votes` : l'expert vote sur la discrimination (poids 2 contre un lecteur de poids 1 : l'expert l'emporte) ; sans aucun vote, la réponse finale est `None`.
- *Lien avec la slide 8 :* le « poids selon la fiabilité de l'expert sur le dev » est déjà un paramètre (`weights`, `expert_weight_discrimination`). Il reste à le **régler sur le dev**.

**F. Métriques : ce qu'on utilisera pour juger**
- `test_balanced_accuracy_and_fpr` : 3 normaux bien classés + 1 défaut raté → exactitude équilibrée 0,5, taux de fausses alarmes 0, défauts manqués 1,0. (L'exactitude simple dirait 75 %, trompeuse.)
- `test_mcnemar_and_bootstrap` : McNemar exact (20 contre 0 → p < 1e-4 ; un système contre lui-même → p = 1) et un IC bootstrap borné entre 0 et 1.
- *Pourquoi :* ce sont les outils de la preuve finale (McNemar p < 0,05 + IC bootstrap par image). On sait qu'ils sont justes avant de s'en servir.

**G. Découpage dev / test**
- `test_split_is_stratified_and_deterministic` : même graine → mêmes images ; aucun recouvrement entre dev et test ; les deux classes (normal / défectueux) sont dans le dev.
- *Lien avec la slide :* « 835 / 7,531, stratifiées, graine fixée et commitée ». 835 + 7 531 = 8 366, soit toutes les images de MMAD.

**H. Orchestrateur : un modèle à la fois, reprise sur cache**
- `test_pass_runner_unloads_each_pass_and_resumes_from_cache` : avec deux passes, les événements sont `load, unload, load, unload` (**jamais deux modèles chargés ensemble**) ; relancé, tout vient du cache et **rien n'est rechargé** ; la latence est journalisée dans le JSONL.
- `test_full_mock_pipeline_end_to_end` : 4 images, pipeline complet simulé → 2 réponses finales par image, `reopen_count ≤ max_reopen`, exactitude moyenne entre 0 et 1.
- *Pourquoi :* la contrainte matérielle (6 Go en local, une seule L4 sur Colab) impose l'exécution **par passes**. Le test prouve que le principe tient, et la reprise « resume » est ce qui nous sauve si Colab coupe.

**I. Expert : seulement les fonctions pures**
- `test_expert_anomaly_map_and_grid` : sur un petit exemple, trois patchs identiques à la référence ont une distance ≈ 0 et le patch différent > 0,4 ; la grille donne bien « top left », « bottom right », « center ».
- *Limite à dire :* le chargement de DINOv3 demande les poids, donc **non testé ici**. Il est validé autrement : par la mesure sur les 835 images dev (slide 5).

**J. Robustesse de la connexion**
- `test_reader_fails_fast_without_llm_server` : sans serveur LLM, le lecteur échoue tout de suite avec « No LLM server », au lieu de tourner dans le vide.

### Attention au chiffre « 21 / 21 »
- Aujourd'hui `tests/test_workflow.py` contient **27 fonctions de test**. Les plus récentes (endpoints Grok / routeur HF, repli `FallbackModel`, rotation de clés, `available_under`) ont été ajoutées après le noyau de 21. S'y ajoutent 2 tests dans `test_llm_connection.py` (ping réel, **ignorés** sans clé, donc hors du compte).
- **Avant l'oral :** lancer `uv run pytest` et dire le nombre exact affiché. Si ça affiche 27, la phrase honnête est : « 21 sur 21 au moment du run, 27 aujourd'hui avec les tests des endpoints distants ».
- La slide 6 de Peter dit « 20/20 » : c'était avant le test du chargeur (21ᵉ test, ajouté après le crash de chargement). À expliquer si on te le demande.

---

## 2. Slide 7, « All five agents ran on real MMAD images » : ce qui a été lancé

### Les faits de la slide
- **Où :** le portable, **RTX 3060 6 Go**. C'est un *smoke test*.
- **Quoi :** **18 images dev**, 2 à 4 par source (MVTec AD, VisA, LOCO, GoodsAD), soit **88 questions** MMAD réelles.
- **Modèles :** les deux lecteurs = **Qwen3-VL-4B quantifié** (via Ollama) ; vérificateur **à règles seulement**. Les lecteurs 8B tournent sur Colab (pas encore).
- **Chaîne et temps :** Expert **18 s** → Reader A **323 s** → Reader B **322 s** → Verifier : **4 rouvertes** → Vote.

### Ce qui s'est passé, étape par étape
1. **Expert (18 s).** DINOv3 compare chaque photo à sa référence (une seule), produit la carte d'anomalie, la région 3×3, un recadrage zoomé et un score. 18 images en 18 s, soit environ 1 s par image avec le chargement.
2. **Lecteur A (323 s)**, puis **lecteur B (322 s)** : chacun répond aux 88 questions, **un modèle à la fois** (c'est le fonctionnement par passes testé en H). Ils voient l'image, la référence, le recadrage et l'indice de l'expert.
3. **Vérificateur :** applique les règles de cohérence. Il a **rouvert 4 questions** (conflit entre lecteurs ou avec l'expert). Les lecteurs répondent alors une 2ᵉ fois sur celles-ci, une seule réouverture max.
4. **Vote :** l'agrégateur donne la réponse finale par question.

### Le compte des appels (ça tombe juste)
- 2 lecteurs × 88 questions = **176 appels**, plus 4 questions rouvertes × 2 lecteurs = **8 appels** → **184 appels**. 
- « **0 errors in 184 calls** » : toutes les réponses étaient du JSON valide, aucune relance n'a fini en échec.
- Le « ~3,7 s » : (323 + 322) / 176 ≈ 3,7 s, soit **par appel d'un lecteur**, et non « par question » comme écrit sur la slide (par question, il y a deux lecteurs, donc ~7 s). À dire correctement à l'oral.

### Le tableau des résultats, lu correctement
| Sous-tâche (88 questions) | Exactitude | Lecture honnête |
|---|---|---|
| Anomaly discrimination (équilibrée) | **52,5 %** | Proche du hasard : le point faible. Cohérent avec l'AUROC de l'expert (0,59 pour DINOv3). |
| Defect classification · description | 62,5 % · 62,5 % | Moyen. |
| Defect localization | **50,0 %** | Moitié juste. C'est là que l'indice de l'expert aide *ou* nuit (voir slide 8). |
| Defect analysis | 87,5 % | Bon, mais sur très peu de questions. |
| Object classification · analysis | 100 % · 83,3 % | L'objet est facile à reconnaître. |
| **Mean of 7** | **71,2 %** | (52,5 + 62,5 + 62,5 + 50 + 87,5 + 100 + 83,3) / 7. |

### Ce que ce run prouve, et ce qu'il ne prouve pas
- **Prouve :** la chaîne complète tourne sur de vraies images ; le JSON est toujours valide ; le cache et la reprise fonctionnent ; la réouverture se déclenche et se limite à une ; le tout tient dans 6 Go.
- **Ne prouve pas :**
  - **IC 95 % [54, 86]** : sur 18 images, l'incertitude est énorme. D'où la phrase de la slide : *« a working pipeline, not a result »*.
  - Les lecteurs sont **le même modèle deux fois** : ils sont d'accord 87 fois sur 88. Ça justifie d'utiliser deux **familles** différentes (Qwen + InternVL) sur Colab, mais cela rend l'accord inutile comme mesure.
  - Le **71,2 % n'est pas comparable** au 73,9 % de Qwen3-VL-8B (publié par AD-Copilot) : autre modèle (4B quantifié), 18 images au lieu de milliers, vérificateur sans LLM.

---

## 3. Slide 8, « Four real cases, four behaviours » : une carte = un comportement du pipeline

Chaque carte montre : référence · pièce · recadrage envoyé aux lecteurs (cadre orange). Les 4 cas viennent des 18 images du run.

1. **Cable (MVTec AD) : 5/5 juste.** Composant manquant, « dans la partie centrale ». L'expert a zoomé sur la bonne zone, les deux lecteurs ont répondu correctement aux 5 questions. *Montre : quand l'indice est bon, tout le système en profite.*
2. **Tile (MVTec AD) : le vérificateur a rouvert.** Les lecteurs disent « en bas à gauche », l'expert « en bas au centre » : désaccord → réouverture (la seule autorisée). Résultat : **toujours faux** (la bonne réponse est « centre-gauche »). *Montre : la réouverture se déclenche bien, mais elle ne garantit pas la bonne réponse. C'est un test de mécanisme, pas de précision.*
3. **Can (GoodsAD) : fausse alarme.** Une pièce saine est déclarée défectueuse parce que la **référence est une autre marque de canette**. *Montre : GoodsAD n'est pas aligné ; une référence différente fabrique un « défaut » (c'est la faiblesse annoncée de l'expert sur cette source, AUROC 0,51).*
4. **Pushpins (LOCO) : a suivi l'indice.** Le lecteur reprend « the expert hint indicates the top right » et se trompe d'endroit. *Montre : un lecteur qui fait confiance à l'expert **amplifie** ses erreurs.*

**Conclusion de la slide (à dire mot pour mot) :** « the expert's hint is powerful both ways, so its weight must depend on how reliable it was on dev. » C'est ce que fait déjà l'agrégateur (tests E) : on règle les poids sur le dev, par sous-tâche, et l'ablation « sans expert » mesurera le gain net.

Précaution : **quatre cas choisis dans 18 images sont des exemples, pas des statistiques**. Ne pas dire « l'expert aide dans X % des cas ».

---

## 4. Trois développements prêts à dire (≈ 30 s chacun)

**Sur les tests (slide 3).**
« Avant de lancer quoi que ce soit sur un modèle réel, on a testé tout le code sans GPU, avec des modèles simulés. Ça couvre les schémas de messages, la règle d'une seule référence, la relance sur réponse invalide, le vérificateur qui ne rouvre qu'une fois, le vote pondéré, les métriques McNemar et bootstrap, le découpage dev/test fixé, et le fait qu'un seul modèle est chargé à la fois avec reprise sur cache. Tout passe. Ces tests prouvent que le code est juste, pas que le système est bon. »

**Sur le run (slide 7).**
« On a ensuite lancé la chaîne complète sur 18 images dev réelles, sur mon portable : 88 questions, 184 appels de lecteurs, zéro erreur de format, quatre questions rouvertes par le vérificateur. Les chiffres, 71,2 % en moyenne, sont ceux d'un petit modèle 4B quantifié utilisé deux fois, avec un intervalle de 54 à 86 %. Ce n'est pas un résultat : c'est la preuve que la chaîne fonctionne de bout en bout. »

**Sur les cas (slide 8).**
« Quatre cas résument les comportements : le câble, tout juste ; la tuile, où le vérificateur rouvre mais ne corrige pas ; la canette, fausse alarme parce que la référence est une autre marque ; les punaises, où le lecteur suit l'indice de l'expert et se trompe. La leçon est que l'indice est puissant dans les deux sens, donc son poids doit être réglé sur le dev. »

---

## 5. Ce qui n'est PAS encore testé (à dire avant que le jury le trouve)

- **Aucun lecteur 8B n'a tourné** : Qwen3-VL-8B et InternVL3-8B sont prévus sur Colab (L4 en bf16 ou T4 en 4 bits), pas encore lancés.
- **Le critique LLM n'a tourné qu'en test simulé**, pas sur de vraies images (le run de 18 images était à règles seules).
- **Les baselines B1 à B3 ne sont pas mesurées** ; le 73,9 % reste le chiffre publié, à remesurer sur notre découpage.
- **Les poids de l'agrégateur ne sont pas encore réglés** sur le dev.
- **Le test (7 531 images) n'a pas été touché** et ne le sera qu'une fois, après le gel du 19 novembre.
- **Les 50 questions de mesure de coût Colab** viennent d'abord : elles fixent la taille du test (1 000 à 2 000 images).

---

## 6. Questions probables du jury sur ma partie

**1. « Les 21 tests prouvent quoi, exactement ? »**
Que le code se comporte comme prévu quand le modèle répond ce qu'on a scripté : validation des schémas, une seule réouverture, vote, métriques, découpage, reprise sur cache. Pas que les réponses sont justes : les modèles sont simulés.

**2. « Pourquoi seulement 18 images pour le run ? »**
C'est un test de fumée sur un portable de 6 Go avec un modèle 4B quantifié : il sert à trouver les pannes (on y a trouvé le crash du chargeur, d'où le 21ᵉ test). Les mesures sérieuses se font sur le dev complet avec les lecteurs 8B sur Colab.

**3. « Le 71,2 % est-il comparable au 73,9 % ? »**
Non : modèle différent (4B quantifié), 18 images (IC 54-86), vérificateur sans LLM, et le 73,9 % est un chiffre publié, à remesurer sur notre découpage.

**4. « Les lecteurs sont d'accord 87 fois sur 88 : le vote ne sert à rien ? »**
Ici oui, parce que c'était le même modèle deux fois. C'est justement pourquoi le vrai système utilise deux familles différentes, et pourquoi l'ablation « lecteurs identiques » est prévue.

**5. « Le vérificateur a rouvert la tuile et s'est quand même trompé : à quoi sert-il ? »**
Il vérifie les **conflits**, il ne devine pas. Sa valeur se mesure avec l'ablation « sans vérificateur » sur le dev, pas avec un cas isolé. La réouverture est limitée à une pour ne pas gaspiller le budget.

**6. « Et si l'expert induit les lecteurs en erreur (punaises) ? »**
Le poids de l'expert est réglé par sous-tâche sur le dev, et l'ablation « sans expert » mesure le gain net. Si l'expert nuit, on le verra et on le dira : un résultat négatif bien mesuré est valide.

**7. « Pourquoi 0 erreur sur 184 appels, est-ce crédible ? »**
Parce que le format est contraint : sortie JSON validée par Pydantic, une relance automatique, et ces mécanismes sont eux-mêmes testés. Sur des modèles 8B et plus de questions, on s'attend à quelques erreurs, qu'on comptera comme fausses.

**8. « Avez-vous touché au test ? »**
Non. Le test (7 531 images) ne sera lancé qu'une fois, après le gel de la configuration le 19 novembre ; tout le réglage se fait sur les 835 images dev.

---

## 7. Pense-bête avant l'oral
- [ ] Lancer `uv run pytest` et noter le nombre exact de tests affiché.
- [ ] Dire « par appel de lecteur » et non « par question » pour les 3,7 s.
- [ ] Ne jamais mettre 71,2 % à côté de 73,9 % sans dire que ce n'est pas comparable.
- [ ] Dire « 21/21 » pour le noyau, et être prêt à expliquer le 20/20 de la slide 6.
- [ ] Vérifier avec Peter le passage 3 → 4 (de « le code existe » à « l'expert fonctionne sur 835 images »).
