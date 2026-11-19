# Delivery I → Delivery II : ce qui était annoncé, ce qui existe, les difficultés

Pour Charles, à lire avant l'oral et pour répondre à « qu'est-ce qui a changé depuis la première présentation ? ».

**Sources :**
- Delivery I : `slides_content.md` (les 14 slides et leurs notes orales) et `docs/full_plan.md` (sections 4, 6 et 7 : workflow, validation, risques, planning).
- Delivery II : le PDF, la branche `peter/expert-dev` (code et résultats de l'expert) et le récapitulatif de Peter.

Les faits sont tirés de ces fichiers. Ce qui est ma déduction est marqué **« inférence »**. Les sorties du run de 18 images sont maintenant sur la branche de Peter (`results/runs/`, `results/pipeline_smoke/`, rapport `docs/reports/2026-11-11_first_end_to_end_run.md`).

---

## 1. Le bilan en trois phrases

1. **Restés fidèles à la Delivery I :** le protocole strict (une référence, pas de texte de domaine, réglage sur le dev seulement), l'architecture à cinq agents, la réouverture limitée à une, l'exécution par passes sur 6 Go.
2. **L'ordre des travaux a été inversé :** on avait annoncé « harnais, découpage, baselines B1/B2, expert, débit Colab » pour le 12 novembre et « lecteurs, vérificateur, agrégateur » pour la semaine suivante. En réalité, tout le workflow est codé et a tourné de bout en bout, mais **B1/B2 et le débit Colab ne sont pas faits** (3 éléments sur 6).
3. **Plusieurs choix d'implémentation diffèrent de ce qu'on avait écrit** (section 2), et **plusieurs difficultés n'avaient pas été prévues** (section 3).

---

## 2. Annoncé en Delivery I contre ce qui existe aujourd'hui

| Sujet | Annoncé (Delivery I) | Ce qui existe (Delivery II) | Écart |
|---|---|---|---|
| **Planning du 12 nov.** | Harnais + découpage, **B1 et B2 sur dev**, expert DINOv3, **débit Colab**. Lecteurs, vérificateur, agrégateur pour le 12-19 nov. | Données, découpage, expert sur 835 images : fait. Harnais : partiel (comparaison avec le `summary.py` officiel à faire). **B1/B2 et débit Colab : non faits.** Lecteurs, vérificateur, agrégateur, orchestrateur : déjà codés et lancés sur 18 images. | **Ordre inversé** : on a construit avant de mesurer |
| **Expert visuel** | DINOv3 ViT-S ou ViT-B, carte calibrée, **filtre de régions connexes**, petite tête de comparaison en option. Slide 4 : carte DINOv2 illustrative dont le pic tombait dans la mauvaise case. | DINOv3 **ViT-B/16**, entrée 448 px, **distance de pic brute**, **un seul seuil global** recalibré sur dev (0,49 → 0,61, in-sample). Ni filtre de régions, ni tête de comparaison. | Plus simple que prévu |
| **Choix DINOv2 / DINOv3** | DINOv3 choisi d'emblée (DINOv2 en repli) | Mesuré sur 835 images : AUROC **0,59 (v3) < 0,61 (v2)**, mais meilleure localisation (pic dans le masque 40 % contre 33 %) | Résultat **contre-intuitif**, à expliquer |
| **Lecteurs** | `transformers` (PyTorch) pour Qwen3-VL-8B et InternVL3-8B ; un 4B en 4 bits en local pour déboguer ; Charles sur Colab | Passage par **pydantic-ai + un serveur compatible OpenAI** (vLLM sur Colab, Ollama en local). Run fait sur le **portable de Peter**, avec le **même Qwen3-VL-4B Q4 deux fois**. Jamais lancé sur un vrai Colab. | Autre pile, autre machine, **pas deux familles** dans le run |
| **Vérificateur** | Critique LLM (Qwen3-VL-8B) : schéma, cohérence entre questions, accord avec l'expert ; rouvre une fois | **Règles + critique LLM optionnelle** (outil `run_consistency_rules`). Le run a utilisé les **règles seules**. La règle « pas de défaut alors qu'une question de localisation existe » est codée mais **désactivée** : en MMAD, la présence de ces questions révèle l'étiquette. | Plus prudent, LLM non testé sur données réelles |
| **Agrégateur** | Poids par lecteur **et** par sous-tâche selon la précision sur dev **et l'accord avec l'expert** | Vote pondéré par lecteur/sous-tâche, l'expert vote sur la discrimination, égalité → confiance. **Tous les poids à 1,0, non réglés ; pas de pondération par l'accord avec l'expert.** | À régler cette semaine |
| **Orchestrateur** | « Plan : quels agents tournent pour chaque sous-tâche » ; routage par sous-tâche en ablation | Séquence fixe de 7 passes, tous les agents sur toutes les questions | Routage non implémenté |
| **Données** | Téléchargement complet de MMAD ; dev ~840, test ~2 000 images | Dev **835** (3 937 questions), pool de test **7 531** ; sous-ensemble de **2,1 Go** lu par requêtes HTTP *range* au lieu de 28 Go | Conforme, avec un outil en plus |
| **Évaluation** | Un harnais qui reproduit le score officiel MMAD | Métriques, McNemar et bootstrap codés et testés ; **pas encore comparés au script officiel** | Partiel |
| **Tests** | Non évoqués | 20 tests sur `main`, 21 sur la branche de Peter après la correction du chargeur, **22 passés** aujourd'hui (rapport §5) | Ajout |
| **Traçabilité** | « Every message, time and token count saved as JSONL » (slide 4) | Tenu : télémétrie JSONL, état de chaque image après chaque passe, tableau par question ; copie partageable dans `results/runs/` | Conforme |
| **Hors plan** | — | `--per-source`, réduction des images à 768 px, requête refusée = erreur, télémétrie JSONL, copie Colab du paquet | Ajouts |
| **Sorties distantes** | Modèles ouverts ≤ 8B, en local | La branche de Charles ajoute des options Groq / Gemini / Grok : elles sortent du protocole strict et ne serviraient qu'à une **borne supérieure** | À ne pas mélanger aux résultats |

---

## 3. Difficultés : prévues et rencontrées

### 3.1 Prévues en Delivery I, et confirmées par les mesures
| Risque annoncé (`full_plan.md`, section 7) | Ce qu'on a mesuré |
|---|---|
| Avec une référence unique, l'expert confond changement de produit et défaut (GoodsAD non aligné) | GoodsAD : AUROC **0,51**, pic dans le masque **10 %** ; cas de la **canette** (fausse alarme : autre marque en référence) |
| Défauts logiques de MVTec LOCO invisibles à un expert par patchs | LOCO : AUROC **0,62**, pic dans le masque 43 % ; cas des **punaises** |
| L'expert peut tromper les lecteurs | Cas des punaises : le lecteur recopie « l'indice indique en haut à droite ». Plus large que prévu : la discrimination suit le drapeau de l'expert sur 17 images sur 18 (section 3.2) |
| Volume d'appels de la semaine de test | **Toujours ouvert** : le débit Colab n'est pas mesuré |
| Le MLLM ne respecte pas le format JSON | Géré : une relance, puis comptée comme erreur ; **0 erreur sur 184 appels** après les corrections de la section 3.2 |

### 3.2 Non prévues
| Difficulté | Preuve | Réponse |
|---|---|---|
| **Le chargeur plante sur le vrai fichier** : MMAD appelle la discrimination « Anomaly Detection » | Les tests sur données factices passaient (20/20), le vrai fichier cassait dès la première ligne | Corrigé, 21ᵉ test sur les 9 noms réels. Leçon de la slide 6 : « run on real data early » |
| **Seuil de l'expert hors échelle** | Distances DINOv3 de 0,11 à 0,31 pour une échelle de 0,2 à 0,8 : le drapeau ne se déclenche jamais (exactitude équilibrée 0,49) | Recalibré sur dev à 0,165 → 0,61 (**in-sample, optimiste**) |
| **Contexte dépassé** par les grandes photos de GoodsAD (8 000 jetons) | Découvert en lançant les lecteurs | Images réduites à 768 px ; une requête refusée compte comme erreur au lieu d'arrêter la passe |
| **Les lecteurs 8B et Colab non lancés** | Slide 6 : « critical path for this week » | Test de coût sur 50 questions, puis dev complet |
| **Run 1 arrêté** | Une photo GoodsAD en pleine résolution : 12 492 jetons pour 8 192 de contexte ; l'exception arrête toute la passe après 3 images | Les deux corrections ci-dessus, puis runs 2 (9 images) et 3 (18 images) |
| **Téléchargement trop lent** | Poids bf16 de Qwen3-VL-4B (8,9 Go) à ~110 Ko/s, puis ~2 Mo/s | Remplacé par le modèle quantifié d'Ollama (3,3 Go) : c'est pourquoi le run utilise Ollama et non `transformers` |
| **Deux lecteurs identiques** | Accord 87/88 | Inutiles comme « équipe » : preuve directe qu'il faut deux familles |
| **Les lecteurs suivent l'expert** | Inférence à partir des traces : le oui ou non final suit le drapeau de l'expert sur 17 images sur 18 ; 70 % des pièces saines rejetées, 25 % des défauts manqués | Le poids de l'expert, et le fait de montrer ou non l'indice, à régler sur le dev par sous-tâche et par source |
| **Réouvertures sans effet net** | 4 réouvertures : 1 corrigée, 1 cassée (tapis sain), 2 inchangées ; la règle 2a ne pousse que vers « défaut » | À mesurer avec l'ablation « sans vérificateur » |
| **Règle de localisation mal posée** | Macaronis : la région de l'expert (« center ») n'est pas parmi les options, la règle se déclenche quoi que réponde le lecteur | Ne comparer la case que si elle figure dans les options (à décider) |

### 3.3 Les protections annoncées qui ont tenu
Une relance sur JSON invalide, puis erreur ; une réouverture maximum ; exécution par passes sur 6 Go ; résultats par source ; réglage sur le dev seulement ; test non touché.

---

## 4. Ce qu'on a dit en Delivery I et qu'il faut maintenant nuancer

| Phrase de la Delivery I | Nuance honnête pour la Delivery II |
|---|---|
| Slide 4 : « Two readers from different model families » | Vrai pour la conception, **faux pour le run de 18 images** (même modèle 4B deux fois) |
| Slide 4 : « The verifier checks ✓ schema ✓ consistency ✓ agrees with expert » | En règles seules ; la critique LLM n'a tourné qu'en test simulé |
| Slide 4 : « weights fixed on the dev split » | Pas encore : tous à 1,0 |
| Slide 4 : carte DINOv2 « calibrating this is part of Delivery II » | Fait **en partie** : un seul seuil global, in-sample |
| Slide 3 : l'expert DINOv3 est la pièce clé, comme chez SiGMA | Vrai pour la localisation (77 % contre 51 % sur MVTec AD), **mais l'AUROC de DINOv3 est plus bas** que celui de DINOv2 |
| Slide 11 : « Preliminary · Nov 12 : B1, B2 and the 4B model on dev » | **Non tenu** : seul le 4B a tourné, sur 18 images |
| Slide 13 : « Nov 12 · Harness, splits, B1 and B2 on dev, DINOv3 expert, Colab throughput » | 3 éléments sur 6 ; le reste est décalé |
| Slide 6 : « Charles : Colab, readers » | Le run des lecteurs s'est fait sur le portable de Peter ; Colab n'a pas encore été exécuté |

---

## 5. Phrases prêtes à dire

**Sur l'écart de planning (slide 10).**
« En Delivery I, nous avions prévu les baselines et le débit Colab pour aujourd'hui, et le workflow complet pour la semaine suivante. Nous avons fait l'inverse : le workflow existe et a tourné sur de vraies images, mais les baselines B1 et B2 et la mesure Colab ne sont pas faites. Le retard est dans l'exécution, pas dans la construction, et c'est la première tâche de cette semaine. »

**Sur les choix qui diffèrent (slide 3).**
« Quelques choix ont évolué par rapport au plan. Le vérificateur fonctionne en règles seules pour ce run, avec une critique LLM déjà codée. Les modèles sont servis par un serveur compatible OpenAI, vLLM sur Colab et Ollama en local, plutôt qu'appelés directement. Et l'expert est plus simple : un seul seuil global, sans filtre de régions. »

**Sur ce qui s'est confirmé.**
« Les deux risques que nous avions annoncés se sont confirmés par les mesures : GoodsAD, où la référence est souvent un autre produit, et les défauts logiques de LOCO. Ce que nous n'avions pas prévu : un chargeur qui plante sur le vrai fichier, un seuil hors échelle et un contexte dépassé. »

---

## 6. Questions probables du jury

**« Pourquoi les baselines B1 et B2 ne sont-elles pas faites alors qu'elles étaient prévues pour aujourd'hui ? »**
On a d'abord rendu le workflow complet exécutable de bout en bout pour fiabiliser le code sur de vraies données : c'est ce qui a révélé le plantage du chargeur. Les baselines réutilisent les mêmes lecteurs et les mêmes caches : elles viennent juste après le test de coût Colab sur 50 questions.

**« Pourquoi le vérificateur diffère-t-il du plan ? »**
Le plan prévoyait une critique LLM. Elle est codée (avec l'outil de règles) mais n'a pas encore tourné sur des images réelles. On a gardé les règles seules pour le premier run, et on a désactivé la règle fondée sur la présence de questions, car elle révélerait l'étiquette et violerait le protocole strict.

**« Pourquoi pydantic-ai et Ollama plutôt que `transformers` ? »**
Pour les mêmes agents, avec validation du JSON et relances intégrées, derrière un serveur compatible OpenAI : vLLM sur Colab pour le débit, Ollama pour déboguer sur 6 Go. Le changement de modèle ne demande qu'une variable de configuration.

**« DINOv3 n'a pas un meilleur AUROC : pourquoi l'avoir choisi ? »**
Le rôle de l'expert est de localiser. Il met le pic dans le masque 40 % du temps contre 33 % pour DINOv2 (77 % contre 51 % sur MVTec AD). L'AUROC pèse GoodsAD (296 des 835 images), où les deux sont au hasard.

**« Qu'est-ce qui vous a le plus surpris ? »**
Que les tests sur données factices passent tous alors que le vrai fichier casse à la première ligne. D'où la règle : lancer tôt sur de vraies données.
