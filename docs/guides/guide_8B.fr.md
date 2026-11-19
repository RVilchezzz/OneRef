# Guide : passer le système S à l'échelle 8B (Colab, vLLM)

Pour Maxime. Le système S complet tourne déjà à l'échelle 4B sur le laptop de Peter. Ce guide explique comment
lancer **le même système avec des lecteurs 8B** sur Colab, sans casser la comparaison avec la version 4B.

## 1. Où on en est (19 nov.)

- Le code est fini et testé : 50 tests sans GPU passent, dans `src/` et dans la copie `Colab/` (identique).
- Le dev complet est fait à l'échelle 4B : 835 images, 3 937 questions. Résultats dans
  `results/compare/2026-11-19_dev_frozen/summary.md` et `docs/reports/2026-11-18_dev_results_4B.md`.

| Système (dev, 4B) | Moyenne des 7 sous-tâches [IC 95 %] | McNemar contre S |
|---|---|---|
| **S** (Qwen + InternVL + vérificateur Gemma + juge Gemma) | **66,5** [64,8 ; 68,2] | — |
| B1 Qwen3-VL-4B seul | 65,9 [64,3 ; 67,5] | p = 0,28 |
| B3 Qwen 4B × 3 tirages, vote | 65,6 [64,0 ; 67,3] | p = 0,14 |
| S sans juge | 64,6 | p = 0,009 |
| B2 InternVL3.5-4B seul | 60,2 | p < 1e-20 |

- La config 4B est gelée : `configs/2026-11-19_frozen.env`. Le test 4B (2 000 images, `scripts/run_test.sh`)
  n'est **pas encore lancé**.
- Lecture : le **juge** est la pièce qui compte (+2,1 points sur S sans juge). Le vérificateur apporte peu. À 4B,
  S ne bat pas encore B1 de façon significative. La question pour le 8B : est-ce que l'écart grandit avec des
  lecteurs plus forts ?

## 2. À lire avant de commencer (30 min)

1. `README.md` : la carte du dépôt et les commandes.
2. `configs/2026-11-19_frozen.env` : chaque réglage de S et la raison qui vient du dev.
3. `docs/reports/2026-11-18_dev_results_4B.md` : les résultats et les ablations.
4. `src/inspector/orchestrator/graph.py` : l'ordre des passes (`1_expert` … `7_aggregator`).

## 3. Ce qui change entre 4B et 8B

| | 4B (fait) | 8B (à faire) |
|---|---|---|
| Où | Laptop RTX 3060 6 Go | Colab, GPU L4 (bf16). Pas d'A100 |
| Service des modèles | Ollama (`localhost:11434`) | vLLM, un serveur par passe (`ONEREF_VLLM_AUTOSTART=true`, `localhost:8000`) |
| Lecteur A | `qwen3-vl:4b-instruct` | `Qwen/Qwen3-VL-8B-Instruct` |
| Lecteur B | InternVL3.5-4B Q4_K_M | `OpenGVLab/InternVL3_5-8B` (à vérifier sous vLLM) |
| Vérificateur et juge | Gemma 4 E2B | Gemma 4 E4B, ou tout modèle ≤ 8B **d'une troisième famille** (id à vérifier) |
| Expert | DINOv3 ViT-B/16 | **Le même**, donc sa calibration est la même |
| Agrégation | config C2 | **La même** (poids 0,5, éteint sur GoodsAD, seuils par source) |

Le modèle de config est prêt : **`configs/2026-11-19_8B_colab_template.env`**. Il reprend la config gelée 4B
et change uniquement le service et les modèles. Son en-tête montre comment le charger dans une cellule du notebook.

## 4. Étapes

1. **Mettre le dépôt sur Drive** : `MyDrive/OneRef/` avec `Colab/`, `configs/` et `data/splits/`.
2. **Images.** Plutôt que les zips de 28 Go, télécharge seulement ce qu'il faut. Il faut le `mmad.json` complet.
   ```bash
   python scripts/fetch_mmad_subset.py --split dev
   python scripts/fetch_mmad_subset.py --split test --sample 2000 --masks   # plus tard, pour le test
   ```
3. **Les mêmes images que le 4B.** Le split se calcule à partir du `mmad.json` complet et de la graine fixe.
   Copie `data/splits/` du dépôt vers le `ONEREF_SPLITS_DIR` de Drive, et **n'utilise jamais `--available-only`**
   pour un vrai run : avec une partie seulement des images, le split change et les résultats ne sont plus
   comparables. (Le notebook l'ajoute en mode `"sample"` : mets `MMAD_MODE = "none"`.)
4. **Charger la config** : la cellule 5 du notebook (chemins Drive), puis le modèle 8B.
5. **Vérifier Gemma sous vLLM** : une requête `curl` sur `/v1/chat/completions`. La réponse ne doit contenir
   aucun bloc de réflexion. Sinon, remplace `CRITIC_EXTRA_BODY` et `JUDGE_EXTRA_BODY` par
   `'{"chat_template_kwargs": {"enable_thinking": false}}'`.
6. **Smoke test, 5 images** : `python -m inspector.main --split dev --per-source 1 --run-name smoke_8B`. Dans
   `outputs/smoke_8B/telemetry.jsonl`, vérifie :
   - 0 réponse invalide des lecteurs (`"error"` dans `calls`) ;
   - le critique a `llm` et pas de `llm_error` ;
   - le juge (`agent: judge`) a des `answers` non vides.
7. **Sonde de coût, 50 images** (`--limit 50`) : mesure les unités Colab avant de lancer le dev complet.
8. **Dev complet** : `python -m inspector.main --split dev --run-name dev_full_S_8B`. Le run reprend tout seul
   après une coupure, il suffit de relancer la cellule.
9. **Baselines et ablations, sans GPU**, depuis le cache :
   ```bash
   PYTHONPATH=src:scripts python scripts/reaggregate.py outputs/dev_full_S_8B --from-pass 3_reader_b \
     --out outputs/reagg/dev_8B_pre \
     --variant 'B1={"aggregate_readers": ["reader_a"], "expert_weight_discrimination": 0}' \
     --variant 'B2={"aggregate_readers": ["reader_b"], "expert_weight_discrimination": 0}'
   PYTHONPATH=src:scripts python scripts/reaggregate.py outputs/dev_full_S_8B --out outputs/reagg/dev_8B_post \
     --variant 'S={}' --variant 'S_nojudge={"judge_mode": "off"}'
   PYTHONPATH=src python scripts/compare_runs.py S_8B=outputs/reagg/dev_8B_post/S \
     B1_8B=outputs/reagg/dev_8B_pre/B1 B2_8B=outputs/reagg/dev_8B_pre/B2 \
     S_nojudge_8B=outputs/reagg/dev_8B_post/S_nojudge --out results/compare/<date>_dev_8B
   ```
   B3 en 8B : la même commande que `scripts/run_test.sh` étape 3, avec `configs/2026-11-18_dev_B3_k3.env`
   par-dessus la config 8B.
10. **Geler** : copie le modèle en `configs/<date>_8B_frozen.env`, commite, et ne le modifie plus.
11. **Test, une seule fois** : `--split test --sample 2000 --frozen-config`. Ce sont les mêmes 2 000 images que le
    test 4B. Ensuite `compare_runs.py`, `official_summary.py`, `error_analysis.py`, `make_figures.py`, comme dans
    `scripts/run_test.sh`.

## 5. Règles du protocole (ne pas casser)

- Référence = uniquement `random_templates[0]`. Pas de `similar_templates`, pas de `domain_knowledge.json`.
- Tout réglage se décide sur le **dev**. Le test tourne une fois, avec la config gelée.
- Pas de modèle entraîné sur les sources de MMAD (AD-Copilot, EMIT, ADOPD, LAD-Reasoner) : fuite de données.
- Les chiffres du rapport viennent uniquement des fichiers de run, jamais d'une estimation.

## 6. Pièges déjà rencontrés à 4B (le code les gère)

- **InternVL renvoie parfois le schéma JSON** au lieu de la réponse : `ReaderAnswer.unwrap_schema_echo` le
  corrige. Il reste environ 0,4 % de réponses invalides, comptées comme erreurs.
- **Gemma avec la réflexion activée** renvoie une réponse vide : d'où `*_EXTRA_BODY`.
- **Erreur 500 du serveur** (Gemma qui répète des tokens) : capturée. L'image garde le verdict des règles et
  l'erreur est journalisée.
- **Coupure en plein run** : le cache est écrit image par image. Relancer la même commande reprend là où ça
  s'est arrêté.
- **Réouverture** : chaque lecteur ne refait que ses propres questions (`reopen_by_reader`).

## 7. Non vérifié (à tester en premier)

- InternVL3.5-8B et Gemma 4 sous vLLM sur Colab (versions de vLLM et de torch, `--trust-remote-code`).
- L'appel d'outil du critique avec `--tool-call-parser hermes` pour Gemma. Ce n'est pas bloquant : les règles sont
  aussi écrites dans le prompt, et le critique marche même sans appeler l'outil.
- La mémoire d'une L4 pour chaque modèle en bf16, avec `--max-model-len 8192`.
- Les ids exacts des checkpoints Hugging Face dans le modèle de config.

Après chaque modification de `src/inspector/`, il faut resynchroniser `Colab/` (commande dans `Colab/README.md`)
et relancer les tests dans les deux copies.
