# Rapport des tests Colab, pipeline OneRef 4B (T4)

Date : 2026-11-24. Branche du dépôt : `Test_implementation`. Les tests ont été faits sur le split **dev** uniquement. Le split test n'a pas été lancé.

## 1. Environnement

| Élément | Valeur |
|---|---|
| GPU | Tesla T4, 15 360 MiB (pas de bf16, pas de FP8 natif) |
| vLLM | 0.34.0, Python 3.13 |
| Arguments vLLM | `--max-model-len 8192 --gpu-memory-utilization 0.85 --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes --dtype half --enforce-eager` |
| Données | 50 images dev (`--sample 50`), sous-ensemble MMAD sur Drive |
| Expert visuel | **DINOv2** (repli). DINOv3 est inaccessible (voir §4) |
| Readers | Qwen3-VL-4B-Instruct (A), InternVL3_5-4B (B) |
| Critic | Qwen3-VL-4B-Instruct (surcharge, voir §3) |
| Judge | voir §2 |
| Aggregator | vote pondéré (règle, sans LLM) |

## 2. Résultats (50 images dev, accuracy par sous-tâche)

| Sous-tâche | Judge Qwen3-VL-4B (`probe`) | Judge Ministral-3-3B, options Mistral explicites (`probe_ministral`) | Judge Ministral-3-3B, arguments par défaut (`probe_full_hf`, 2026-11-24) |
|---|---|---|---|
| anomaly_discrimination | 60,4 | 60,4 | 60,4 |
| defect_classification | 25,8 | 35,5 | 35,5 |
| defect_localization | 33,3 | 33,3 | 33,3 |
| defect_description | 31,2 | 34,4 | 34,4 |
| defect_analysis | 77,4 | 77,4 | 77,4 |
| object_classification | 77,8 | 83,3 | 83,3 |
| object_analysis | 69,4 | 77,6 | 77,6 |
| **MEAN (7 sous-tâches)** | **53,6 %** IC95 [48,1 ; 59,5] | **57,4 %** IC95 [51,1 ; 64,1] | **57,4 %** IC95 [51,1 ; 64,1] |

- Questions arbitrées par le judge : 44. Qwen : 1 échec de format. Ministral : 0 échec.
- L'écart Qwen / Ministral (+3,8 points) **n'est pas significatif** : les intervalles se chevauchent, il n'y a que 50 images, et l'expert est DINOv2.
- Ces chiffres **ne sont pas comparables** aux 66,5 % de la référence locale (DINOv3, 835 images, config gelée).
- Les deux runs Ministral donnent exactement les mêmes nombres. Ce n'est pas une mesure indépendante : ils reprennent le même cache des passes 1 à 6.
- Dans `probe_full_hf`, les passes 1 à 6 (expert, readers, critic, reopens) ont été **reprises du cache Drive** de `probe`, copié sans les dossiers `6b_judge` et `7_aggregator`. Seuls le judge et l'aggregator ont été rejoués. Ce n'est pas un run entièrement frais.

## 3. Ce qui fonctionne

- Qwen3-VL-4B et InternVL3.5-4B (readers) sous vLLM sur T4, en fp16 avec `--enforce-eager`.
- Critic et judge avec Qwen3-VL-4B, avec `ONEREF_CRITIC_EXTRA_BODY={}` et `ONEREF_JUDGE_EXTRA_BODY={}`. La config 4B du dépôt (`configs/2026-11-19_4B_colab_vllm.env`) pointe sinon vers `google/gemma-4-E2B-it`, qui ne tourne pas sur T4.
- **Ministral-3-3B-Instruct-2512-BF16** comme judge, 44 questions arbitrées sans échec.
  - vLLM 0.34 détecte seul le format Mistral du dépôt (fichiers `consolidated.safetensors`, `params.json`, `tekken.json`, architecture `PixtralForConditionalGeneration`). Il charge en fp16 avec les arguments vLLM par défaut. Les options `--tokenizer-mode mistral --config-format mistral --load-format mistral` ne sont pas nécessaires.
  - Le démarrage du serveur prend environ 5 min.
- Cache par image sur Drive : une passe sans travail à faire ne charge pas de modèle, et les runs reprennent au dernier cache.

## 4. Ce qui ne fonctionne pas, ou avec réserve

| Sujet | Constat |
|---|---|
| Gemma 4 E2B sur T4 | Ne tourne pas : kernel Triton qui demande plus de 96 Ko de shared memory. |
| Ministral version FP8 (par défaut) | Inutilisable sur T4 (pas de FP8 natif). Il faut le dépôt BF16, chargé en fp16. |
| `google/gemma-3-4b-it` | HTTP 403 avec le token actuel. |
| DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) | HTTP 403 : le token a `canReadGatedRepos: false`. Il faut l'accès au dépôt (licence acceptée) et la permission sur le token. L'expert reste donc en DINOv2. |
| `microsoft/Phi-4-multimodal-instruct` | HTTP 200 (accès OK), mais **non testé** comme judge. |
| DeepSeek, Kimi | Écartés sur recherche web : pas de VLM d'environ 4B (Kimi-VL = MoE 16B, trop gros pour un T4). Rien n'a été testé. |
| Log « Detected Mistral tokenizer when using a Hermes model » | Répété à chaque requête avec Ministral. Il concerne le parseur d'appels d'outils, que le judge n'utilise pas (sortie JSON texte). Aucun échec observé. |
| `EngineDeadError`, `resource_tracker ... leaked semaphore` | Bruit d'arrêt du serveur vLLM, sans effet sur les résultats. |
| Avertissements `pip install` | Conflits du résolveur avec des paquets préinstallés de Colab (`requests`, `pillow`, `huggingface-hub`, `numba`, `opentelemetry`, ...). L'installation a abouti et les cellules suivantes ont tourné. |

## 5. Incidents d'exécution

- Montage de Drive : « ValueError: mount failed » deux fois, parce que la fenêtre d'autorisation n'a pas été validée à temps. Il a fallu relancer la cellule après autorisation.
- Runtime Colab déconnecté pour inactivité pendant une session précédente (nouvelle VM, `/content` perdu, sorties Drive conservées).
- La première sonde s'était arrêtée en passe 6 (18/27 questions) avant d'être reprise.
- Le notebook ouvert depuis GitHub affiche l'avertissement « notebook non créé par Google », à valider avant exécution.

## 6. Non testé

- Run entièrement frais avec Ministral sur toutes les passes (les passes 1 à 6 sont reprises du cache).
- **Aggregator LLM** : écrit en local (`src/inspector/agents/llm_aggregator.py`, 52 tests GPU-free passent avec des modèles simulés). Il n'a **jamais tourné sur Colab**, car le notebook clone GitHub et le code n'y est pas poussé.
- Cellule « Run dev complet » (835 images) et split test.
- Judge de repli (Phi-4-multimodal ou autre).
- DINOv3 comme expert.

## 7. Limites

- 50 images : les intervalles de confiance sont larges (environ ±6 points).
- Les valeurs de ce rapport viennent de la sortie des cellules Colab et des fichiers de config lus dans le dépôt, pas d'un export des `telemetry.jsonl`.
