# Erreurs, blocages et corrections à faire (OneRef 4B sur Colab T4)

Date : 2026-11-24. Statut : batterie de modèles terminée (§5) ; run frais 50 images et Phi-4-multimodal comme judge non lancés (voir §8). Toutes les lignes marquées « vérifié » viennent d'une exécution réelle ce jour-là.

## 1. Accès Hugging Face (vérifié, depuis la machine locale)

Test : lecture de `config.json` de chaque dépôt avec le token du projet. Le token est un token **à grain fin** : `canReadGatedRepos: false`, permissions `repo.content.read`, `inference.serverless.write`, `inference.endpoints.infer.write`.

| Dépôt | Résultat | Message |
|---|---|---|
| `google/gemma-3-4b-it` | **403** | « Access to model ... is restricted and you are not in the authorized list » |
| `google/gemma-3n-E2B-it` | **403** | idem |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | **403** | idem |
| `meta-llama/Llama-3.2-3B-Instruct` | **403** | idem |
| `google/gemma-4-E2B-it`, `microsoft/Phi-4-multimodal-instruct`, `Phi-4-mini-instruct`, `Phi-3.5-vision-instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`, `openbmb/MiniCPM-V-4`, `HuggingFaceTB/SmolLM3-3B`, `SmolVLM2-2.2B-Instruct`, `mistralai/Ministral-3-3B-Instruct-2512` et `-Reasoning-2512`, `ibm-granite/granite-3.3-2b-instruct`, `LiquidAI/LFM2-VL-3B` | **200/206 OK** | lisibles |

**Cause :** les 4 dépôts en 403 sont « gated » : la licence doit être acceptée avec le compte HF, et le token doit avoir le droit de lire les dépôts à accès restreint.
**À faire (utilisateur, impossible pour Claude) :**
1. Ouvrir chaque page du modèle sur huggingface.co avec le compte du token et accepter la licence (DINOv3, Gemma 3, Gemma 3n, Llama 3.2). Attendre « Accepted » sur https://huggingface.co/settings/gated-repos.
2. Dans les réglages du token, cocher « Read access to contents of all public gated repos you can access », ou créer un nouveau token Read et mettre à jour le secret Colab `HF_TOKEN` et `.env`.
3. Prévenir Claude : retest immédiat (attendu 200), puis `probe_dinov3`.
**Statut :** bloquant pour DINOv3 (l'expert reste en DINOv2) ; contournable pour Gemma 3 (autres 3B/4B accessibles).

## 2. API hébergée Hugging Face (vérifié)

Le routeur `https://router.huggingface.co/v1` liste `google/gemma-3-4b-it` et `meta-llama/Llama-3.2-3B-Instruct`. Appel de test (texte JSON et image synthétique, aucune donnée du projet) :

- Résultat : **HTTP 402** : « Purchase pre-paid credits to continue using Inference Providers. Alternatively, subscribe to PRO ... »
- **Cause :** le compte n'a plus de crédits d'inférence.
- **À faire (utilisateur) :** acheter des crédits ou prendre PRO. Claude ne paie pas.
- Même avec crédits : ce chemin sort du protocole local strict, donc à rapporter comme borne supérieure seulement.

## 3. Passerelle LLM locale (vérifié)

`ONEREF_LLM_BASE_URL` et `ONEREF_GROK_BASE_URL` valent `http://127.0.0.1:20128/v1` (modèles Gemini et Groq : `gemini/gemini-3.8-flash`, `groq/qwen/qwen3.8-27b`, `groq/openai/gpt-oss-20b`).

- Résultat : **connexion refusée** (WinError 10061). La passerelle n'est pas démarrée.
- Conséquence : `tests/test_llm_connection.py` échoue (« no model reachable »). Ce test appelle un vrai endpoint, il est indépendant de l'aggregator.
- **À faire (utilisateur) :** démarrer le service qui écoute sur le port 20128, puis relancer. Colab ne peut pas atteindre `127.0.0.1` de ta machine.
- Remarque : ces modèles ne sont pas des 3B/4B (27B, 20B, flash), donc hors du protocole 4B.

## 4. Quota Colab / compte Google

Pas de problème de quota à ce stade (T4 disponible). Si le quota saute : changer de compte Google se fait via le sélecteur de comptes déjà connectés ; Claude ne saisit aucun mot de passe et chaque compte demande de nouveau l'autorisation Drive (à valider par l'utilisateur).

## 5. Batterie de modèles sur T4 (en cours)

Script : chargement de chaque modèle seul avec vLLM (`--dtype half --enforce-eager --max-model-len 8192`), 3 appels JSON bruts, 1 appel avec image si modèle visuel, 3 appels via l'agent `LLMAggregator` (pydantic-ai, `PromptedOutput` + validateur de candidat). Résultats détaillés et journaux vLLM sur Drive : `MyDrive/OneRef/outputs/model_battery/` (`results.json`, un `.log` par modèle).

Modèles testés, dans l'ordre : Gemma 4 E2B, Ministral-3-3B (FP8 par défaut), Gemma 3 4B, Qwen2.5-VL-3B, Qwen3-4B-Instruct-2507, Phi-4-mini, SmolLM3-3B, Granite-3.3-2B, SmolVLM2-2.2B, LFM2-VL-3B, MiniCPM-V-4, Phi-3.5-vision, Phi-4-multimodal, Ministral-3-3B-Reasoning.

### Résultats du 1er passage (vérifié, sortie de la cellule Colab, T4)

| Modèle | Démarrage vLLM | JSON brut (3 appels) | Appel avec image | Constat |
|---|---|---|---|---|
| `google/gemma-4-E2B-it` | **échec** (685 s) | - | - | `triton.runtime.errors.OutOfResources: out of resource: shared memory` puis `RuntimeError: Engine core initialization failed`. Le T4 (sm75) n'a pas assez de shared memory pour ce kernel Triton. **Confirmé.** |
| `mistralai/Ministral-3-3B-Instruct-2512` (défaut) | OK (135 s) | 3/3 | valide, répond B (correct) | **Charge sur T4.** L'hypothèse « FP8 inutilisable » du rapport précédent est donc **fausse** pour vLLM 0.34 : il démarre. À corriger dans le rapport. |
| `google/gemma-3-4b-it` | **échec** (20 s) | - | - | HTTP 403 « not in the authorized list » (accès gated, voir §1). |
| `Qwen/Qwen2.5-VL-3B-Instruct` | OK (275 s) | 3/3 | valide, B correct | utilisable |
| `Qwen/Qwen3-4B-Instruct-2507` (texte) | OK (215 s) | 3/3 | n/a | utilisable |
| `microsoft/Phi-4-mini-instruct` (texte) | OK (245 s) | 3/3 | n/a | utilisable |
| `HuggingFaceTB/SmolLM3-3B` (texte) | OK (175 s) | **0/3** | n/a | émet `<think>...` avant le JSON et dépasse `max_tokens` : inutilisable tel quel (mode raisonnement à couper). |
| `ibm-granite/granite-3.3-2b-instruct` (texte) | OK (185 s) | 3/3 | n/a | utilisable |
| `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | **échec** (40 s) | - | - | `ImportError: Package num2words is required to run SmolVLM processor`. Correction : `pip install num2words`, puis retest (non refait). |
| `LiquidAI/LFM2-VL-3B` | OK (190 s) | 3/3 | JSON mais répond « B or C » (pas de choix) | vision faible sur le test |
| `openbmb/MiniCPM-V-4` | OK (260 s) | 3/3 | **sortie corrompue** (`<reserved_934>` répété) | incompatible avec vLLM fp16 tel quel pour la vision |
| `microsoft/Phi-3.5-vision-instruct` | OK (220 s) | 3/3 | « B or C » | vision faible |
| `microsoft/Phi-4-multimodal-instruct` | OK (410 s) | 3/3 | valide, B correct | **utilisable comme judge** ; clé `brief_justification` au lieu de `brief_rationale`, `confidence` en chaîne (le schéma du projet peut les refuser : à surveiller) |
| `mistralai/Ministral-3-3B-Reasoning-2512` | OK (195 s) | 3/3 | « B or C » | raisonnement : pas adapté |

Le GPU est bien libéré après chaque modèle (`gpu_freed: True` partout).

### Test « agent pydantic-ai » : bug de MON script, pas des modèles
Les 11 modèles lancés affichent `agent 0/3`. Cause exacte : `RuntimeError: Cannot run the event loop while another loop is running`. Mon script appelait `loop.run_until_complete` depuis le noyau Jupyter, qui a déjà une boucle. Ce 0/3 **n'est pas une information sur les modèles**. Correction : exécuter le script dans un thread. **Rerun fait (vérifié)** sur 5 modèles, avec l'agent pydantic-ai réel (`PromptedOutput` + validateur de candidat) :

| Modèle | Démarrage | JSON brut | Agent pydantic-ai |
|---|---|---|---|
| Ministral-3-3B-Instruct-2512 (défaut) | 85 s | 3/3 | **3/3** |
| Qwen2.5-VL-3B | 125 s | 3/3 | **3/3** |
| Qwen3-4B-Instruct-2507 | 90 s | 3/3 | **3/3** |
| Phi-4-mini-instruct | 90 s | 3/3 | **3/3** |
| Phi-4-multimodal-instruct | 135 s | 3/3 | **3/3** |

Les 6 autres (SmolLM3, Granite, LFM2-VL, MiniCPM-V-4, Phi-3.5-vision, Ministral-Reasoning) n'ont pas été relancés : leur 0/3 « agent » reste un artefact du bug de boucle, et leurs autres défauts (JSON, vision) sont dans le tableau ci-dessus.

### Aggregator LLM sur Colab, vraie pipeline (vérifié, run `probe_llmagg`)
Conditions : 50 images dev, passes 1 à 6b reprises du cache `probe_full_hf`, seule la passe 7 rejouée avec Ministral-3-3B-Instruct-2512-BF16.
- L'aggregator **tourne** : 33 images concernées, 54 appels LLM, **5 appels en erreur** (retombent sur le vote), 157 questions au total, **12 réponses différentes** du vote.
- **MEAN 35,8 %** (IC95 [26,9 ; 44,6]) contre **57,4 %** pour le vote sur les mêmes caches (recalculé dans la même session : 57,4 %). Découpe : discrimination 34,0 ; classification défaut 19,4 ; localisation 20,0 ; description 25,0 ; analyse défaut 51,6 ; classification objet 55,6 ; analyse objet 44,9.
- **Non expliqué** : 12 réponses changées sur 157 devraient déplacer la moyenne de quelques points, pas de 21,6. À investiguer avant toute conclusion : comparer sous-tâche par sous-tâche les réponses des deux runs, et vérifier les 12 réponses `None` du run de référence. Je ne peux donc pas dire si l'aggregator LLM est mauvais ou s'il y a un défaut de mesure ; **ne pas citer 35,8 % comme résultat**.
- **Indice de diagnostic (cellule de diff, sortie tronquée à l'écran)** : 157 questions, 12 différentes, 0 absente du run de référence, **0 `None` dans `probe_llmagg`, 12 `None` dans le run de référence**. Les 12 réponses « différentes » sont donc très probablement ces 12 questions sans vote valide (aucun lecteur n'a répondu) que le LLM a remplies avec l'option du judge ou un candidat. Conséquence : le run LLM ne peut pas être **pire** que le vote (il ne change que des `None`, déjà comptés comme faux). Un écart de −21,6 points est donc incompatible avec ces chiffres : l'écart vient d'un défaut de mesure (jeux de questions ou d'images différents entre les deux scores, ou score de référence recalculé sur autre chose), pas de l'aggregator. Non confirmé : le runtime Colab s'est déconnecté avant la comparaison sous-tâche par sous-tâche.
- Sur les appels, le modèle choisit la première lettre des candidats dans 29 cas sur 49 (pas de biais net avec 2 candidats).

### Incident de session (vérifié)
Ma cellule de lecture, lancée avec Ctrl+Entrée, a été suivie de l'exécution des cellules du dessous du notebook : la sonde 50 images (cellule 12, 24 s) et le début du « Run dev complet » (cellule 13, 9 s) ont démarré. Je les ai interrompues (`KeyboardInterrupt` pendant la lecture du cache, GPU à 0 Go, aucun modèle chargé). Aucune donnée n'a été perdue. Lancer ces cellules uniquement avec le bouton ▶ de la cellule.

## 6. Déjà constaté avant cette session (voir aussi `colab_4B_probe_report.md`)

- Gemma 4 E2B sur T4 : kernel Triton qui demande plus de 96 Ko de shared memory (la batterie §5 le reteste).
- Ministral FP8 par défaut : **hypothèse fausse**. `mistralai/Ministral-3-3B-Instruct-2512` démarre sur T4 (85 à 135 s) et répond correctement (JSON 3/3, image, agent 3/3). Le dépôt `...-BF16` fonctionne aussi (fp16). Les deux sont utilisables.
- Montage Drive : `ValueError: mount failed` quand l'autorisation n'est pas validée à temps. Correction : relancer la cellule après autorisation.
- Runtime Colab déconnecté pour inactivité : relancer les cellules 1 à 5 ; les sorties Drive sont conservées.

## 7. Ce qui est modifié côté code

- Aggregator LLM ajouté (commit `57f8757`, branche `Test_implementation`) : `Settings.aggregator_model` ; sans lui, comportement inchangé.
- **À modifier pour la config Colab T4** : `configs/2026-11-19_4B_colab_vllm.env` pointe critic et judge vers `google/gemma-4-E2B-it` (ne tourne pas sur T4) et `ONEREF_*_EXTRA_BODY` pour Gemma. Proposition : critic Qwen3-VL-4B-Instruct, judge et aggregator `mistralai/Ministral-3-3B-Instruct-2512-BF16`, extra body vide. **Contourné** (2026-11-25) : la cellule 5 du notebook applique ce remplacement sur T4 ; le fichier de config reste inchangé. Marche à suivre : `Colab/README.md`.

## 8. Non lancé volontairement

- Run frais 50 images (toutes passes, sans cache) : non lancé, environ 1 h à 1 h 30 de T4 ; à confirmer avec l'utilisateur.
- Phi-4-multimodal comme judge dans la pipeline : seulement testé en appel brut, image et agent (3/3) ; pas de run `probe_phi4`.
- DINOv3 : toujours 403 (§1).
- Run dev complet (835 images) : plusieurs heures de T4, quota Colab.
- Split test : une seule fois, config gelée. Pas pour déboguer.

## 9. Réponses rejetées dans la sonde 50 images (2026-11-25, run `probe`, `telemetry.jsonl`)

MEAN 57,1 % (241 questions, 13 sans réponse) : ne mesure pas correctement la pipeline à cause de ces rejets.

| Problème (1er passage) | Cause | Correction | Statut |
|---|---|---|---|
| Lecteur A : 114/241 appels en échec | surtout les clés `answer` / `rationale` au lieu de `selected_option` / `brief_rationale` | alias acceptés dans `ReaderAnswer`, contenu inchangé | rejeu local des réponses enregistrées : 104/114 acceptées (relecture : 71/87) |
| Lecteur B : 45/241 en échec | entrée d'environ 10 300 tokens, limite 8 192 | `--max-model-len 16384` | à vérifier sur T4 (mémoire) |
| Critic LLM : 50/50 images en échec | Qwen répond `{"error": ...}` ; cause supposée : outil proposé avec la sortie JSON | `ONEREF_CRITIC_USE_TOOL=false` dans la config Colab | **hypothèse**, à vérifier sur T4 |
| Relecture : 5 réponses valides effacées | un échec à la relecture écrasait la première réponse | la première réponse est conservée | testé en local |

Les résultats dev locaux du 18 nov. (Ollama) ont été produits avant le changement de relecture.
