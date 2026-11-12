# Guide : version de base (ligne de commande, en local)

Pour qui : tu veux lancer le projet depuis un terminal Windows (PowerShell) avec le code de `src/inspector`.
Les LLM (lecteurs A/B + critique) sont appelés par une API (Gemini gratuit, routeur Hugging Face payant) **ou** par un serveur local (Ollama). Seul l'expert DINOv3 tourne dans ton Python.

> Tout se lance depuis le dossier `OneRef` (le `.env` est lu dans le dossier courant).

## 0. Prérequis

- Python ≥ 3.11 (`python --version`).
- **Au moins 5 Go libres sur `C:`** (torch + transformers, plus ~28 Go si tu télécharges MMAD). Si le disque est plein : `uv cache clean` libère le cache de uv (jusqu'à ~6 Go).
- Un compte Hugging Face avec la **licence DINOv3 acceptée** : https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m, puis un token (Settings → Access Tokens, droit « Read »).
- Une clé Gemini gratuite : https://aistudio.google.com/apikey

## 1. Se placer dans le projet

```powershell
cd "C:\Users\Maxime\OneDrive\Bureau\Travail\UTC\UNICAMP\IA025 - Introduction to Deep Learning\Projet\OneRef"
```

## 2. Installer les dépendances

### Option A : avec uv (recommandé)
```powershell
uv sync
```
Ensuite, préfixe les commandes par `uv run` (ex. `uv run pytest`).

### Option B : sans uv (pip)
```powershell
pip install pydantic pydantic-settings "pydantic-ai-slim[openai]" "transformers>=4.56" scipy pillow numpy torch pytest
```
Dans ce cas, utilise `python -m ...`, et ajoute `$env:PYTHONPATH="src"` avant de lancer `inspector.main` (pas nécessaire pour pytest).

## 3. Créer le fichier `.env`

Crée `OneRef\.env` (ignoré par git ; **ne le commite jamais**) :

```ini
# Token Hugging Face : DINOv3 + téléchargement de MMAD
HF_TOKEN=hf_XXXXXXXXXXXXXXXX

# ---- LLM : Gemini (gratuit) ----
ONEREF_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
ONEREF_LLM_API_KEY=XXXXXXXXXXXXXXXX
ONEREF_READER_A_MODEL=gemini-3.8-flash
ONEREF_READER_B_MODEL=gemini-3.5-flash-lite
ONEREF_CRITIC_MODEL=gemini-3.8-flash
ONEREF_FALLBACK_MODEL=gemini-3.5-flash-lite

# Expert visuel : "cuda" si GPU NVIDIA avec torch CUDA, sinon "cpu"
ONEREF_DEVICE=cpu
```

Autres fournisseurs (remplace le bloc LLM) :

| Fournisseur | Variables |
|---|---|
| Routeur Hugging Face (**crédits payants**, sinon erreur 402) | `ONEREF_LLM_BASE_URL=https://router.huggingface.co/v1`, `ONEREF_LLM_API_KEY=hf_...`, modèles `Qwen/Qwen3-VL-30B-A3B-Instruct:novita` et `Qwen/Qwen3.8-27B:novita` |
| Grok (xAI), clé à fournir | `ONEREF_USE_GROK=true`, `ONEREF_GROK_API_KEY=xai-...`, `ONEREF_GROK_READER_A_MODEL`, `ONEREF_GROK_READER_B_MODEL`, `ONEREF_GROK_CRITIC_MODEL` (ids à vérifier sur docs.x.ai). Le bloc est déjà dans `.env`. |
| Ollama local (petit modèle quantifié) | `ONEREF_LLM_BASE_URL=http://localhost:11434/v1`, `ONEREF_LLM_API_KEY=ollama`, `ONEREF_LLM_PROVIDER=vllm`, `ONEREF_VLLM_AUTOSTART=false`, `ONEREF_READER_A_MODEL=<nom du modèle Ollama>` |
| vLLM | voir `guide_colab.md` (vLLM ne tourne pas sous Windows) |

> Les modèles Gemini ou HF routés sortent du protocole strict (≤ 8B, locaux) : à présenter comme **borne haute**.

## 4. Vérifier l'installation (sans réseau)

```powershell
uv run pytest
# ou : python -m pytest
```
Attendu : environ 21 tests passent (agents factices, aucun appel réseau). Les 2 tests « live » sont ignorés sans clé, ou appellent le LLM si la clé est présente.

## 5. Vérifier la connexion au LLM

```powershell
uv run pytest tests/test_llm_connection.py -s
# ou : python -m pytest tests/test_llm_connection.py -s
```
`-s` affiche les réponses. Attendu :
```
[gemini-3.8-flash] 391
[gemini-3.8-flash] Statue of Liberty
2 passed
```
Le test échoue si la réponse est fausse (391 pour 17×23, « liberty » pour la photo). Si le modèle principal échoue, `ONEREF_FALLBACK_MODEL` est essayé.

## 6. Récupérer le jeu de données MMAD

Il faut les **images** (~28 Go de zips) et `mmad.json`. Place tout sous `data\MMAD\` (les chemins d'images de `mmad.json` sont relatifs à `ONEREF_MMAD_ROOT`).

```powershell
# a) Images (Hugging Face) : long ; HF_TOKEN doit être défini
$env:HF_TOKEN="hf_XXXX"
python -c "from huggingface_hub import snapshot_download; snapshot_download('jiang-cc/MMAD', repo_type='dataset', local_dir='data/MMAD')"

# b) Décompresser les zips en place
Get-ChildItem data\MMAD -Recurse -Filter *.zip | ForEach-Object { Expand-Archive $_.FullName -DestinationPath $_.DirectoryName -Force }

# c) mmad.json (GitHub)
git clone --depth 1 https://github.com/jam-cc/MMAD data\MMAD_code
Get-ChildItem data\MMAD_code -Recurse -Filter mmad.json | Select-Object -First 1 | Copy-Item -Destination data\MMAD\mmad.json
```
Vérifie : `Test-Path data\MMAD\mmad.json` doit afficher `True`.

> Non vérifié : les noms exacts des zips et l'arborescence du dépôt GitHub. Si `mmad.json` est introuvable : `Get-ChildItem data\MMAD_code -Recurse -Filter *.json`.

Autre emplacement : ajoute dans `.env` `ONEREF_MMAD_ROOT=D:\MMAD` et `ONEREF_MMAD_JSON=D:\MMAD\mmad.json`.

## 6bis. Échantillon de MMAD (pour tester sans télécharger 28 Go)

```powershell
python scripts/download_mmad.py --sample                       # mmad.json + DS-MVTec.zip (1,6 Go), zip supprimé après extraction
python scripts/download_mmad.py --sample --zips VisA.zip       # autre petit zip (1,9 Go)
python scripts/download_mmad.py --sample --dest D:\MMAD        # autre disque, puis ONEREF_MMAD_ROOT=D:\MMAD et ONEREF_MMAD_JSON=D:\MMAD\mmad.json dans .env
```
Ensuite ajoute `--available-only` à chaque commande du §7 : seules les images téléchargées sont utilisées (le programme affiche combien il en a trouvé, et s'arrête avec un message si c'est 0). Non vérifié : la correspondance entre l'arborescence des zips et les chemins de `mmad.json` (si 0 image trouvée, regarde `Get-ChildItem data\MMAD`). Attention au disque : il faut ~2 à 4 Go libres pendant le téléchargement.

## Groq avec repli sur Gemini

Avec `ONEREF_USE_GROK=true` (Groq) et les variables Gemini `ONEREF_LLM_*`, toute erreur de Groq (quota 429, panne) relance automatiquement l'appel sur le modèle Gemini du même rôle. Désactivable : `ONEREF_BACKUP_ENABLED=false`.

**Plusieurs clés :** `ONEREF_GROK_API_KEY` (clé 1) + `ONEREF_GROK_API_KEYS=cle2,cle3,cle4,cle5` pour Groq ; `ONEREF_LLM_API_KEY` (clé 1) + `ONEREF_LLM_API_KEYS=cle2,...` pour Gemini. Ordre d'essai : clés Groq, puis clés Gemini ; on passe à la suivante à chaque erreur d'API (quota...). Chaque appel recommence par la première clé, donc une clé épuisée ajoute un petit délai par appel : retire-la du `.env` quand elle est vide pour la journée.

## 7. Le workflow, étape par étape

```powershell
# $env:PYTHONPATH="src"   # seulement si tu n'utilises pas uv

# 7.1 Smoke test sans GPU ni réseau (agents factices), 20 images
uv run python -m inspector.main --split dev --mock --limit 20 --run-name smoke

# 7.2 Sonde : 50 images avec les vrais modèles (mesure temps et quota avant de grossir)
uv run python -m inspector.main --split dev --limit 50 --run-name probe

# 7.3 Run dev complet (~10 % de MMAD, ~840 images) : on règle prompts, seuils et poids ICI uniquement
uv run python -m inspector.main --split dev --run-name dev01

# 7.4 Test final : UNE seule fois, configuration figée
uv run python -m inspector.main --split test --frozen-config --run-name test01
```

Chaque commande affiche la précision par sous-tâche, la moyenne des 7 sous-tâches et un intervalle de confiance bootstrap à 95 %.

### Ce qui est écrit sur disque
- `outputs\<run>\telemetry.jsonl` : une ligne JSON par événement (sorties de chaque agent, métriques finales).
- `outputs\<run>\<passe>\<image>.json` : cache par passe et par image.
- `data\splits\` : découpage dev/test (graine fixe).

### Reprise après une coupure
Relance **exactement la même commande** (même `--run-name`) : les images déjà traitées sont relues dans le cache.

## 8. Quota Gemini gratuit

Un run complet = des milliers de questions × 3 passes LLM. Le quota gratuit (par minute et par jour) peut être dépassé : lance d'abord la sonde de 50 images, regarde le temps et les erreurs 429, puis augmente `--limit`.

## 9. Dépannage

| Message | Cause | Solution |
|---|---|---|
| `Espace insuffisant sur le disque` | `C:` plein | libérer de la place, `uv cache clean` |
| `402 ... Purchase pre-paid credits` | routeur HF sans crédits | passer à Gemini ou vLLM |
| `400 ... not supported by any provider you have enabled` | modèle absent du routeur | `GET <base_url>/models`, choisir un id servi |
| `400 ... Please pass a valid API key` | clé Gemini absente ou fausse | vérifier `ONEREF_LLM_API_KEY` dans `.env` |
| `404 ... is no longer available` | modèle Gemini retiré | suivre le nom suggéré dans le message |
| `401` au chargement de DINOv3 | licence non acceptée ou `HF_TOKEN` absent | accepter la licence, définir `HF_TOKEN` |
| `No LLM server at ...` | mode vLLM sans serveur | démarrer le serveur ou `ONEREF_VLLM_AUTOSTART=true` |
| `Refusing to run the test split without --frozen-config` | garde-fou | ajouter `--frozen-config` (une seule fois) |

## 10. Non vérifié à ce jour

DINOv3 réel, run de bout en bout sur MMAD, format JSON des lecteurs avec Gemini, envoi d'images en base64 par le pipeline (le test de connexion envoie une URL).
