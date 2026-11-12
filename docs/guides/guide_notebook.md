# Guide : version notebook (`Notebook/OneRef.ipynb`)

Pour qui : tu veux **tout dans un seul fichier**. Les cellules `%%writefile` recréent le package `inspector/` dans le dossier courant du noyau, puis lancent tests et pipeline. Fonctionne dans VS Code, Jupyter ou Colab.

## 0. Prérequis

- Un noyau Python ≥ 3.11 avec `ipykernel` (VS Code te le propose, ou `pip install jupyterlab`).
- Token Hugging Face avec **licence DINOv3 acceptée** : https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- Clé Gemini gratuite : https://aistudio.google.com/apikey
- ~5 Go libres pour torch + transformers, et ~28 Go pour MMAD si tu le télécharges.

## 1. Ouvrir le notebook

1. Ouvre `OneRef\Notebook\OneRef.ipynb` dans VS Code (ou `jupyter lab`).
2. Choisis le noyau Python (en haut à droite).
3. Le **dossier courant du noyau est en général le dossier du notebook** (`OneRef\Notebook`) : `inspector/`, `tests/`, `data/` et `outputs/` seront créés **dedans**. Vérifie avec une cellule `import os; print(os.getcwd())`.

## 2. Exécuter dans l'ordre

| Cellule | À faire |
|---|---|
| **1. Installation** | Exécuter (`!pip install ...`). La cellule suivante (`vllm`) reste commentée sauf mode vLLM. |
| **2. Configuration** | **Modifier** `PROVIDER`, `HF_TOKEN`, `GEMINI_API_KEY` (voir ci-dessous), puis exécuter. |
| Télécharger MMAD (optionnel) | Décommenter et exécuter une seule fois (voir §4). |
| **3. Code du package** | Exécuter toutes les cellules `%%writefile inspector/...` (elles écrivent les fichiers). |
| Test de connexion | Exécuter : doit afficher une phrase décrivant la Statue de la Liberté. |
| **4. Tests** | Exécuter : environ 21 tests passent. |
| **5. Exécution** | Smoke test → sonde 50 images → run dev → derniers logs. |
| Test final | Cellule commentée, à lancer **une seule fois**. |

### La cellule de configuration

```python
PROVIDER = "gemini"           # "gemini" | "router" | "vllm"
HF_TOKEN       = "hf_XXXX"    # ou os.environ.get("HF_TOKEN", "")
GEMINI_API_KEY = "XXXX"       # ou os.environ.get("GEMINI_API_KEY", "")
```
- `gemini` : gratuit, hors protocole strict (borne haute).
- `router` : routeur Hugging Face, **crédits payants** requis (erreur 402 sinon).
- `vllm` : modèles Qwen3-VL-8B / InternVL3-8B servis par vLLM (GPU Linux/Colab, voir `guide_colab.md`).

Les variables d'environnement posées par cette cellule **l'emportent** sur un éventuel fichier `.env`. Ne commite pas le notebook avec de vraies clés écrites dedans : préfère les variables d'environnement.

## 2bis. Groq, repli Gemini, échantillon MMAD

Dans la cellule de configuration :
```python
PROVIDER = "groq"              # Qwen sur Groq ; repli auto sur Gemini si une clé Gemini est fournie
GROQ_API_KEY   = "gsk_XXXX"
GEMINI_API_KEY = "XXXX"        # optionnel mais conseillé (backup quand le quota Groq est épuisé)
MMAD_MODE = "sample"           # "sample" | "full" | "none"
SAMPLE_ZIPS = ["DS-MVTec.zip"]
MMAD_ROOT = "data/MMAD"        # autre disque si C: est plein
```
Les deux cellules suivantes téléchargent MMAD (`sample` = `mmad.json` + 1 zip de 1,6 Go, supprimé après extraction). Avec `sample`, les commandes d'exécution reçoivent automatiquement `--available-only`. Fonctionne sous Windows (pur Python).

## 3. Les cellules d'exécution

```text
# Smoke test sans GPU (agents factices)
!python -m inspector.main --split dev --mock --limit 20 --run-name smoke
# Sonde : 50 images
!python -m inspector.main --split dev --limit 50 --run-name probe
# Run dev complet (reprenable)
!python -m inspector.main --split dev --run-name dev01
# Test final (cellule commentée)
# !python -m inspector.main --split test --frozen-config --run-name test01
```
Chaque commande affiche la précision par sous-tâche, la moyenne des 7 et un IC bootstrap à 95 %. Relancer la même cellule après une coupure **reprend** au cache (`outputs/<run>/`).

## 4. Télécharger MMAD depuis le notebook

La cellule optionnelle utilise des commandes Linux (`find`, `unzip`, `cp`). **Sous Windows**, remplace-la par :

```python
from huggingface_hub import snapshot_download
snapshot_download("jiang-cc/MMAD", repo_type="dataset", local_dir="data/MMAD")

import zipfile, pathlib, shutil
for z in pathlib.Path("data/MMAD").rglob("*.zip"):
    zipfile.ZipFile(z).extractall(z.parent)

!git clone --depth 1 https://github.com/jam-cc/MMAD MMAD_code
shutil.copy(next(pathlib.Path("MMAD_code").rglob("mmad.json")), "data/MMAD/mmad.json")
```
(`HF_TOKEN` doit être défini avant : exécute d'abord la cellule de configuration.)

## 5. Si le code source change

Le notebook est **généré** à partir de `src/inspector` : si tu modifies le code source, le notebook ne se met pas à jour tout seul. Il faut le régénérer, ou modifier à la main les cellules `%%writefile` correspondantes puis les réexécuter. Le script de génération n'est pas dans le dépôt (il a servi une fois, dans le dossier temporaire de la session).

## 6. Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: inspector` | exécuter d'abord la configuration (`PYTHONPATH`) et les cellules « Code du package » |
| autre `ModuleNotFoundError` | cellule 1 non exécutée, ou mauvais noyau |
| Erreur 402 | `PROVIDER="router"` sans crédits : passer à `gemini` |
| Erreur 400 « valid API key » | `GEMINI_API_KEY` vide dans la cellule de configuration |
| Erreur 404 modèle retiré | adapter le nom dans `PROFILES` |
| Cellule des logs vide | `outputs/dev01` n'existe pas : lancer d'abord le run dev |
| `Espace insuffisant` | libérer `C:` |

## 7. Non vérifié

Le notebook n'a pas été exécuté de bout en bout dans Jupyter : seul le code de ses cellules a été extrait et testé (21 tests passent), et chaque profil de configuration a été chargé. Le run complet sur MMAD, DINOv3 réel et vLLM restent à valider.
