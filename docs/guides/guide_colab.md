# Guide : version Colab (vLLM, Qwen3-VL-8B et InternVL3-8B, sans payer Hugging Face)

Pour qui : tu veux le **plan de base** du projet : modèles open ≤ 8B servis par **vLLM** sur un GPU Colab, protocole strict respecté. Coût : uniquement des unités Colab.

Deux façons de faire, au choix :

| | Dossier `Colab/` + `OneRef_Colab.ipynb` | `Notebook/OneRef.ipynb` en mode vLLM |
|---|---|---|
| Code | copie de `src/inspector` (ancienne version, vLLM par défaut) | recréé par le notebook |
| Données et sorties | sur Google Drive (reprise facile) | disque Colab, perdu à la déconnexion (sauf chemins vers Drive) |
| Recommandé | **oui** | si tu veux un seul fichier |

## Comment fonctionne vLLM

`vllm serve <modèle>` charge un modèle sur le GPU et ouvre un serveur HTTP compatible OpenAI (`http://localhost:8000/v1`). Le pipeline l'appelle comme n'importe quelle API. Avec `ONEREF_VLLM_AUTOSTART=true`, **chaque passe démarre son propre serveur** avec son modèle puis l'arrête : jamais deux modèles de 8B en même temps sur le GPU. Ordre des passes : expert DINOv3 → lecteur A (Qwen3-VL-8B) → lecteur B (InternVL3-8B) → critique (Qwen3-VL-8B).

## 0. Prérequis

- Google Colab (Pro conseillé : ~100 unités). **GPU L4** (bf16, recommandé). T4 possible avec versions quantifiées (voir §6).
- Token Hugging Face avec **licence DINOv3 acceptée** : https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
- Les poids Qwen et InternVL se téléchargent gratuitement depuis Hugging Face.
- ~28 Go d'espace Drive pour MMAD (ou disque local Colab, volatil).

## A. Version dossier `Colab/` (recommandée)

### A1. Préparer Google Drive
Dans « Mon Drive », crée l'arborescence :
```
MyDrive/OneRef/
├── Colab/   <- uploade ici le dossier OneRef\Colab tel quel (inspector/, tests/, pytest.ini, requirements_colab.txt, OneRef_Colab.ipynb)
└── MMAD/    <- images + mmad.json (upload, ou cellule 5b du notebook)
```

### A2. Ouvrir le notebook
Drive → `OneRef/Colab/OneRef_Colab.ipynb` → ouvrir avec Google Colab.
Menu **Exécution → Modifier le type d'exécution → GPU L4**.

### A3. Secret `HF_TOKEN`
Icône clé (Secrets) à gauche → « Ajouter un secret » → nom `HF_TOKEN`, valeur ton token, **accès au notebook activé**.

### Échantillon de MMAD
Cellule 5 : `MMAD_MODE = "sample"` (défaut) télécharge `mmad.json` + `DS-MVTec.zip` (1,6 Go) sur Drive via la cellule 5b, et les cellules 6 à 8 ajoutent `--available-only`. Pour tout MMAD : `"full"` ; si MMAD est déjà sur Drive : `"none"`. Le fichier `download_mmad.py` doit être à la racine du dossier `Colab/` sur Drive (ré-uploade `Colab/` après ces changements). Non vérifié : correspondance zips / chemins de `mmad.json`.

### A4. Exécuter les cellules dans l'ordre

| Cellule | Rôle | Attendu |
|---|---|---|
| 1 | détecte le GPU, profil de modèles (T4 → AWQ), `ONEREF_VLLM_ARGS` | nom du GPU affiché |
| 2 | monte Drive, copie le code dans `/content/OneRef_Colab` | autorisation Drive à accepter |
| 3 | `pip install -r requirements_colab.txt` (vLLM inclus) | long ; si Colab demande de **redémarrer l'exécution**, fais-le puis relance depuis la cellule 1 |
| 4 | lit le secret `HF_TOKEN` | pas d'erreur |
| 5 | variables `ONEREF_*` (chemins sur Drive, autostart vLLM) | — |
| 5b | télécharge MMAD sur Drive selon `MMAD_MODE` (échantillon 1,6 Go par défaut), **une seule fois** | `mmad.json` dans `MyDrive/OneRef/MMAD` |
| 6 | pytest + smoke test `--mock --limit 20` | tests verts, métriques affichées |
| 7 | sonde 50 images avec les vrais modèles | **note le temps et les unités Colab consommées** |
| 8 | run dev complet | reprenable |
| 9 | dernières lignes du log JSONL | — |
| fin | test final, cellule commentée | **une seule fois**, `--frozen-config` |

### A5. Reprise après une coupure
Sorties dans `MyDrive/OneRef/outputs` et `splits`. Après une déconnexion : relance les cellules 1 à 5, puis la cellule du run (même `--run-name`) : le cache est relu.

### A6. Resynchroniser `Colab/` après une modification du code
Depuis `OneRef\` (Git Bash) :
```bash
rm -rf Colab/inspector && cp -r src/inspector Colab/inspector && cp tests/test_workflow.py Colab/tests/
```
Puis supprimer les `__pycache__` et ré-uploader sur Drive. Attention : la copie actuelle de `Colab/` est antérieure aux réglages `ONEREF_LLM_PROVIDER` (routeur par défaut) ; la copie a été resynchronisée et la cellule 5 définit désormais `ONEREF_LLM_PROVIDER=vllm`.

## B. Version `Notebook/OneRef.ipynb` en mode vLLM

1. Colab → Fichier → Importer le notebook `OneRef/Notebook/OneRef.ipynb`. GPU L4.
2. Cellule 1 : exécute le `pip install`, **puis décommente et exécute** `!pip install -q vllm` (redémarre l'exécution si demandé, puis reprends).
3. Cellule 2 (configuration) :
   ```python
   PROVIDER = "vllm"
   from google.colab import userdata
   HF_TOKEN = userdata.get("HF_TOKEN")
   ```
   Pour garder données et sorties sur Drive :
   ```python
   from google.colab import drive; drive.mount("/content/drive")
   ```
   puis remplace les chemins `ONEREF_MMAD_ROOT`, `ONEREF_MMAD_JSON`, `ONEREF_OUTPUTS_DIR`, `ONEREF_SPLITS_DIR` par des chemins sous `/content/drive/MyDrive/OneRef/...`.
4. MMAD : décommente la cellule optionnelle (versions Linux, valables sur Colab).
5. Exécute « Code du package », « Tests », puis les cellules d'exécution (smoke → sonde 50 → dev).

> En mode vLLM, **ne lance pas la cellule « Test de connexion »** : elle appelle `http://localhost:8000/v1` directement, et en mode autostart le serveur n'existe que pendant les passes.

## C. Variables vLLM (rappel)

```
ONEREF_LLM_PROVIDER=vllm
ONEREF_LLM_BASE_URL=http://localhost:8000/v1
ONEREF_LLM_API_KEY=EMPTY
ONEREF_VLLM_AUTOSTART=true        # false = tu lances `vllm serve <modèle>` toi-même
ONEREF_VLLM_ARGS=["--max-model-len","8192","--gpu-memory-utilization","0.9","--trust-remote-code","--enable-auto-tool-choice","--tool-call-parser","hermes"]
```
`--enable-auto-tool-choice --tool-call-parser hermes` sont nécessaires au critique (il utilise un outil).

## 6. T4 (16 Go, pas de bf16)

La cellule 1 de `OneRef_Colab.ipynb` bascule automatiquement sur des checkpoints AWQ avec `--dtype half` :
`Qwen/Qwen3-VL-8B-Instruct-AWQ` et `OpenGVLab/InternVL3-8B-AWQ`. **Ces identifiants ne sont pas vérifiés** : cherche-les sur huggingface.co et corrige le dictionnaire `T4_MODELS` si besoin.

## 7. Dépannage

| Problème | Solution |
|---|---|
| `vllm serve ... exited with code` | mémoire GPU insuffisante : baisser `--gpu-memory-utilization` ou `--max-model-len`, ou passer sur L4 |
| `not ready after 900s` | téléchargement des poids lent : augmenter `ONEREF_VLLM_STARTUP_TIMEOUT_S` |
| Conflit `torch` après `pip install` | redémarrer l'exécution Colab, relancer depuis la cellule 1 |
| 401 au chargement de DINOv3 | licence non acceptée ou secret `HF_TOKEN` manquant |
| `FileNotFoundError ... mmad.json` | MMAD pas au bon endroit sur Drive |
| Run interrompu | relancer la même cellule : reprise au cache |

## 8. Non vérifié

Jamais exécuté sur un vrai Colab : installation de vLLM à côté du torch de Colab, mémoire L4/T4, IDs AWQ pour T4, InternVL3 sous vLLM, parser d'outils `hermes`. Mesure d'abord le coût sur 50 images (cellule 7) avant de fixer la taille du run.
