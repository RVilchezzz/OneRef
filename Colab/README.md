# OneRef – version Colab

Copie autonome de `src/inspector/` + notebook `OneRef_Colab.ipynb`. Modèles servis par **vLLM local** (un serveur par passe, `ONEREF_VLLM_AUTOSTART=true`) : protocole strict respecté.

## Marche à suivre (vLLM)

Deux profils, choisis en cellule 1 (`PROFILE`) :

| Profil | Lecteurs | Critic et juge | GPU |
|---|---|---|---|
| `8B` (défaut) | Qwen3-VL-8B, InternVL3.5-8B | Gemma 4 E4B | **L4** minimum (poids ≈ 16 Go) |
| `4B` | Qwen3-VL-4B, InternVL3.5-4B | T4 : Qwen3-VL-4B et Ministral-3-3B ; sinon Gemma 4 E2B | T4 possible |

Le profil 4B a tourné sur T4. **Le profil 8B n'a encore jamais été exécuté.**

### Une seule fois
1. Hugging Face : créer un token **Read**. Pour DINOv3, accepter la licence sur la page `facebook/dinov3-vitb16-pretrain-lvd1689m` ; un token à grain fin doit en plus autoriser la lecture des dépôts « gated ». Sans cet accès (403), l'expert bascule sur **DINOv2** : le run fonctionne, mais les seuils de l'expert ont été calibrés avec DINOv3.
2. Colab : panneau **Secrets** (icône clé) → ajouter `HF_TOKEN` et activer l'accès pour le notebook. Les secrets sont propres à chaque compte Google.

### À chaque session
1. Ouvrir le notebook : dans [Colab web depuis GitHub](https://colab.research.google.com/github/RVilchezzz/OneRef/blob/Test_implementation/Colab/OneRef_Colab.ipynb), ou en local dans VS Code (extension Google Colab : **Select Kernel → Colab → New Colab Server**). Rien à uploader : la cellule 2 clone le code.
2. GPU **L4** (profil 8B) ou T4 (profil 4B).
3. Lancer les cellules **une par une avec leur bouton ▶** :

| Cellule | Rôle | À faire |
|---|---|---|
| 1 | profil, GPU, arguments vLLM | régler `PROFILE` ; 8B sur T4 s'arrête avec un message |
| 2 | Drive + clone | valider l'autorisation Drive ; dans VS Code, si le montage échoue : commande **Colab: Mount Google Drive to Server** |
| 3 | installation | si Colab demande un redémarrage : repartir de la cellule 1 |
| 4 | token HF | si le secret n'est pas accessible : `from getpass import getpass; os.environ["HF_TOKEN"] = getpass("HF_TOKEN : ")` |
| 5 | variables et config du profil | |
| 5b | MMAD sur Drive | `subset` : `N_SAMPLE` images dev et leurs références |
| 6 | tests sans GPU | doit passer avant d'utiliser le GPU |
| 6b | démarre chaque modèle une fois (hors T4) | la réponse ne doit contenir aucun bloc de réflexion |
| 7 | sonde, run `probe_<profil>` | se termine par un **VERDICT** |
| 8 | **dev complet**, run `dev_full_S_<profil>` | ne démarre que si la sonde passe ; vérifier d'abord les unités restantes |
| 9 | bilan du dev complet | |

4. À la fin : **déconnecter et supprimer l'environnement d'exécution** (sinon les unités continuent d'être consommées).

### Seuils de la sonde (`scripts/check_run.py`)
Toutes les passes terminées ; moins de 5 % de réponses invalides par lecteur au 1er passage ; critic LLM en échec sur moins de 10 % des images ; le juge répond ; au plus 2 % de questions sans réponse ; expert **DINOv3** (sa calibration en dépend). Le script projette la durée sur 835 images, mais **ne connaît pas les unités restantes** : heures projetées × débit du GPU doit tenir dans le solde avec 15 % de marge.

### Résultats et reprise
Tout est sur Drive : `MyDrive/OneRef/outputs/<run>/<passe>/<id>.json` et `telemetry.jsonl`. Après une coupure, relancer les cellules 1 à 5 puis la même commande : les images déjà en cache sont sautées. Le cache est lié au nom du run : **changer de modèle ou de réglage = nouveau `--run-name`**.

### Variantes
- **Aggregator LLM** : `os.environ["ONEREF_AGGREGATOR_MODEL"] = os.environ["ONEREF_JUDGE_MODEL"]` après la cellule 5, avec un nouveau nom de run. Résultat non validé, voir `docs/reports/erreur.md` §5.
- **Split test** : une seule fois, config figée, jamais pour régler des paramètres.

## Ce qui ne marche pas sur T4
Gemma 4 E2B (shared memory Triton), SmolVLM2 (`num2words` manquant), MiniCPM-V-4 en vision. Détails et messages d'erreur : `docs/reports/erreur.md`.

## Non vérifié
Profil 8B et GPU L4 (mémoire, Gemma 4 sous vLLM, durée), A100, modes MMAD `sample` et `full`, dev complet sur Colab.

## Resynchroniser après une modification du code
MMAD en échantillon : dans la cellule 5, `MMAD_MODE = "sample"` télécharge `mmad.json` + un petit zip (DS-MVTec, 1,6 Go) sur Drive, et les cellules d'exécution ajoutent `--available-only`. `"full"` = tout (~28 Go), `"none"` = déjà sur Drive. Le fichier `download_mmad.py` doit rester à la racine de `Colab/`.

Depuis `OneRef/` : `rm -rf Colab/inspector && cp -r src/inspector Colab/inspector && cp tests/test_workflow.py Colab/tests/ && cp scripts/download_mmad.py Colab/` (puis supprimer les `__pycache__`), commit et push : le notebook clone GitHub.
