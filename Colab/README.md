# OneRef – version Colab

Copie autonome de `src/inspector/` + notebook `OneRef_Colab.ipynb`. Modèles servis par **vLLM local** (un serveur par passe, `ONEREF_VLLM_AUTOSTART=true`) : protocole strict respecté.

## Marche à suivre (T4, vLLM 0.34)

La pipeline a tourné sur T4 avec ces modèles, appliqués par une cellule ajoutée à la main ; la cellule 5 les applique désormais elle-même (vérifié en local seulement).

### Une seule fois
1. Hugging Face : créer un token **Read**. Pour DINOv3, accepter la licence sur la page `facebook/dinov3-vitb16-pretrain-lvd1689m` ; un token à grain fin doit en plus autoriser la lecture des dépôts « gated ». Sans cet accès (403), l'expert bascule sur **DINOv2** : le run fonctionne, mais les seuils de l'expert ont été calibrés avec DINOv3.
2. Colab : panneau **Secrets** (icône clé) → ajouter `HF_TOKEN` et activer l'accès pour le notebook. Les secrets sont propres à chaque compte Google.

### À chaque session
1. Ouvrir [le notebook depuis GitHub](https://colab.research.google.com/github/RVilchezzz/OneRef/blob/Test_implementation/Colab/OneRef_Colab.ipynb). Rien à uploader : la cellule 2 clone le code.
2. Exécution → Modifier le type d'exécution → **GPU T4**.
3. Lancer les cellules **une par une avec leur bouton ▶** (Ctrl+Entrée peut enchaîner sur les suivantes) :

| Cellule | Rôle | À faire |
|---|---|---|
| 1 | GPU et arguments vLLM | vérifier que `T4` s'affiche |
| 2 | Drive + clone | valider l'avertissement Colab et l'autorisation Drive (si `mount failed` : relancer) |
| 3 | installation | si Colab demande un redémarrage : redémarrer puis repartir de la cellule 1 |
| 4 | token HF | `SecretNotFoundError` = secret absent sur ce compte |
| 5 | variables et modèles | sur T4 : critic Qwen3-VL-4B, judge Ministral-3-3B (BF16) |
| 5b | MMAD sur Drive | mode `subset` : `N_SAMPLE` images dev et leurs références |
| 6 | tests sans GPU | doit passer avant d'utiliser le GPU |
| 7 | sonde, run `probe` | durée estimée : plus d'une heure pour 50 images sur T4 |

4. À la fin : Exécution → **Déconnecter et supprimer l'environnement d'exécution** (sinon les unités de calcul continuent d'être consommées).

### Résultats et reprise
Tout est sur Drive : `MyDrive/OneRef/outputs/<run>/<passe>/<id>.json` et `telemetry.jsonl`. Après une coupure, relancer les cellules 1 à 5 puis la même commande : les images déjà en cache sont sautées. Le cache est lié au nom du run : **changer de modèle ou de réglage = nouveau `--run-name`**.

### Variantes
- **200 à 300 images** : dans la cellule 5 mettre `N_SAMPLE`, relancer 5 et 5b, puis `!python -m inspector.main --split dev --run-name dev250 {OPT}`. Durée estimée (non mesurée) : 6 à 9 h sur T4.
- **Aggregator LLM** : `os.environ["ONEREF_AGGREGATOR_MODEL"] = os.environ["ONEREF_JUDGE_MODEL"]` après la cellule 5, avec un nouveau nom de run. Résultat non validé, voir `docs/reports/erreur.md` §5.
- **Cellule 8** : run dev (`dev01`) sur les images sélectionnées par `{OPT}` ; en mode `full` ce sont les 835 images, soit plusieurs heures.
- **Split test** : une seule fois, config figée, jamais pour régler des paramètres.

## Ce qui ne marche pas sur T4
Gemma 4 E2B (shared memory Triton), SmolVLM2 (`num2words` manquant), MiniCPM-V-4 en vision. Détails et messages d'erreur : `docs/reports/erreur.md`.

## Non vérifié
GPU L4 et A100 (dont Gemma 4 E2B dessus), modes MMAD `sample` et `full`, run de 200 images ou plus.

## Resynchroniser après une modification du code
MMAD en échantillon : dans la cellule 5, `MMAD_MODE = "sample"` télécharge `mmad.json` + un petit zip (DS-MVTec, 1,6 Go) sur Drive, et les cellules d'exécution ajoutent `--available-only`. `"full"` = tout (~28 Go), `"none"` = déjà sur Drive. Le fichier `download_mmad.py` doit rester à la racine de `Colab/`.

Depuis `OneRef/` : `rm -rf Colab/inspector && cp -r src/inspector Colab/inspector && cp tests/test_workflow.py Colab/tests/ && cp scripts/download_mmad.py Colab/` (puis supprimer les `__pycache__`), commit et push : le notebook clone GitHub.
