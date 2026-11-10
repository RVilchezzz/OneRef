# OneRef – version Colab

Copie autonome de `src/inspector/` + notebook `OneRef_Colab.ipynb`. Modèles servis par **vLLM local** (un serveur par passe, `ONEREF_VLLM_AUTOSTART=true`) : protocole strict respecté.

## Mise en place
1. Sur Drive, crée `MyDrive/OneRef/` avec :
   - `Colab/` (ce dossier, uploadé tel quel)
   - `MMAD/` (images + `mmad.json`) : à uploader, ou à télécharger avec la cellule 5b du notebook (commentée, ~28 Go)
2. Colab : Runtime → GPU **L4** (recommandé ; T4 = checkpoints quantifiés, IDs à vérifier dans la cellule 1).
3. Secret Colab `HF_TOKEN` (accepter d'abord la licence DINOv3 sur Hugging Face).
4. Ouvre `OneRef_Colab.ipynb` et exécute les cellules dans l'ordre : smoke test → sonde 50 images → run dev complet. Le split test se lance une seule fois (cellule commentée).

## Reprise
Les sorties (`outputs/`, `splits/`) sont sur Drive : relancer la cellule après une coupure reprend au dernier cache.

## Resynchroniser après une modification du code
Depuis `OneRef/` : `rm -rf Colab/inspector && cp -r src/inspector Colab/inspector && cp tests/test_workflow.py Colab/tests/` (puis supprimer les `__pycache__`).

## Non vérifié
Jamais exécuté sur un vrai Colab : installation de vLLM à côté du torch de Colab (un redémarrage du runtime peut être demandé), mémoire L4/T4, IDs des checkpoints quantifiés, InternVL3 sous vLLM, parser d'outils `hermes`.
