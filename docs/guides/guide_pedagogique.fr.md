# Guide Complet et Explicatif : Inspecteur Visuel Industrielle Multi-Agent (1-Shot)

## 1. Vision Globale du Projet

### Quel est le problème concis ?
Dans une chaîne de fabrication industrielle (automobile, électronique, pièces mécaniques, agroalimentaire), vérifier si une pièce est conforme ou défectueuse est crucial pour éviter la livraison de produits cassés ou dangereux. 

Aujourd'hui, entraîner un modèle de Deep Learning classique demande **des milliers d'images annotées à la main** pour chaque nouvelle pièce ou anomalie. En usine, c'est impraticable : un nouveau modèle de pièce sort chaque semaine !

### Notre Solution : Inspection 1-Shot ("Une seule image de référence")
L'objectif est d'utiliser un système d'Intelligence Artificielle capable de regarder **1 seule image d'un objet parfait (sans défaut)** et de la comparer avec la pièce actuelle en sortie de chaîne pour :
1. Répondre à des questions complexes de contrôle qualité ("Le logo est-il présent ?", "La vis en haut à droite est-elle manquante ?", "La fissure sur le boîtier dépasse-t-elle le toléré ?").
2. Localiser précisément les anomalies visuelles.

---

## 2. Le Benchmark de Référence : MMAD

Pour mesurer l'efficacité de notre système, nous utilisons le benchmark **MMAD (Multi-modal Industrial Anomaly Detection)** :
* **8 366 images** industrielles haute définition.
* **39 670 questions-réponses** d'évaluation.
* **7 Sous-tâches fondamentales** :
  1. *Anomaly Existence* (Existe-t-il un défaut ?)
  2. *Anomaly Location* (Où se situe l'anomalie ?)
  3. *Anomaly Type* (Rayure, bosse, composant manquant...)
  4. *Anomaly Color* (Changement de teinte ou brûlure)
  5. *Anomaly Quantity* (Combien de défauts ?)
  6. *Industrial Compliance* (Respect des normes)
  7. *Reasoning & Action* (Recommandation d'action suite au défaut)

### Le Constat Scientifique (Gap de performance)
* **Les experts humains** atteignent **86,7%** d'exactitude sur MMAD.
* **GPT-4o seul** n'atteint que **74,9%** (soit un écart majeur de 13 points !).
* **Pourquoi GPT-4o ou Qwen-VL peinent-ils seuls ?** Parce que les modèles Multimodaux Généraux (MLLM) souffrent d'un "effet de cécité spatiale" : ils ont du mal à repérer des micro-fissures ou de minuscules rayures sur des images globales haute résolution.
* **Découverte clé** : Dès qu'on indique manuellement à un MLLM l'endroit exact de l'anomalie (Crop / Boîte englobante), sa précision explose.

---

## 3. Inspirations et Différenciation : SiGMA vs Notre Approche

### Ce que fait le papier de référence SiGMA (2026) :
SiGMA obtient 85.2% de précision en combinant des MLLM avec des cartes de saillance. Cependant, SiGMA possède deux limites majeures :
1. **Accès mémoire massif** : SiGMA utilise 256 images d'objets sains en mémoire (Memory Bank) pour comparer.
2. **Fine-Tuning sur MMAD** : Ils ré-entraînent le modèle sur le jeu de données test.

### Notre Inscription Scientifique (Training-Free 1-Shot) :
1. **Strictement 1-Shot** : 1 seule photo de référence conforme, pas de Memory Bank géante.
2. **Zero-Training / Training-Free** : Aucun entraînement ou fine-tuning sur MMAD. Tout repose sur l'assemblage intelligent de modèles pré-entraînés via un workflow agentique.
3. **Open-Source & Modèles $\le 8\text{B}$** : Pas de dépendance aux API payantes comme GPT-4o. Exécution locale et sur Google Colab.

---

## 4. Architecture Agentique (Inspirée de PasteurAIze / Graphify)

Dans le projet **PasteurAIze**, une architecture modulaire solide basée sur **PydanticAI**, **vLLM/OpenAI API standards**, **Logfire** (traçabilité), et un **orchestrateur déterministe** a été développée. Nous reprenons exactement ce paradigme pour notre inspecteur visuel.

```
                  +-----------------------------------+
                  |         Image de Test             |
                  |     + Image Ref (1-shot)          |
                  |     + Question MMAD               |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     ORCHESTRATEUR CENTRAL         |
                  |    (State Graph Déterministe)     |
                  +-----------------------------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
+-----------------------+ +--------------------+ +--------------------+
|  WORKER 1: Visual     | | WORKER 2: Reader A | | WORKER 3: Reader B |
|  Feature Expert       | | (Qwen3-VL-8B)      | | (InternVL3-8B)     |
| (DINOv3 Patch Match)  | +--------------------+ +--------------------+
+-----------------------+           |                       |
            |                       |                       |
            | (Anomaly BBox & Crop) |                       |
            +----------------------->                       |
                                    |                       |
                                    v                       v
                  +-----------------------------------+
                  |        SERVICE AGENT (QA)         |
                  |    Critic & Schema Validator      |
                  |    (Pydantic-AI Structured)       |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |       AGRÉGATEUR & LOGGING        |
                  |   (Voting Engine + Logfire)       |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |         Décision Finale           |
                  +-----------------------------------+
```

### Rôle des Composants :

#### A. Workers (Agents Spécialisés d'Exécution)
1. **Worker 1 - Visual Feature Expert (DINOv3)** :
   * Ne lit pas de texte. Il compare mathématiquement les cartes d'éléments visuels (*embeddings*) de l'image de test et de l'image de référence 1-shot.
   * Il calcule une carte de différence (*Difference Heatmap*) et isole la région anormale sous forme de **Bounding Box (Crop)**.
2. **Worker 2 - Reader A (Qwen3-VL-8B)** :
   * Modèle multimodal qui prend la question, l'image globale et le **Crop zoomé** généré par DINOv3.
   * Répond avec un raisonnement structuré.
3. **Worker 3 - Reader B (InternVL3-8B)** :
   * Deuxième modèle multimodal indépendant (diversité d'architecture). Reçoit les mêmes éléments pour éviter les biais d'un seul modèle.

#### B. Service Agent (Agent Critic & Contrôleur de Qualité)
* Construit avec **PydanticAI** (comme dans PasteurAIze).
* Reçoit les sorties des Readers A et B.
* **Validation de Schéma Strict (Pydantic)** : S'assure que le format de réponse (JSON), la justification et le choix (A/B/C/D ou Oui/Non) sont valides.
* **Validation Logique (Critic)** : Si Reader A et Reader B sont en contradiction flagrante, le Service Agent relance **un seul tour de ré-évaluation** (Re-query loop) en fournissant les arguments contradictoires.

#### C. Support & State Management
* **State & Memory** : Un objet `InspectState` (Pass-based State) transmet l'historique de chaque étape (Images, BBoxes, réponses temporaires).
* **Orchestrateur Déterministe** : Workflow Python explicite (sans boucle infinie imprévisible d'agents).
* **Logging & Telemetry** : Suivi des traces d'exécution avec **Logfire**, enregistrement séquentiel dans un fichier `results.jsonl`.

---

## 5. Gestion des Ressources Calculs (Compute)

* **Contraintes** : Modèles 8B multimodal + DINOv3 sur une seule machine.
* **Environnement** :
  * Machine Locale : NVIDIA RTX 3060 (12 Go VRAM) - utile pour le prototype et les validations légères.
  * Google Colab Pro : GPU L4 (24 Go VRAM) ou T4. Budget limité à 100 unités Colab.
* **Stratégie d'optimisation (Pass-Based Pipeline)** :
  * Au lieu de charger Qwen3-VL (8B) et InternVL3 (8B) simultanément en VRAM (ce qui ferait déborder les 24 Go de VRAM), le workflow s'exécute par **passes successives** :
    1. *Passe 1* : Extraction DINOv3 sur tout le sous-ensemble $\rightarrow$ Sauvegarde des Crops.
    2. *Passe 2* : Chargement de Qwen3-VL-8B via `vLLM` / `Unsloth` (Quantification 4-bit / 8-bit) $\rightarrow$ Génération des réponses Reader A.
    3. *Passe 3* : Déchargement du modèle A, Chargement de InternVL3-8B $\rightarrow$ Génération Reader B.
    4. *Passe 4* : Agrégation et évaluation par le Critic.

---

## 6. Protocole Évaluatif et Baselines

Pour prouver que l'architecture multi-agent apporte une réelle valeur ajoutée par rapport aux modèles isolés, nous comparons nos résultats avec 5 baselines :

* **B0 - Random Baseline** : Choix aléatoire (Borne inférieure).
* **B1 - Qwen3-VL-8B seul** (Sans crop DINOv3, sans multi-agent).
* **B2 - InternVL3-8B seul** (Sans crop DINOv3, sans multi-agent).
* **B3 - Vote Majoritaire Isomorphe (Budget Égal)** : Qwen3-VL interrogé 3 fois avec des températures différentes.
* **B4 - Vote Hétérogène Simple** : Qwen3-VL + InternVL3 sans visual expert (sans crop).
* **Notre Système (MAS 1-Shot)** : DINOv3 + Qwen3-VL + InternVL3 + Critic Pydantic.

### Validation Statistique :
* **Test de McNemar** : Pour vérifier si le gain de précision entre B1/B2 et notre MAS est statistiquement significatif ($p < 0.05$).
* **Intervalle de Confiance Bootstrap (95%)** : Pour garantir la robustesse des métriques sur les 7 sous-tâches de MMAD.
* **Taxonomie MAST (Multi-Agent System Taxonomy)** : Classification méthodique des erreurs (Erreur de perception visuelle, Erreur de validation Critic, Erreur de format Pydantic).