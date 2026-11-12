# Script oral : livraison I (15 min)

Multi-agent visual inspector · IA025 · Unicamp 2026 · Peter Roque Vilchez & Charles Maxime Doudy

Deck : `IA025 – Final Project Plan (Delivery I).pptx` (14 slides, en anglais). Chaque bloc `Script oral` ci-dessous figure aussi dans les notes du présentateur de `IA025 – Final Project Plan (Delivery I) – notes FR.pptx`. Version anglaise : `speaker_script_delivery_I.en.md`. Le texte affiché à l'écran est cité en anglais, comme sur les slides.

## Minutage et intervenants

Environ 14 min 30 de parole (le français est un peu plus long en mots, mais se dit plus vite).

| # | Slide | Intervenant | Durée |
|---|---|---|---|
| 1 | Couverture | Charles | 30 s |
| 2 | Description du projet | Charles | 90 s |
| 3 | Motivation | Charles | 90 s |
| 4 | Méthodologie | Peter | 120 s |
| 5 | Orchestration | Charles | 75 s |
| 6 | Ressources de calcul | Peter | 60 s |
| 7 | Données | Peter | 60 s |
| 8 | Ce que voit le système | Peter | 55 s |
| 9 | Les questions | Peter | 45 s |
| 10 | Métriques | Charles | 90 s |
| 11 | Résultats | Charles | 60 s |
| 12 | Références | Peter | 15 s |
| 13 | Planning | Peter | 45 s |
| 14 | Merci | Charles | 15 s |

Charles (environ 7 min 30) présente le problème, les lecteurs, le vérificateur, l'agrégateur et l'évaluation. Peter (environ 6 min 30) présente l'expert visuel, le matériel, les données et le planning. Chacun répond aux questions du jury sur sa partie.

## À savoir par cœur

1. **Les écarts.** Les humains experts obtiennent **86,7 %** (86,65). GPT-4o obtient **74,9 %**, soit **11,8 points** de moins. Qwen3-VL-8B, notre B1, obtient **73,9 %**, soit **12,8 points** de moins. Dire « environ 12 points derrière GPT-4o, environ 13 derrière notre baseline ouverte ». Ne pas dire « 13 points entre GPT-4o et les humains ».
2. **SiGMA (85,2 %)** utilise « un protocole différent, plus favorable ». Ne jamais dire « triche ». Les faits : jusqu'à 256 images normales par catégorie, récupération de texte MMAD, équipe et hyperparamètres choisis à partir des réponses de MMAD, et 4 GPU L20. Le leaderboard officiel précise lui-même que l'exclusion « signifie une différence de protocole, pas une accusation de faute ».
3. **Notre propre limite.** Notre split de dev (10 %) vient de MMAD, donc nous ne sommes pas éligibles à la piste stricte du leaderboard non plus. Le dev et le test sont disjoints, la configuration est gelée le 19 novembre et le test ne tourne qu'une fois. Le dire avant que le jury ne le fasse.

## Transitions

- 3 → 4, Charles : « Peter va maintenant suivre une image à travers le workflow. »
- 4 → 5, Peter : « Charles va maintenant montrer comment ces pièces s'inscrivent dans le cadre d'orchestration. »
- 5 → 6, Charles : « Peter, comment fait-on tourner tout ça sur notre matériel ? »
- 9 → 10, Peter : « Charles va maintenant expliquer comment nous allons juger tout cela. »
- 11 → 12, Charles : « Peter va conclure avec les références et le planning. »
- 13 → 14, Peter : « Charles ? »

---

## 1 · Couverture : Multi-agent visual inspector

### Contenu visuel
- Surtitre : IA025 · Deep Learning · Unicamp 2026
- Titre : **Multi-agent visual inspector**
- Sous-titre : une équipe de modèles pré-entraînés trouve les défauts de pièces industrielles à partir d'**une seule photo de pièce conforme**, validée sur le benchmark **MMAD**
- Équipe : Peter Roque Vilchez · Charles Maxime Doudy · Delivery I · 29 octobre 2026
- Images : « Reference » (bouteille conforme, MVTec AD) | « Part to inspect » (bouteille au goulot cassé)

### Script oral — Charles (~30 s)
Bonjour. Je suis Charles Maxime Doudy et, avec Peter Roque Vilchez, nous vous présentons notre projet final d'IA025 : un inspecteur visuel multi-agent. L'idée tient en une phrase : une équipe de modèles pré-entraînés détecte les défauts d'une pièce industrielle à partir d'une seule photo d'une pièce conforme, sans aucun entraînement, et nous le validons sur le benchmark public MMAD. À droite, un exemple réel : une bouteille conforme, et une bouteille au goulot cassé.

### À ajouter / améliorer
- Ajouter sous le titre la question de recherche en une ligne : « How much of the multi-agent gain survives with one reference and no tuning? »
- Légender la photo de gauche « Reference: the only normal image the system gets ».
- Le lien du dépôt sur la slide 14 (`MULTIAGENT_INSPECTOR`) n'est pas le même que dans `slides_content.md` (`OneRef`). Vérifier lequel fonctionne.

### Question du jury
**« Est-ce vraiment un système multi-agent, ou juste un pipeline ? »**
Dans le vocabulaire d'Anthropic, c'est un *workflow orchestré* : le flux de contrôle est fixé dans le code. Chaque agent reste un modèle distinct, avec son rôle, son schéma d'entrée/sortie et des messages journalisés, ce qui correspond exactement à la répartition travailleur / service / support d'Adimulam et al. Le flux fixe est un choix délibéré : il est reproductible, et MAST montre que c'est la coordination libre qui fait le plus échouer les systèmes multi-agents. La seule décision qu'un agent prend seul est la réouverture par le vérificateur.

---

## 2 · Description du projet : une bonne photo, une pièce, une question

### Contenu visuel
- Référence (conforme) | pièce à inspecter | question « Where is the defect? » A. Top right / B. Top left / C. Center / **D. Bottom right ✓**
- Hypothèse : une équipe **sans entraînement** (expert visuel, deux lecteurs MLLM, vérificateur) bat le meilleur modèle seul **et** un vote majoritaire à coût égal
- Barres (axe à partir de 50 %) : inspecteurs humains experts **86,7 %** · GPT-4o **74,9 %** · Qwen3-VL-8B (ouvert, notre baseline) **73,9 %**
- Légende : « A 13-point gap to humans »

### Script oral — Charles (~90 s)
Voici exactement ce que reçoit le système. Une photo d'une pièce conforme : la référence. La pièce à inspecter. Et une question à choix multiples, ici : où est le défaut ? La réponse est D, en bas à droite. Dans MMAD, cette seule bouteille est accompagnée de cinq questions de ce type : y a-t-il un défaut, de quel type, où, à quoi il ressemble, et quelle est sa conséquence.

Pourquoi ce cadre est-il important ? Quand une ligne de production change de produit, on n'a ni le temps ni les données pour entraîner un détecteur dédié. Souvent, on a une seule bonne photo. Les modèles de langage multimodaux, les MLLM, savent suivre ce type de consigne sans entraînement. Mais ils ne sont pas encore au niveau. Les inspecteurs humains experts atteignent 86,7 % sur MMAD. GPT-4o atteint 74,9 %, soit environ douze points de moins. Qwen3-VL-8B, le modèle ouvert qui nous sert de référence, atteint 73,9 %, environ treize points de moins. C'est cet écart que nous cherchons à réduire.

Notre hypothèse est mesurable : une équipe sans entraînement, faite de modèles ouverts d'au plus huit milliards de paramètres, un expert visuel, deux lecteurs MLLM et un vérificateur, bat à la fois le meilleur modèle seul et un vote majoritaire qui dépense le même calcul. Si ce n'est pas le cas, nous voulons comprendre pourquoi.

### À ajouter / améliorer
- Remplacer la légende « A 13-point gap to humans » par **« −12.8 pts vs our baseline · −11.8 vs GPT-4o »**.
- Ajouter une barre **humains non experts, 78,7 %** : les MLLM sont même sous les non-experts, c'est un argument fort.
- Afficher les 5 types de questions de la bouteille en petites pastilles (défaut ? / type / où / aspect / effet).

### Question du jury
**« En vrai, l'inspection est ouverte. Pourquoi évaluer avec des QCM ? »**
Le QCM donne un score objectif et reproductible, et rend 39 670 questions comparables entre tous les modèles publiés. Le niveau du hasard (50 % / 25 %) est pris en compte explicitement par la baseline B0. Le système rédige quand même un rapport libre pour l'opérateur, mais seules les réponses au QCM sont notées, et nous le déclarons comme limite.

---

## 3 · Motivation : les MLLM raisonnent bien dès qu'on leur dit où regarder

### Contenu visuel
- À gauche, MMAD (InternVL2-40B), Jiang et al. ICLR 2025, tableau 6 :
  - Discrimination d'anomalie : MLLM seul **63,8** · + carte PatchCore 60,0 · + carte AnomalyCLIP 59,6 · + vraie localisation **88,7**
  - Localisation du défaut : MLLM seul **52,9** · + meilleur détecteur 60,4 · + vraie localisation **81,0**
- À droite : figure de SiGMA (arXiv 2610.12310) + tableau : images normales **jusqu'à 256** vs **1** · réglé sur MMAD : oui (équipe + paramètres) vs split de dev seulement · GPU **4 × L20** vs Colab L4 / T4

### Script oral — Charles (~90 s)
Pourquoi une architecture multi-agent, et pourquoi un expert visuel en son cœur ? La réponse vient d'une expérience de l'article MMAD lui-même. À gauche, InternVL2-40B répond seul : 63,8 % en discrimination d'anomalies. Si on lui ajoute la carte d'anomalie d'un détecteur classique, PatchCore ou AnomalyCLIP, c'est pire, autour de 60 % : ces détecteurs se déclenchent aussi sur des pièces normales. Mais si on lui dit où se trouve vraiment le défaut, en transformant le masque de vérité terrain en texte, il passe à 88,7 %. La localisation passe de 52,9 à 81. C'est un oracle, une borne supérieure, mais le message est clair : les MLLM raisonnent bien dès qu'on leur indique où regarder. Leur goulot d'étranglement, c'est la perception fine.

Le travail le plus proche est SiGMA, publié ce mois-ci. Une équipe de MLLM avec un expert visuel DINOv3 atteint 85,2 %, tout près des experts humains. Mais sous un protocole différent et plus favorable : jusqu'à 256 images normales par catégorie au lieu d'une seule, du texte métier récupéré, une équipe et des hyperparamètres choisis à partir des réponses de MMAD, et quatre GPU L20. D'où notre question de recherche : quelle part de ce gain survit avec une seule photo de référence, des modèles ouverts, et aucun réglage sur le test ? Peter va maintenant suivre une image à travers le workflow.

### À ajouter / améliorer
- Hachurer les barres « + true location » avec la mention **« oracle (borne supérieure) »**, et les écarts en gras : **+24,9** et **+28,1**.
- Intituler le tableau SiGMA **« Different protocol »** et ajouter une ligne « Domain text: yes / no ». Éviter toute formulation qui suggère une faute.
- Note de bas de page : le leaderboard officiel classe SiGMA hors de la piste principale, pour une différence de protocole.

### Question du jury
**« Si les cartes des détecteurs classiques dégradent le MLLM, pourquoi la vôtre aiderait-elle ? »**
Dans MMAD, les cartes PatchCore et AnomalyCLIP étaient fournies brutes et se déclenchaient aussi sur des images normales, ce qui a dégradé la discrimination ; la localisation a quand même gagné +7,5. Nous faisons trois choses différemment. La sortie de l'expert est convertie en une région d'une grille 3×3, dans le vocabulaire des options de réponse, plus un recadrage. Il ne vote que sur la discrimination, avec un poids fixé sur le dev, et ce sont les lecteurs qui ont le dernier mot. Et SiGMA montre qu'un expert DINOv3 aide bel et bien au sein d'une équipe. S'il n'aide pas ici, les ablations le montreront, y compris « carte brute vs carte calibrée vs texte seul ».

---

## 4 · Méthodologie : une image à travers le workflow

### Contenu visuel
- **Entrée** : 1 référence + la pièce · « Where is the defect? » A / B / C / D
- **1 · Expert visuel** : compare les patchs avec la référence → carte, région 3×3, recadrage, score
- **2 · Deux lecteurs** : Qwen3-VL-8B et InternVL3-8B → réponse · confiance (mêmes photos + indice de l'expert ; réponse JSON)
- **3 · Vérificateur** : ✓ schéma valide / ✓ réponses cohérentes / ✓ accord avec l'expert · conflit → **réouverture une seule fois**, avec les preuves
- **4 · Agrégateur** : vote pondéré par lecteur et par sous-tâche ; l'expert vote sur « défaut ? »
- **5 · Journal** : chaque message, temps et nombre de tokens, en JSONL par image
- Images : référence, pièce, carte d'anomalie (exécution DINOv2 ViT-S/14 illustrative)

### Script oral — Peter (~120 s)
Merci Charles. Suivons une vraie image à travers le workflow. En entrée : la bouteille conforme, la bouteille cassée, et la question : où est le défaut ?

Étape un, l'expert visuel. C'est un encodeur DINOv3 gelé : rien n'est entraîné. Il découpe les deux photos en petits patchs. Pour chaque patch de la pièce, il cherche le patch le plus proche dans la référence et mesure la distance. Ce qui est loin de tout ce qui est normal est suspect. On obtient cette carte de chaleur, que nous avons calculée nous-mêmes sur cette paire avec DINOv2, la version précédente. Elle illumine le goulot cassé. L'expert traduit ensuite le pic en mots, par exemple « en bas à droite, score élevé », une case d'une grille trois par trois dans le vocabulaire des réponses de MMAD, plus un recadrage zoomé. En toute transparence : sur cette paire, notre version rapide avec DINOv2 a placé le pic dans la case du bas au centre, juste à côté de la bonne réponse. Le calibrer fait partie de la livraison II.

Étape deux, deux lecteurs de familles de modèles différentes, Qwen3-VL-8B et InternVL3-8B. Ils voient les mêmes photos plus l'indice de l'expert, et répondent en JSON : une réponse et une confiance.

Étape trois, le vérificateur. Il contrôle le format, la cohérence entre les réponses, par exemple « pas de défaut » en même temps que « le défaut est au centre », et l'accord avec l'expert. En cas de conflit, il rouvre la question une seule fois, avec les preuves, jamais plus, pour éviter les boucles.

Étape quatre, l'agrégateur vote avec des poids fixés sur le split de dev. Étape cinq, chaque message, chaque temps et chaque nombre de tokens est journalisé. Charles va maintenant montrer comment ces pièces s'inscrivent dans le cadre d'orchestration.

### À ajouter / améliorer
- Dessiner la boucle de réouverture en flèche pointillée du vérificateur vers les lecteurs, avec la mention **« max 1 »**.
- Légender la flèche expert → agrégateur « votes on discrimination only ».
- Légender honnêtement la carte : « exécution DINOv2 illustrative ; pic dans la case du bas au centre, vraie réponse en bas à droite ».
- Montrer une réponse JSON illustrative : `{"answer": "D", "confidence": 0.82}`.
- Indiquer sur la slide que le vérificateur est Qwen3-VL-8B avec un prompt de critique.

### Question du jury
**« Votre propre carte place le pic dans la mauvaise case. L'expert ne va-t-il pas induire les lecteurs en erreur ? »**
C'est précisément pour ça que l'expert ne décide jamais seul. Bas-centre contre bas-droite, c'est un cas limite de la grille 3×3. Les lecteurs voient quand même les images complètes et le recadrage, et le vérificateur ne rouvre qu'en cas de conflit net. Sur le dev, nous allons calibrer le seuil et filtrer les régions connexes. Une option consiste à donner les deux meilleures cases quand le pic tombe sur une frontière. L'ablation « carte brute / calibrée / texte seul » mesure quel format aide vraiment.

---

## 5 · Orchestration : chaque agent a un rôle dans le cadre

### Contenu visuel
- Figure : Adimulam et al., arXiv 2601.13671, fig. 2
- **Couche d'orchestration** → orchestrateur Python : plan fixe par sous-tâche
- **Agents travailleurs** → expert visuel · lecteur A · lecteur B
- **Agent de service (contrôle qualité)** → vérificateur : schéma, cohérence, réouverture une fois
- **Support · état** → agrégateur + journal JSONL de chaque message
- Leçons de conception : commencer par un workflow fixe simple (Anthropic, 2024) ; les gains multi-agents sont souvent faibles, donc chaque agent doit mériter sa place dans une ablation (MAST, NeurIPS 2025)

### Script oral — Charles (~75 s)
Merci Peter. L'article associé au cours, Adimulam et collègues, 2026, est un article de position. Il ne contient aucune expérience : nous l'utilisons donc comme vocabulaire et comme modèle de conception, pas comme preuve. Nous en retenons trois idées. D'abord, les rôles. Les agents travailleurs exécutent la tâche : notre expert visuel et les deux lecteurs. Un agent de service fait le contrôle qualité : notre vérificateur. Les agents de support gardent l'état : notre agrégateur et le journal JSONL. Ensuite, une couche d'orchestration qui décide qui s'exécute, et dans quel ordre. Chez nous, c'est du Python simple, avec un plan fixe par sous-tâche. Enfin, la validation de chaque sortie contre un schéma, que nous faisons avec Pydantic.

Deux leçons gardent la conception honnête. Anthropic recommande de commencer par le workflow fixe le plus simple, et de n'ajouter des agents que si une métrique s'améliore. Et MAST, une étude NeurIPS 2025 sur les échecs des systèmes multi-agents, montre que les gains face à un agent seul ou à un simple best-of-N sont souvent minimes. Chaque agent doit donc mériter sa place dans une ablation. Peter, comment fait-on tourner tout ça sur notre matériel ?

### À ajouter / améliorer
- Ajouter une petite colonne « Ce que nous ne reprenons pas » : les protocoles MCP / A2A (surdimensionnés pour 4–5 agents dans un seul processus ; exposer l'expert en MCP reste optionnel) et les chiffres industriels de l'article (tirés de blogs et de rapports de conseil, pas d'expériences).
- Note : « Plusieurs références de l'article ont de mauvais auteurs ; nous citons les articles originaux. »

### Question du jury
**« L'article recommande MCP et A2A. Pourquoi ne pas les utiliser ? »**
Ils standardisent la communication entre de nombreuses équipes et de nombreux services en production. Avec quatre ou cinq agents dans un seul processus Python, des messages JSON typés et validés par Pydantic donnent la garantie qui compte vraiment pour l'article, la validation de schéma, sans surcoût réseau. Exposer l'expert visuel comme outil MCP est une extension optionnelle. Cela ne changerait aucun résultat.

---

## 6 · Ressources de calcul : un GPU portable, Colab et un cache partagé

### Contenu visuel
- **Local · Peter** : RTX 3060 Laptop, **6 Go** → expert visuel DINOv3 ; débogage de tout le pipeline avec Qwen3-VL-4B en 4 bits
- **Partagé · Google Drive** : images MMAD + un cache JSONL par agent, reprise possible
- **Cloud · Charles** : Colab Pro, **L4 / T4** → lecteurs 8B servis avec **vLLM**, baselines et test final (**100 unités, pas d'A100**)
- Exécution par passes, un seul modèle sur le GPU à la fois : passe expert → passe lecteur A → passe lecteur B → passe vérificateur → agrégation
- Budget : coût mesuré sur 50 questions avant le test ; B1, B2 et B4 réutilisent les réponses des lecteurs en cache

### Script oral — Peter (~60 s)
Oui, nous avons le matériel nécessaire, à condition d'être rigoureux. En local, j'ai un GPU portable RTX 3060 avec 6 gigaoctets. C'est suffisant pour l'expert DINOv3, et pour déboguer tout le pipeline avec un petit modèle Qwen de 4 milliards de paramètres quantifié en 4 bits. Les deux lecteurs 8B tournent sur Colab Pro, sur des GPU L4 ou T4, servis avec vLLM, qui traite beaucoup de questions par lots. Nous n'utilisons jamais l'A100 : il consomme les unités de calcul plusieurs fois plus vite. L'idée clé est l'exécution par passes : chaque agent traite tout le split, écrit ses réponses sur Drive, puis libère le GPU. Deux modèles 8B ne sont donc jamais chargés ensemble, et une déconnexion ne fait rien perdre. Les baselines B1, B2 et B4 réutilisent gratuitement ces réponses en cache. Et avant le test, nous mesurons le coût sur 50 questions.

### À ajouter / améliorer
- Ajouter les estimations de coût de sources tierces : **T4 ≈ 1,76 unité/h, L4 ≈ 2,5–5, A100 ≈ 13–15**.
- Ajouter le plan B : **+100 unités pour 9,99 US$** avant de réduire le test.
- Après la livraison II, remplacer les estimations par les unités par question mesurées.

### Question du jury
**« 100 unités Colab, est-ce vraiment suffisant ? »**
Le coût est dans les appels aux MLLM, et le cache en supprime la plupart. B1, B2 et B4 réutilisent les réponses des lecteurs du système complet ; seuls les appels propres à S (lecteurs, vérificateur, réouvertures) et les échantillons supplémentaires de B3 sont nouveaux. 100 unités, c'est environ 57 heures de T4 ou 20 à 40 heures de L4. Nous mesurons les unités par question sur 50 questions avant de fixer la taille du test (1 000 à 2 000 images). Si cela ne suffit pas, nous achetons 100 unités de plus pour 9,99 US$ plutôt que de réduire le test.

---

## 7 · Données : MMAD, 8 366 images, 39 670 questions

### Contenu visuel
- Figure : Jiang et al., MMAD (ICLR 2025), fig. 1 (normal / anormal, 7 sous-tâches, 38 catégories)
- Tableau (normales / défectueuses / licence) : MVTec AD **465 / 1 226** CC BY-NC-SA 4.0 · VisA **943 / 1 198** CC BY 4.0 · MVTec LOCO **573 / 993** CC BY-NC-SA 4.0 · GoodsAD **1 309 / 1 659** usage recherche
- Split (graine fixe, stratifié) : **dev 10 % (~840 images)** pour les prompts et les poids · **test ~2 000 images** tirées des 90 % restants, exécuté une fois avec la configuration gelée
- Évaluation uniquement, aucun entraînement. Révision corrigée du 22 octobre 2026

### Script oral — Peter (~60 s)
Nos données, c'est MMAD, un benchmark public publié à ICLR 2025 : 8 366 images et 39 670 questions à choix multiples, construites sur quatre jeux de données industriels, MVTec AD, VisA, MVTec LOCO et GoodsAD, qui couvrent 38 catégories de produits. Tout est déjà annoté : nous n'annotons rien, et les étiquettes servent uniquement à noter. MMAD est un jeu de test uniquement, donc nous faisons notre propre découpage, avec une graine fixe, stratifié par source et par catégorie. Dix pour cent, environ 840 images, forment notre split de dev, où nous avons le droit de regarder les réponses pour choisir les prompts et les poids. Dans les quatre-vingt-dix pour cent restants, nous tirons environ 2 000 images de test, exécutées une seule fois, après le gel de la configuration le 19 novembre. Comme nous réglons sur une partie de MMAD, nous ne revendiquons pas de place dans la piste stricte du leaderboard ; nous comparons toutes les baselines sur le même split.

### À ajouter / améliorer
- Ajouter les totaux : **3 290 normales / 5 076 défectueuses**, 38 catégories.
- Ajouter une ligne honnête : « Dev comes from MMAD → not eligible for the strict main track (declared) ».
- Mentionner que le SHA-256 de `mmad.json` a été vérifié par rapport au fichier des auteurs.
- Licences : le non-commercial convient à un projet de cours, mais interdit un usage commercial direct.

### Question du jury
**« Votre test est un échantillon de 2 000 images. Comment vous comparer au leaderboard ? »**
Pas exactement, et nous le disons. La comparaison qui compte est interne : toutes les baselines sont relancées sur les mêmes images et les mêmes questions. Les chiffres du leaderboard ne servent que de contexte. L'échantillon est stratifié par source, catégorie et normal/défectueux, avec des intervalles bootstrap à 95 %. Si le débit le permet, nous l'étendons aux 90 % complets, soit environ 7 530 images et 35 700 questions.

---

## 8 · Données : ce que voit le système

### Contenu visuel
- 4 vraies paires référence / défaut :
  - MVTec AD : bouteille, **grande cassure sur le goulot**
  - VisA : bougies, **petit morceau manquant**, en haut à gauche
  - MVTec LOCO : défaut **logique**, l'étiquette manque
  - GoodsAD : canette, **rayure en bas**
- Encadré : l'unique référence (`random_templates[0]`) peut être **une autre variante du produit** (un autre jus, une autre marque de canette). Le système doit distinguer variante et défaut.

### Script oral — Peter (~55 s)
Ce sont de vraies paires MMAD, exactement telles que le protocole strict les fournit : la référence est toujours la première image normale de référence de la catégorie. Les défauts vont de l'évident, une bouteille cassée, au minuscule, un éclat sur l'une des quatre bougies, jusqu'au logique : dans MVTec LOCO, l'étiquette qui devrait être là est tout simplement absente, ce qui ne ressemble pas du tout à une tache. Et voici un piège que nous avons découvert en préparant ces slides : dans LOCO et GoodsAD, l'unique référence peut être une autre variante du produit, un autre jus, une autre marque de canette. Une comparaison naïve des patchs signale alors tout l'objet. C'est exactement pour ça qu'il faut des lecteurs et un vérificateur, et que nous publions les résultats par source.

### À ajouter / améliorer
- Entourer le défaut en rouge sur chaque image « Defect ».
- Étiqueter chaque paire avec son type de défaut : structurel / minuscule / logique / variante.
- Marquer les références de LOCO et GoodsAD avec **« different variant! »**.

### Question du jury
**« Si la référence est une autre variante, le cadre 1-shot n'est-il pas injuste pour GoodsAD ? »**
C'est la règle standard du benchmark, et elle est réaliste : sur une ligne, la photo de référence n'est pas toujours exactement la même version du produit. Choisir une référence plus proche, ce serait le réglage « similar template », un autre protocole. Nous attendons des gains plus faibles sur cette source et publions les résultats par source. C'est là que la compréhension sémantique des lecteurs doit l'emporter sur la carte de patchs, et l'analyse d'erreurs vérifiera si c'est le cas.

---

## 9 · Données : les questions

### Contenu visuel
- Figure : un exemple de question par sous-tâche (MMAD, fig. 2)
- Tableau (questions / options) : discrimination d'anomalie **8 297** oui/non · classification du défaut **4 688** · localisation du défaut **4 878** · description du défaut **4 710** · analyse du défaut **4 782** · classification de l'objet **3 155** · analyse de l'objet **9 160** (4 options partout sauf pour la discrimination)
- Score = exactitude par sous-tâche, puis **moyenne simple des 7** (exactitude équilibrée pour la discrimination)

### Script oral — Peter (~45 s)
Chaque image est accompagnée de plusieurs de ces questions, déjà corrigées. Il y a sept sous-tâches. La discrimination, « y a-t-il un défaut ? », se répond par oui ou par non. Toutes les autres ont quatre options : le hasard donne donc 50 et 25 %. Les modèles sont bons sur les questions d'objet, au-dessus de 90 % en classification d'objet, et faibles sur les questions de défaut : notre baseline n'obtient que 61 % en localisation. C'est là qu'un expert visuel devrait le plus aider. Le score est l'exactitude par sous-tâche, équilibrée pour la discrimination, puis la moyenne simple des sept. Charles va maintenant expliquer comment nous allons juger tout cela.

### À ajouter / améliorer
- Ajouter une colonne **B1 (Qwen3-VL-8B)** : 68,7 · 63,3 · **61,0** · 68,2 · 80,0 · 92,3 · 84,0, avec la plus faible mise en évidence.
- Ajouter une colonne hasard : 50 / 25.
- Note : « Object analysis = Object Structure + Object Details (script officiel) ».

### Question du jury
**« La discrimination a 8 297 questions mais ne compte que pour 1/7 du score. La moyenne n'est-elle pas trompeuse ? »**
C'est la métrique officielle de MMAD, donc nous la gardons pour pouvoir comparer. Nous publions aussi chaque sous-tâche séparément, ainsi que les deux types d'erreur de discrimination qui comptent en usine : les pièces conformes rejetées et les défauts manqués. Leurs coûts sont très différents, et aucune moyenne seule ne les capture.

---

## 10 · Métriques : ce que le système complet doit battre

### Contenu visuel
- Échelle des baselines :
  - **B0** hasard (plancher)
  - **B1** Qwen3-VL-8B, 1 appel (meilleur modèle seul)
  - **B2** InternVL3-8B · Qwen3-VL-4B (diversité, taille)
  - **B3** B1 × k échantillons, vote (**calcul égal**)
  - **B4** vote B1 + B2 (sans expert)
  - **S** système complet + ablations
- Succès : **S bat B1 et B3** avec un test de **McNemar** (apparié, mêmes questions). Un résultat négatif bien mesuré est aussi valide.
- Cartes : exactitude (par sous-tâche, moyenne des 7, par source) · erreurs d'inspection (pièces conformes rejetées, défauts manqués) · significativité (McNemar + **bootstrap 95 %**) · coût (appels, tokens, s/image)
- Analyse d'erreurs : **100 échecs** de S classés perception / raisonnement / orchestration (**MAST**) / erreur d'annotation

### Script oral — Charles (~90 s)
Merci Peter. Nos baselines sont conçues pour écarter les explications faciles d'un éventuel gain. B0, c'est le hasard : le plancher. B1, le meilleur modèle ouvert seul, Qwen3-VL-8B, un appel par question. B2, InternVL3-8B, notre second lecteur, plus le modèle 4B pour voir l'effet de la taille. B3 est la clé : on donne à B1 exactement le même nombre d'appels que notre système complet, et on prend un vote majoritaire. Si notre système ne bat que B1, ce n'est peut-être qu'un effet de calcul supplémentaire : il doit battre B3. B4 fait voter les deux lecteurs différents, sans expert ni vérificateur, pour isoler la valeur de la diversité des modèles. Ensuite, les ablations retirent un agent à la fois.

Le succès, c'est que S batte B1 et B3 avec un test de McNemar, un test apparié qui compare, question par question, où un système a raison et l'autre tort. Nous ajoutons des intervalles bootstrap à 95 %. Au-delà de l'exactitude, nous rapportons les erreurs qui comptent en usine, les pièces conformes rejetées et les défauts manqués, ainsi que le coût par image. Enfin, nous classerons 100 échecs en erreurs de perception, de raisonnement, d'orchestration selon MAST, ou d'annotation. Un résultat négatif bien mesuré reste un résultat.

### À ajouter / améliorer
- Donner le **k** de B3 en chiffres : les appels MLLM de S par question (2 lecteurs + vérificateur + réouvertures, environ 3–4), mesurés sur le dev.
- Ajouter une **correction de Holm** sur les deux comparaisons principales (S vs B1, S vs B3). C'est un nouvel engagement, à valider avec Peter avant la présentation.
- Indiquer l'effet minimal détectable, **environ 1 point** (voir Q&R de secours n° 3), et préciser que le bootstrap rééchantillonne des **images**.

### Question du jury
**« Avec cinq baselines et cinq ablations, vous faites beaucoup de tests. Et les comparaisons multiples ? »**
L'hypothèse principale est fixée à l'avance et porte sur deux comparaisons, S vs B1 et S vs B3, avec une correction de Holm. B2, B4 et les ablations sont explicatives : nous les publions avec des intervalles, pas comme des conclusions confirmatoires. Et comme la configuration est gelée avant l'unique exécution du test, nous ne pouvons pas aller pêcher des résultats sur le test.

---

## 11 · Résultats : l'objectif est de battre 73,9 % de façon significative

### Contenu visuel
- Exactitude moyenne MMAD, 1-shot (axe à partir de 50 %) : expert humain **86,7** · SiGMA (autre protocole, pointillés) **85,2** · AD-Copilot 7B (entraîné) **82,3** · GPT-4o **74,9** · **Qwen3-VL-8B, notre B1, 73,9** · Qwen2.5-VL-7B 72,2 · InternVL3.5-8B 65,3 · notre système S : *26 nov.*
- Ligne verticale à 73,9 = la barre à franchir
- Préliminaire, 12 nov. : B1, B2 et le modèle 4B sur le dev ; expert visuel avec ses cartes ; débit Colab
- Final, 26 nov. : S contre toutes les baselines sur le test ; plus grand gain attendu en discrimination et en localisation

### Script oral — Charles (~60 s)
Voici où nous en sommes aujourd'hui : pas encore de résultats propres, et la barre à franchir. Notre baseline, Qwen3-VL-8B, est à 73,9 %, environ un point sous GPT-4o. SiGMA est en pointillés à cause de son protocole différent. AD-Copilot, à 82,3, est entraîné sur d'autres données industrielles, ce que nous ne faisons pas. La ligne vide, c'est notre système : elle sera remplie une seule fois, le 26 novembre, à partir du split de test. Pour la livraison II, le 12 novembre, nous montrerons B1 et B2 mesurés sur notre split de dev, et l'expert visuel produisant ses cartes. Notez que 73,9 est un chiffre rapporté ; SiGMA mesure le même modèle à 74,5. Nous re-mesurons donc B1 nous-mêmes, et c'est ce chiffre, pas le leaderboard, qui sert de référence. Peter va conclure avec les références et le planning.

### À ajouter / améliorer
- Note : « 73.9 is reported by the AD-Copilot paper; SiGMA measures 74.5 → we re-measure B1 on our split. »
- Ajouter une ligne pointillée **humains non experts, 78,7**.
- Colorer les barres selon le protocole : 1-shot strict vs autre protocole vs modèle entraîné.
- Pour la livraison II : remplacer la ligne vide par B1 mesuré sur le dev.

### Question du jury
**« Quel gain appelleriez-vous un succès, de façon réaliste ? »**
Un gain statistiquement significatif sur B3 ; avec environ 9 500 questions appariées, un point environ est détectable. Nous attendons le gain surtout en discrimination et en localisation, là où B1 est le plus faible (68,7 et 61,0) et où l'oracle de MMAD montrait une marge de +25 à +28 points. Nous attendons peu sur les questions d'objet, déjà au-dessus de 84 %. Nous ne promettons pas les 85,2 de SiGMA : il utilise jusqu'à 256 références.

---

## 12 · Références

### Contenu visuel
- Orchestration : Adimulam et al. 2026 · Guo et al. 2024 · Anthropic 2024 · Du et al. ICML 2024 · Huang et al. (AgentCoder) · Cemri et al. NeurIPS 2025 (MAST)
- Benchmark et méthodes : Jiang et al. ICLR 2025 (MMAD) · Zhang et al. 2026 (SiGMA) · Chen et al. 2025 (Echo) · Jiang et al. 2026 (AD-Copilot)
- Données et encodeurs : MVTec AD, MVTec LOCO, VisA, GoodsAD · DINOv3 (Siméoni et al. 2025) · Real-IAD

### Script oral — Peter (~15 s)
Nos références sont ici. Deux à retenir : MMAD pour le benchmark, et MAST pour comprendre pourquoi les systèmes multi-agents échouent. Nous citons directement les articles originaux, car quelques attributions de l'article du cours sont erronées.

### À ajouter / améliorer
- Regrouper les références sous les trois rubriques ci-dessus.
- Note listant les attributions corrigées (voir la question du jury).

### Question du jury
**« Vous dites que l'article du cours contient des références erronées. Lesquelles ? »**
Vérifié sur arXiv : [5] Agent Laboratory est de Schmidgall et al., pas de Deng et al. ; [6] AgentCoder est de Huang et al., pas de Luo et al. ; [8] KoMA est de Jiang et al., pas de Zhao et al. ; et [9], arXiv 2402.01680, est la revue de Guo et al. sous un autre titre. C'est pourquoi le rapport cite les originaux.

---

## 13 · Planning : quatre semaines, une configuration gelée

### Contenu visuel
- **29 oct. · maintenant** : I · Plan (état de l'art, données, modèles, workflow, métriques)
- **12 nov.** : II · Préliminaire (harnais, splits, B1 et B2 sur le dev, expert DINOv3 avec cartes, débit Colab)
- **19 nov. · gel** : configuration gelée (lecteurs, vérificateur, agrégateur, poids fixés sur le dev ; tag git `v-frozen`)
- **26 nov.** : III · Final (test exécuté une fois, B3 et ablations, analyse d'erreurs, rapport)
- Peter (GPU local) : expert visuel, harnais d'évaluation, analyse d'erreurs · Charles (Colab) : lecteurs, vérificateur, agrégateur, exécution des baselines · Les deux : prompts sur le dev, rapport et slides

### Script oral — Peter (~45 s)
Quatre semaines, une configuration gelée. La livraison II, le 12 novembre, porte sur l'infrastructure : le harnais qui reproduit le score officiel de MMAD, les splits fixés, B1 et B2 sur le dev, et l'expert visuel qui produit ses cartes. Le 19 novembre, nous gelons chaque prompt, chaque poids et chaque seuil, et nous étiquetons le commit. Seulement ensuite, nous lançons le test, une seule fois, avec le vote à calcul égal et les ablations, et nous terminons par l'analyse d'erreurs le 26 novembre. Le travail suit le matériel : je prends l'expert visuel, le harnais et l'analyse d'erreurs sur mon GPU local ; Charles prend les lecteurs, le vérificateur, l'agrégateur et les baselines sur Colab. Charles ?

### À ajouter / améliorer
- Faire ressortir la semaine de test (**19–23 nov.**) comme fenêtre critique.
- Ajouter un risque par jalon avec sa parade, par exemple débit Colab → échantillon stratifié.

### Question du jury
**« La fenêtre de test ne va que du 19 au 23 novembre. Et si le système complet n'est pas prêt le 19 ? »**
La date de gel ne bouge pas, parce qu'elle protège le test. Un composant qui n'est pas prêt est retiré de S et présenté comme travail futur ; grâce à la conception par ablations, S sans lui reste un système valide et mesurable. Le débit mesuré pendant les premières semaines fixe la taille du test (1 000 à 2 000 images), pour que l'exécution tienne dans la fenêtre.

---

## 14 · Merci : questions ?

### Contenu visuel
- « Thank you! Questions? »
- Code, plan et slides : github.com/RVilchezzz/MULTIAGENT_INSPECTOR
- Peter Roque Vilchez · Charles Maxime Doudy · carte d'anomalie de la bouteille cassée

### Script oral — Charles (~15 s)
En résumé : une seule photo de référence, une équipe de modèles ouverts, aucun entraînement, et un test équitable à calcul égal. Le plan est public sur GitHub, et le code suivra. Merci ; nous sommes prêts pour vos questions.

### À ajouter / améliorer
- Vérifier que le lien du dépôt fonctionne (le nom est différent dans `slides_content.md`).
- Ajouter un QR code vers le dépôt.
- Garder 2 slides de secours prêtes pour les questions : B1 par sous-tâche, et le budget Colab.

### Question du jury
**« Que feriez-vous avec plus de calcul ? »**
Lancer le test sur les 90 % complets, puis sur MMAD entier pour une comparaison directe avec le leaderboard. Utiliser MMAD-BBox pour mesurer la localisation de l'expert avec l'IoU. Et tester l'extension multi-vues sur Real-IAD (5 vues par pièce), qui répond à notre limite de vue unique.

---

## Q&R de secours

**1. « Régler sur un split de dev de 10 % de MMAD, n'est-ce pas exactement ce que vous reprochez à SiGMA ? »**
En partie oui, et nous le disons. Les différences : une seule référence au lieu de jusqu'à 256, aucun texte métier ajouté, un dev et un test disjoints, et une configuration gelée avant une unique exécution du test. Nous ne sommes pas éligibles à la piste principale stricte, et nous présentons nos résultats comme une comparaison interne équitable : toutes les baselines tournent sur le même split, avec les mêmes règles. L'éligibilité stricte imposerait de développer sur des données externes comme MVTec AD 2 ; nous le déclarons comme limite.

**2. « Comment comptez-vous exactement le "calcul égal" ? »**
B3 reçoit le même nombre d'appels MLLM par question que S : 2 lecteurs, le vérificateur et les éventuelles réouvertures, mesurés sur le dev. Nous rapportons aussi les tokens, les secondes par image et la mémoire GPU. Une passe DINOv3 est négligeable face à un modèle 8B qui génère du texte. Si S consomme plus de tokens par appel à cause du recadrage, le tableau des coûts le montre.

**3. « Avez-vous la puissance statistique pour détecter un gain ? »**
Le test compte environ 9 500 questions appariées. Si environ 15 % sont discordantes (une hypothèse que nous mesurerons sur le dev), cela fait environ 1 425 paires, et un gain d'un point donne un z de McNemar ≈ 95 / √1 425 ≈ 2,5 : il est détectable à p < 0,05. Les questions sur une même image sont corrélées, donc le bootstrap rééchantillonne des images, 1 000 fois.

**4. « Le vérificateur est Qwen3-VL-8B, comme le lecteur A. Et le biais d'auto-préférence ? »**
La plupart de ses contrôles sont déterministes : le schéma Pydantic, les règles de cohérence entre questions, et l'accord avec la région de l'expert. Le critique LLM décide seulement s'il faut rouvrir, une fois au plus. L'ablation « sans vérificateur » et la catégorie MAST « vérification incorrecte » mesurent ce risque. Si un biais apparaît, InternVL3 peut servir de critique.

**5. « Pourquoi un workflow fixe et pas un planificateur LLM ? »**
Anthropic conseille de commencer par le workflow le plus simple. MAST montre que les boucles et une coordination mal spécifiée sont des causes d'échec fréquentes. Et un plan fixe par sous-tâche rend les ablations interprétables et les exécutions reproductibles. La seule étape dynamique est une réouverture unique.

**6. « Pourquoi sans entraînement ? Un petit fine-tuning aiderait sûrement. »**
Le scénario visé est un changement de produit, sans temps ni données pour entraîner. Les modèles entraînés en tête (AD-Copilot, 82,3) demandent de grands jeux de données industriels. Rester sans entraînement isole l'effet de l'orchestration. Une éventuelle petite tête de comparaison serait entraînée uniquement sur des données externes et présentée à part.

**7. « Les MLLM ont peut-être vu des images MVTec pendant leur pré-entraînement. »**
Un pré-entraînement à l'échelle du web ne peut pas être audité ; le leaderboard dit la même chose. Cela touche toutes les baselines de la même façon, puisque S utilise les mêmes modèles. Nos conclusions sont relatives : S contre B1 et B3.

**8. « Une vraie pièce est en 3D. Une seule photo ne suffit pas. »**
D'accord. MMAD évalue une image par question, donc nous inspectons ce que voit un poste caméra. Sur une vraie ligne, on résout ce problème avec plusieurs caméras ou un scan 3D comparé au modèle CAO. Le multi-vues avec Real-IAD (5 vues) est notre extension optionnelle.
