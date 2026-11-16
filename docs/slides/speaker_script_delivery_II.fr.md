# Script oral : Delivery II (10 min)

OneRef · IA025 · Unicamp 2026 · Peter Roque Vilchez & Charles Maxime Doudy

Support : `OneRef – Preliminary Results (Delivery II).pdf` (11 slides, en anglais). Le texte des slides cité ci-dessous reste en anglais ; le script oral est en français. Versions : anglais `speaker_script_delivery_II.en.md`, espagnol `speaker_script_delivery_II.es.md`.

## Minutage et intervenants

Environ 1 470 mots à l'oral, soit à peu près 10 min à un débit normal en français (~145 mots/min). Si vous dépassez en répétition, raccourcissez d'abord les slides 4 et 7.

| # | Slide | Intervenant | Durée |
|---|---|---|---|
| 1 | Couverture | Charles | 25 s |
| 2 | Contexte | Charles | 60 s |
| 3 | Ce qu'on a construit | Charles | 60 s |
| 4 | Premiers chiffres réels | Peter | 75 s |
| 5 | Ce qui a marché, ce qui a échoué | Peter | 50 s |
| 6 | Ce qui n'a pas marché | Peter | 60 s |
| 7 | Premier run de bout en bout | Charles | 70 s |
| 8 | Exemples du run | Charles | 55 s |
| 9 | Prévu vs fait | Peter | 40 s |
| 10 | Prochaines étapes | Peter | 45 s |
| 11 | Merci | Charles | 10 s |

Charles (environ 4,7 min) présente le contexte, le code des lecteurs, du vérificateur, de l'agrégateur et de l'orchestrateur, et le run de bout en bout. Peter (environ 4,5 min) présente l'expert visuel DINOv3, les données et les splits, les problèmes rencontrés, le bilan par rapport au plan et les prochaines étapes. Chacun répond aux questions du jury sur sa partie.

## À savoir par cœur

1. **DINOv3 n'est pas meilleur en AUROC.** Sur toutes les images de dev, v3 fait **0,59** et v2 **0,61**. On garde v3 parce que le rôle de l'expert est de *localiser* : son pic tombe sur le défaut **77 %** du temps sur MVTec AD, contre **51 %** pour v2. GoodsAD (296 des 835 images) tire la moyenne globale vers le bas.
2. **La recalibration 0,49 → 0,61 est in-sample.** Elle a été ajustée et mesurée sur dev, donc elle est optimiste. Le dire avant que le jury le fasse.
3. **71,2 % sur 18 images n'est pas un résultat.** L'IC à 95 % est **[54 %, 86 %]**. Le run utilisait Qwen3-VL-4B quantifié pour les deux lecteurs et un vérificateur à règles seules. Ne jamais le comparer à 73,9 %.
4. **La discrimination est à 52,5 % équilibrée, proche du hasard (50 %).** C'est la principale faiblesse pour l'instant, cohérente avec l'AUROC faible de l'expert hors MVTec AD.
5. **20/20 vs 21/21.** La leçon de la slide 6 parle de 20 tests passés sur données factices ; la slide 3 en annonce 21. Le 21ᵉ est le test du loader avec les 9 vrais noms, ajouté après le crash.
6. **Les chiffres tombent juste.** 835 + 7 531 = 8 366 images. 184 appels = 2 lecteurs × 88 questions + 4 réouvertures × 2 lecteurs. « ~3,7 s par question » est par appel de lecteur : (323 + 322) / 176 ≈ 3,7 s. Dev contient 3 937 questions.
7. **73,9 % pour Qwen3-VL-8B est un chiffre rapporté** (papier AD-Copilot ; SiGMA mesure 74,5 %). B1 n'a pas encore été re-mesuré sur notre split : c'est la première tâche de cette semaine.

## Passages de parole

- 3 → 4, Charles : « Peter va maintenant montrer ce que fait l'expert sur les vraies données. »
- 6 → 7, Peter : « Charles va présenter ce premier run de bout en bout. »
- 8 → 9, Charles : « Peter, où en est-on par rapport au plan ? »
- 10 → 11, Peter : « Charles ? »

---

## 1 · Couverture : OneRef

### Contenu visuel
- Bandeau : IA025 · Deep Learning · Unicamp 2026
- Titre : **OneRef**
- Sous-titre : preliminary results: the pipeline runs end to end, and the visual expert is measured on real MMAD data
- Équipe : Peter Roque Vilchez · Charles Maxime Doudy · Delivery II · November 12, 2026
- Images : une grille métallique de MVTec AD (« Part ») et la carte d'anomalie de notre expert, avec le pic rouge sur le défaut

### Script oral — Charles (~25 s, ~65 mots)
Bonjour à tous. Je suis Charles Maxime Doudy et, avec Peter Roque Vilchez, nous présentons la Delivery II de OneRef, notre projet d'IA025. Il y a deux semaines, on vous a montré le plan. Aujourd'hui, on fait le point : le pipeline tourne de bout en bout, et l'expert visuel est mesuré sur les vraies données de MMAD. À droite, une vraie image de dev et la carte de notre expert : la tache rouge, c'est le défaut.

### À ajouter / améliorer
- Ajouter la question de recherche en une ligne : « How much of the multi-agent gain survives with one reference and no tuning? »
- Légender l'image du haut « Part (MVTec AD grid, dev) » pour que le jury sache que ce sont de vraies données et pas une illustration.

### Question du jury
**« Qu'est-ce qui a changé depuis la Delivery I ? »**
Trois choses : tout le workflow existe maintenant en code, l'expert visuel a tourné sur les 835 images de dev avec des scores mesurés par source, et les cinq agents ont tourné ensemble une première fois sur de vraies images. Ce qui n'a pas changé, c'est le protocole : une seule référence, aucun entraînement, un seul passage sur le test.

---

## 2 · Contexte : one good photo, one part, a question

### Contenu visuel
- Référence (bouteille saine) et pièce (bouteille au goulot cassé) → **1 · Visual expert** → **2 · Two readers** → **3 · Verifier** → **4 · Vote**
- Benchmark : MMAD, **8 366 images**, **39 670** questions à choix multiples, **7 sous-tâches**, strict 1-shot
- Baseline à battre : Qwen3-VL-8B seul, **73,9 %**, et un vote majoritaire à coût égal
- Règles : aucun entraînement, modèles **≤ 8B**, réglages sur dev uniquement, test lancé une seule fois
- Pied de page : hypothèse = battre les deux baselines de façon significative (McNemar) ; un résultat négatif bien mesuré est aussi valable

### Script oral — Charles (~60 s, ~140 mots)
Petit rappel du problème. Le système reçoit une photo d'une pièce saine, la photo de la pièce à inspecter, et une question à choix multiples. Il répond en quatre étapes : un expert visuel compare les deux photos, deux lecteurs, qui sont des modèles de langage multimodaux, répondent à la question, un vérificateur les contrôle, et un vote tranche. On évalue sur MMAD : 8 366 images, 39 670 questions, sept sous-tâches, et on reste strictement en one-shot, avec une seule référence par question. La baseline à battre, c'est Qwen3-VL-8B seul, rapporté à 73,9 %, et un vote majoritaire avec le même budget de calcul. Les règles n'ont pas bougé : aucun entraînement, des modèles de 8 milliards de paramètres au maximum, tous les choix faits sur dev, et le test lancé une seule fois. Un résultat négatif bien mesuré reste valable.

### À ajouter / améliorer
- Écrire « 73.9 % (reported) » et ajouter « re-measured on our split », pour que personne ne le prenne pour notre propre mesure.
- Remplacer « strict 1-shot » par « 1 reference image per question » : « 1-shot » peut se lire comme un exemple d'entraînement.

### Question du jury
**« Pourquoi un résultat négatif compterait-il ? »**
Parce que la question est de savoir combien des 85,2 % de SiGMA survivent avec un protocole strict. SiGMA utilise jusqu'à 256 images normales, du texte métier de MMAD et une équipe choisie à partir des réponses de MMAD. Si le gain disparaît avec une seule référence et sans réglage, c'est une mesure utile, à condition de l'accompagner d'un intervalle de confiance et d'une analyse d'erreurs.

---

## 3 · Ce qu'on a construit : the full workflow exists as code

### Contenu visuel
- Cinq cartes : **Visual expert** (DINOv3, distance entre patchs, *ran on all 835 dev*) · **Readers A · B** (Qwen3-VL · InternVL3 via vLLM) · **Verifier** (règles + critique LLM, réouverture ≤ 1) · **Aggregator** (vote pondéré par sous-tâche) · **Orchestrator** (un modèle par passe, cache, reprise) ; les quatre derniers : *ran on real data*
- **21 / 21** tests sans GPU passent (schémas, critique, vote, métriques, reprise)
- **835 / 7 531** images dev / test, stratifiées, graine fixée et committée
- **2,1 Go** à télécharger pour dev au lieu de 28 Go
- Pied de page : JSON validé par Pydantic entre agents, une relance puis erreur ; code sur github.com/RVilchezzz/OneRef

### Script oral — Charles (~60 s, ~145 mots)
Premier message : tout ce qui était dans le plan existe maintenant en code. Cinq composants. L'expert visuel DINOv3. Les lecteurs A et B, Qwen3-VL et InternVL3, servis par vLLM. Le vérificateur, qui combine des règles fixes et une critique par un LLM, et qui peut rouvrir un conflit une seule fois au maximum. L'agrégateur, un vote pondéré par sous-tâche. Et l'orchestrateur, qui charge un seul modèle par passe, sauvegarde chaque sortie sur disque et peut reprendre après un plantage. Les agents n'échangent que du JSON validé par Pydantic : une réponse invalide est relancée une fois, puis comptée comme une erreur. Trois chiffres : 21 tests passent sans GPU ; dev et test sont fixés à 835 et 7 531 images, stratifiées, avec une graine committée ; et on télécharge 2,1 gigaoctets pour dev au lieu de 28, en lisant seulement les fichiers utiles dans les zips distants. Peter va maintenant montrer ce que fait l'expert sur les vraies données.

### À ajouter / améliorer
- « Ran on real data » sur les lecteurs et le vérificateur est vrai mais demande une précision : écrire « ran on real data (4B, 18 images) ». Sinon le jury croira que les lecteurs 8B ont tourné.
- Aligner le nombre de tests avec la slide 6 (21 ici, 20 là-bas) : voir le point 5 de « À savoir par cœur ».

### Question du jury
**« Pourquoi un modèle par passe plutôt qu'un pipeline en temps réel ? »**
Parce que le GPU ne tient qu'un seul modèle 8B : 6 Go en local, et une L4 ou une T4 sur Colab. Chaque agent passe sur tout le split, écrit ses sorties sur disque et libère le GPU avant le suivant. En plus, toutes les baselines réutilisent les mêmes sorties de lecteurs en cache, donc B1 à B4 ne coûtent presque rien de plus.

---

## 4 · Premiers chiffres réels : the expert works on aligned photos only

### Contenu visuel
| Source (dev) | Images | AUROC v2 | AUROC v3 | In mask v2 | In mask v3 |
|---|---|---|---|---|---|
| MVTec AD | 166 | 0,88 | 0,84 | 51 % | **77 %** |
| VisA | 215 | 0,77 | 0,78 | 42 % | 41 % |
| MVTec LOCO | 156 | 0,64 | 0,62 | 37 % | 43 % |
| GoodsAD | 296 | 0,55 | 0,51 | 12 % | 10 % |
| All | 835 | 0,61 | 0,59 | 33 % | 40 % |
- Cartes : « DINOv3 localizes better: 77 % vs 51 % on MVTec AD » et « GoodsAD ≈ chance: the one reference is often another product »
- Pied de page : DINOv2-base et DINOv3 ViT-B/16 gelés, requête vs `random_templates[0]`, dev uniquement, un seul run, **0,35 s** par image sur la RTX 3060 Laptop ; masques utilisés seulement pour noter

### Script oral — Peter (~75 s, ~180 mots)
Merci Charles. L'expert visuel compare la pièce à son unique référence avec des features DINO gelées. Pour chaque patch de la pièce, la distance au patch le plus proche de la référence donne une carte d'anomalie. On a testé deux backbones, DINOv2 et DINOv3, sur les 835 images de dev, avec deux mesures. L'AUROC, c'est la probabilité qu'une pièce défectueuse ait un score plus élevé qu'une pièce normale : 0,5, c'est le hasard. « In mask », c'est la fréquence à laquelle le point le plus chaud de la carte tombe sur le vrai défaut. Sur MVTec AD, où les photos sont alignées, l'expert fonctionne : AUROC au-dessus de 0,84, et DINOv3 place son pic sur le défaut 77 % du temps, contre 51 % pour DINOv2. C'est pour ça qu'on garde DINOv3 : localiser, c'est le travail de l'expert, même si son AUROC global est un peu plus bas, 0,59 contre 0,61. Hors MVTec AD, ça se dégrade : VisA autour de 0,78, LOCO autour de 0,63, et GoodsAD est au niveau du hasard, parce que l'unique référence est souvent un autre produit. Coût : 0,35 seconde par image sur notre portable.

### À ajouter / améliorer
- L'en-tête « Images » se coupe en « Image / s » : élargir la colonne.
- Mettre en gras la meilleure valeur de chaque paire pour que l'œil la trouve tout de suite.
- Ajouter une note sous « All » : la ligne globale est pondérée par les images, et GoodsAD représente 35 % de dev (296 / 835).

### Question du jury
**« Pourquoi garder DINOv3 si son AUROC est plus bas ? »**
L'écart d'AUROC est faible (0,59 contre 0,61, un seul run) et vient surtout de GoodsAD, où les deux sont au niveau du hasard. En localisation, qui est ce que les lecteurs reçoivent comme indice, DINOv3 est nettement meilleur sur MVTec AD (77 % contre 51 %) et sur LOCO (43 % contre 37 %). La détection elle-même est confiée aux lecteurs et au vote, pas au seul score de l'expert.

---

## 5 · Ce qui a marché, ce qui a échoué : reference · part · our DINOv3 map

### Contenu visuel
- À gauche, **✓ Worked: defect found** : grille MVTec AD ; chewing-gum VisA (référence, pièce, carte)
- À droite, **× Failed: reference is another product** : boîte alimentaire GoodsAD (référence verte, pièce rose) ; jus MVTec LOCO (étiquette orange contre étiquette banane)
- Pied de page : vraies images de dev ; licences MMAD (CC BY-NC-SA 4.0, CC BY 4.0, usage recherche)

### Script oral — Peter (~50 s, ~125 mots)
Voilà à quoi ça ressemble. Chaque ligne montre la référence, la pièce et notre carte DINOv3. À gauche, deux réussites : une grille métallique de MVTec AD et un chewing-gum de VisA. La tache rouge tombe pile sur le défaut. À droite, deux échecs avec la même cause. Sur GoodsAD, la référence est une boîte verte et la pièce une boîte rose : la carte allume tout le produit, parce que tout est différent. Sur MVTec LOCO, le jus de référence a une étiquette orange et la pièce une étiquette banane : l'expert voit bien une différence, mais il ne sait pas distinguer une variante de produit d'un défaut. La leçon : avec une seule référence, l'expert ne vaut que ce que vaut la correspondance entre les deux photos.

### À ajouter / améliorer
- Mettre une petite légende au-dessus de chaque colonne : « Reference · Part · Map », répétée des deux côtés.
- Sous chaque cas raté, ajouter le score de l'expert à côté du seuil de décision, pour que le jury voie la fausse alarme en chiffres.

### Question du jury
**« Pourquoi ne pas choisir une meilleure référence, par exemple l'image normale la plus proche ? »**
Ce serait utiliser `similar_templates` ou une galerie d'images normales, exactement ce que fait SiGMA et ce que notre protocole strict interdit. On n'utilise que `random_templates[0]`. Le but est de mesurer ce qui survit avec une référence que le système n'a pas choisie ; une meilleure référence peut faire l'objet d'une variante séparée, clairement étiquetée.

---

## 6 · Ce qui n'a pas marché : four problems found, and what we do about them

### Contenu visuel
| Problem | Evidence | Fix / plan |
|---|---|---|
| Loader never tried on real data | crashed on "Anomaly Detection", the name `mmad.json` uses | fixed + test with the 9 real names |
| Expert threshold off-scale for DINOv3 | distances **0.11–0.31** vs range 0.2–0.8: flag never fires (bal. acc. **0.49**) | recalibrated on dev: **0.49 → 0.61** (in-sample) |
| Single reference ≠ same product | GoodsAD AUROC **0.51**, peak on defect **10 %** | trust the expert less there: weights fixed on dev |
| 8B readers not run yet | Colab not executed; local 4B run: **18 images** only | 50-question cost test, then full dev |
- Leçon : unit tests with fake data passed 20/20 while the real file broke the first line. Run on real data early.

### Script oral — Peter (~60 s, ~145 mots)
On a rencontré quatre problèmes, et voici ce qu'on a fait pour chacun. Un : notre chargeur de données n'avait jamais vu le vrai fichier. Il a planté sur le nom de catégorie « Anomaly Detection », qui est celui qu'utilise vraiment mmad.json. C'est corrigé, avec un test sur les neuf vrais noms. Deux : le seuil de l'expert était prévu pour des distances entre 0,2 et 0,8, mais celles de DINOv3 tombent entre 0,11 et 0,31, donc l'alerte ne se déclenchait jamais : précision équilibrée de 0,49. On l'a recalibré sur dev et on atteint 0,61, mais c'est mesuré sur les mêmes images, donc optimiste. Trois : l'unique référence n'est souvent pas le même produit, comme sur GoodsAD, donc le vote fera moins confiance à l'expert là-dessus. Quatre : les lecteurs 8B n'ont pas encore tourné. La leçon : nos tests unitaires passaient sur des données factices pendant que le vrai fichier cassait dès la première ligne. Charles va présenter ce premier run de bout en bout.

### À ajouter / améliorer
- Écrire « 20/20 at the time » dans la leçon, ou la passer à 21/21, pour qu'elle colle à la slide 3.
- Placer « (in-sample) » juste à côté du 0,61, en orange : c'est la précision honnête que le jury va chercher.

### Question du jury
**« Recalibrer le seuil sur dev, ce n'est pas du surapprentissage ? »**
C'est un ajustement sur dev, et dev sert justement à ça : le protocole interdit de régler sur le test, pas sur dev. Le 0,61 est in-sample, donc on le présente comme une estimation haute. La vraie vérification, c'est le run de test du 19 novembre, avec le seuil gelé, et on y rapportera aussi la précision équilibrée de l'expert.

---

## 7 · Premier run de bout en bout : all five agents ran on real MMAD images

### Contenu visuel
- Flux : Expert **18 s** → Reader A **323 s** → Reader B **322 s** → Verifier : **4 reopened** → Vote
- Précision sur **88 questions** : discrimination (équilibrée) **52,5 %** · classification **62,5 %** · description **62,5 %** · localisation **50,0 %** · analyse **87,5 %** · classification d'objet **100 %** · analyse d'objet **83,3 %** · **moyenne des 7 : 71,2 %**
- Cartes : **0 errors in 184 calls** (tout en JSON valide, ~3,7 s par question) · **Readers agree 87/88** (même modèle deux fois) · **95 % CI [54 %, 86 %]** (18 images : un pipeline qui marche, pas un résultat)
- Pied de page : smoke test sur le portable, RTX 3060 6 Go, 18 images de dev (2 à 4 par source), les deux lecteurs = Qwen3-VL-4B quantifié (Ollama), vérificateur à règles seules ; les lecteurs 8B tournent sur Colab

### Script oral — Charles (~70 s, ~165 mots)
Merci Peter. Voici notre premier run de bout en bout : les cinq agents, sur de vraies images MMAD, sur le GPU de 6 gigaoctets du portable. Dix-huit images de dev, deux à quatre par source, 88 questions. Pour tenir en mémoire, les deux lecteurs étaient le même petit modèle, Qwen3-VL-4B quantifié, et le vérificateur n'utilisait que les règles. En haut : 18 secondes pour l'expert, environ 320 secondes par lecteur, et le vérificateur a rouvert quatre conflits. À gauche, la précision par sous-tâche, avec une moyenne de 71,2 %. La discrimination est à 52,5 % équilibrée, proche du hasard, ce qui colle avec l'AUROC faible de l'expert. À droite, ce qu'on a vraiment appris. Zéro erreur sur 184 appels : chaque réponse était un JSON valide. Les deux lecteurs sont d'accord sur 87 questions sur 88, parce que c'était deux fois le même modèle, et c'est précisément pour ça que le vrai système utilise deux familles différentes. Et l'intervalle de confiance va de 54 à 86 % : c'est un pipeline qui fonctionne, pas un résultat.

### À ajouter / améliorer
- Ajouter le nombre de questions par sous-tâche à côté de chaque score : avec 88 questions, certaines cases en ont moins de dix, et 100 % en classification d'objet ne veut pas dire grand-chose.
- Écrire « ~3.7 s per reader call » plutôt que « per question » (323 + 322 s pour 176 appels).
- Ajouter une ligne pointillée à 50 % à côté du score de discrimination.

### Question du jury
**« Avec 52,5 % en discrimination, l'expert n'empire-t-il pas les choses ? »**
On ne peut pas le dire avec 18 images, et c'est exactement ce que mesure l'ablation « sans expert ». Ce qu'on sait, c'est que l'expert est fiable sur les photos alignées et au niveau du hasard sur GoodsAD, donc son vote sera pondéré par sous-tâche selon sa fiabilité sur dev. S'il dégrade quand même les résultats, l'ablation le montrera et on le rapportera.

---

## 8 · Exemples du run : four real cases, four behaviours

### Contenu visuel
- Chaque carte : référence · pièce · le zoom que l'expert a envoyé aux lecteurs (en orange)
- **✓ Cable** (MVTec AD) : 5/5 justes, « Missing component, in the central part »
- **↻ Tile** (MVTec AD) : vérificateur a rouvert ; lecteurs « bottom-left », expert « bottom-center » ; toujours faux (center-left)
- **× Can** (GoodsAD) : fausse alarme, pièce saine déclarée défectueuse ; la référence est une autre marque
- **× Pushpins** (MVTec LOCO) : a suivi l'indice, « the expert hint indicates the top right » : mauvais endroit
- Takeaway : l'indice de l'expert est puissant dans les deux sens, donc son poids doit dépendre de sa fiabilité sur dev

### Script oral — Charles (~55 s, ~140 mots)
Quatre vrais cas tirés de ce run, et quatre comportements différents. Chaque carte montre la référence, la pièce, et en orange le zoom que l'expert a envoyé aux lecteurs. En haut à gauche, un câble de MVTec AD : cinq questions sur cinq justes, avec la réponse « composant manquant, dans la partie centrale ». En haut à droite, un carrelage : les lecteurs disent en bas à gauche, l'expert dit en bas au centre. Le vérificateur a repéré le conflit et l'a rouvert, mais la réponse finale reste fausse : le défaut est au centre gauche. En bas à gauche, une canette de GoodsAD : fausse alarme, parce que la référence est une autre marque. En bas à droite, des punaises de LOCO : le lecteur écrit « l'indice de l'expert indique en haut à droite » et le suit au mauvais endroit. À retenir : l'indice de l'expert est puissant dans les deux sens, donc son poids doit dépendre de sa fiabilité sur dev. Peter ?

### À ajouter / améliorer
- Sur la carte du carrelage, ajouter la vraie position (« truth: center-left ») dans une autre couleur, pour que l'histoire de la réouverture se lise d'un coup d'œil.
- Sur la carte des punaises, rappeler que les défauts de LOCO sont logiques (mauvais nombre, pièce manquante) : une distance entre patchs ne peut pas les voir.

### Question du jury
**« Le vérificateur a rouvert le carrelage et la réponse est restée fausse. À quoi sert-il ? »**
Son rôle est de détecter un désaccord et d'imposer un second regard, pas de connaître la réponse. Dans ce run, il n'utilisait que les règles et les deux lecteurs étaient le même modèle 4B, donc le second regard n'apportait pas grand-chose de neuf. Avec deux familles de modèles et la critique LLM, l'ablation « sans vérificateur » mesurera ce qu'il apporte ; quatre réouvertures sur 88 questions montrent aussi qu'il ne tourne pas en boucle.

---

## 9 · Prévu vs fait : what the plan promised for Nov 12

### Contenu visuel
| Planned (Delivery I) | Status | Detail |
|---|---|---|
| MMAD download | Done | dev subset, **2.1 GB**, **2,084 files** |
| Dev and test splits fixed | Done | **835 / 7,531** images, committed |
| DINOv3 expert with anomaly maps | Done | all 835 dev images, per-source scores |
| Evaluation harness | Partial | metrics, McNemar, bootstrap coded; check vs official `summary.py` pending |
| B1 and B2 on dev | Pending | readers coded and tested, not run |
| Colab throughput measured | Pending | first task this week |
- **3 / 6** items faits · **+ extra** : tout le workflow multi-agents codé et déjà lancé de bout en bout sur 18 images de dev (prévu pour le 12–19 nov.)

### Script oral — Peter (~40 s, ~100 mots)
Par rapport à ce qu'on avait promis pour aujourd'hui : trois points sur six sont faits. Le téléchargement de dev de MMAD, 2,1 gigaoctets et 2 084 fichiers. Les splits, fixés et committés. Et l'expert DINOv3 sur toutes les images de dev, avec des scores par source. Le harnais d'évaluation est partiel : les métriques, McNemar et le bootstrap sont codés, mais pas encore vérifiés contre le script officiel de MMAD. B1 et B2 sur dev, et le débit sur Colab, sont encore en attente. En revanche, tout le workflow multi-agents, prévu pour la semaine prochaine, est déjà codé et a tourné de bout en bout.

### À ajouter / améliorer
- Écrire « 3 done · 1 partial · 2 pending » au lieu de « 3 / 6 » : c'est plus honnête et ça se lit plus vite.
- Dire pourquoi B1 et B2 ont glissé (Colab pas encore lancé) dans la colonne Detail.

### Question du jury
**« Vous êtes en retard sur les baselines. Le gel du 19 novembre est-il réaliste ? »**
Les lecteurs sont codés et testés ; ce qui manque, c'est du temps GPU, pas du code. Le test de coût sur 50 questions fixe la taille du test : de 1 000 à 2 000 images selon les unités mesurées. Toutes les baselines réutilisent les mêmes sorties de lecteurs en cache, donc B3 et B4 sont quasiment gratuites une fois B1 et B2 lancés.

---

## 10 · Prochaines étapes : same goal, adjusted path to Nov 26

### Contenu visuel
- **Nov 12–15 · Readers on dev** : test de coût sur 50 questions sur Colab L4, puis B1 et B2 sur les **3 937** questions de dev
- **Nov 15–19 · Calibrate on dev** : seuil de l'expert, poids du vote par lecteur et par sous-tâche, prompts ; vérification avec `summary.py` officiel
- **Nov 19 · Freeze · Test, run once** : S contre B1–B4, ablations ; **~1 000–2 000** images de test selon le coût mesuré
- **Nov 26 · Final** : McNemar + bootstrap, analyse d'erreurs par source, rapport
- Ajustement : le vote de l'expert dépendra de sa fiabilité sur dev pour chaque sous-tâche ; résultats toujours rapportés par source (GoodsAD et LOCO à part)

### Script oral — Peter (~45 s, ~120 mots)
L'objectif ne change pas, mais le chemin est ajusté. Jusqu'au 15 : le test de coût sur 50 questions sur une L4 de Colab, puis B1 et B2 sur les 3 937 questions de dev. Du 15 au 19 : la calibration sur dev, c'est-à-dire le seuil de l'expert, les poids du vote par lecteur et par sous-tâche, les prompts, et la vérification avec le script officiel. Le 19, on gèle tout et on lance le test une seule fois : le système complet contre B1 à B4, plus les ablations, sur 1 000 à 2 000 images selon le coût mesuré. Le 26 : McNemar, bootstrap, analyse d'erreurs par source, et le rapport. Un ajustement : le vote de l'expert dépendra de sa fiabilité sur dev. Charles ?

### À ajouter / améliorer
- Ajouter le budget Colab (100 unités, jamais d'A100) et le plan B (+100 unités) sous la première case.
- Marquer le gel comme une ligne dure sur la frise, par exemple « no change after this point ».

### Question du jury
**« Et si le test sur 50 questions montre que Colab coûte trop cher ? »**
Alors le test se réduit vers 1 000 images, ce qui donne encore assez de questions appariées pour McNemar, et on passe sur la T4 en 4 bits si la L4 consomme trop d'unités. Le coût est mesuré avant de fixer la taille du test, justement pour qu'il ne nous force jamais à changer de protocole.

---

## 11 · Merci !

### Contenu visuel
- Bandeau : OneRef · Delivery II
- **Thank you! Questions?**
- Code et résultats : github.com/RVilchezzz/OneRef
- Image : carte DINOv3 sur le chewing-gum de VisA

### Script oral — Charles (~10 s, ~30 mots)
Merci pour votre attention. Le code et les résultats sont publics sur github.com/RVilchezzz/OneRef. Nous sommes prêts à répondre à vos questions.

### À ajouter / améliorer
- Rien à changer. Garder la slide 4 prête pour y revenir : la plupart des questions porteront dessus.

### Question du jury
**« Quel est le chiffre à retenir aujourd'hui ? »**
77 % : sur des photos alignées, le pic de notre expert tombe sur le vrai défaut trois fois sur quatre, à partir d'une seule référence. Tout le reste, y compris la question de savoir si l'équipe complète bat Qwen3-VL-8B, sera mesuré sur le test après le 19 novembre.

---

## Questions de secours

**« Pourquoi ne pas fine-tuner l'expert ou les lecteurs sur MMAD ? »**
Le protocole est sans entraînement par construction : MMAD est notre banc d'essai, et entraîner dessus rendrait la comparaison avec le leaderboard sans valeur. Tout ce qui est entraîné sur MMAD relève d'une autre piste.

**« Un AUROC de 0,59, c'est assez pour être utile ? »**
Pas comme détecteur à lui seul, et on ne l'utilise pas comme ça. Sur MVTec AD, il atteint 0,84 à 0,88 ; au global, il est tiré vers le bas par GoodsAD. Sa vraie valeur est la localisation, transmise aux lecteurs sous forme d'un zoom et d'une région 3×3.

**« Pourquoi GoodsAD est-il si difficile ? »**
Les photos ne sont pas alignées et l'unique référence est souvent un autre produit ou une autre marque, donc une distance entre patchs signale tout l'objet. On le garde et on le rapporte à part plutôt que de le retirer : le retirer gonflerait nos chiffres.

**« Qu'est-ce que l'expert envoie exactement aux lecteurs ? »**
Une région 3×3, un zoom autour du pic et un score. Dans les ablations, on compare aussi une carte brute, une carte calibrée et un indice uniquement textuel.

**« Comment prouver que les lecteurs ne recopient pas simplement l'expert ? »**
Le cas des punaises montre qu'ils le font parfois. L'ablation « sans expert » mesure l'effet net, et l'analyse d'erreurs comptera combien de fois un mauvais indice a mené à une mauvaise réponse, avec les catégories MAST pour les échecs d'orchestration.

**« Pourquoi Ollama en local et vLLM sur Colab ? »**
vLLM ne tourne pas sous Windows et 6 Go ne suffisent pas pour un modèle 8B en bf16. Ollama sert un modèle 4B quantifié via la même API compatible OpenAI, donc le code est identique ; seule l'adresse du serveur change.

**« 18 images, ça suffit pour dire quelque chose ? »**
Seulement que le pipeline fonctionne : zéro erreur de format sur 184 appels, le vérificateur rouvre sans boucler, et les logs sont complets. Les chiffres de précision ne sont pas un résultat, c'est pour ça que la slide montre l'intervalle de confiance [54 %, 86 %].

**« Le test est-il vraiment intact ? »**
Oui. Il est fixé par une graine committée, disjoint de dev, et la CLI refuse `--split test` sans configuration gelée. Il sera lancé une seule fois, après le 19 novembre.
