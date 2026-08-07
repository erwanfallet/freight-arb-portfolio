# Carnet de méthode

Ce fichier n'est pas la vitrine. La vitrine, c'est `README.md` et le dashboard. Ici on
garde les règles qu'on s'impose et l'historique des décisions, y compris les mauvaises.

## Règles qu'on ne relitige pas

1. **On ne choisit pas la thèse avant d'avoir fait tourner le test.** Le projet A a un
   test décisif — quelle part du premium 65-62 le fret explique-t-il — et ce test se fait
   avant de construire les six sections. Si la part fret est faible et stable, il n'y a
   pas d'histoire, et on le dit.
2. **On ne renomme pas un résidu.** Un résidu contient tout ce qu'on n'a pas modélisé. Le
   nommer honnêtement est plus crédible que de l'appeler « tension physique ».
3. **On ne modélise pas ce dont on n'a pas les données.** Pas de modèle de valeur-en-usage
   sidérurgique. On pose le terme à zéro, on dit que le résidu l'absorbe, et on demande à
   l'interlocuteur ce qu'il en pense — ça devient une question au lieu d'être une faute.
4. **On déclare la limite de résolution avant de faire le test.** Les douanes sont
   mensuelles ; si la boucle de rétroaction de S4 ne conclut pas, c'est une limite de
   granularité annoncée, pas un échec découvert après coup.
5. **On ne publie pas un chiffre issu du mode synthétique.** Jamais.
6. **On préfère un biais dont on connaît le signe.** A-H1 sous-estime la part fret ; comme
   la thèse affirme que cette part est grande, le biais joue contre nous. C'est la bonne
   direction, et il faut le dire dans l'email.

## Ordre d'attaque

1. **Vérifier la liquidité du M65F** avant toute autre chose. Si le contrat 65 % Fe est
   trop peu traité, le premium calculé est un artefact et il faut basculer sur un premium
   62-58, moins élégant mais liquide.
2. **Le test décisif de S3** : part de la variance du premium expliquée par C3 − C5, en
   niveau et en variation. Si l'écart entre les deux lectures est énorme, c'est la
   variation qui gagne.
3. Puis les six sections, puis la validation de flux GACC, puis l'email.

## Journal

- **2026-08-07** — Passage de « trois projets » à « portefeuille » : `src/freight/portfolio.py`
  devient la seule source de vérité (un `Project` par chaîne, `status` ready/planned), et
  `app/Home.py` est réécrit pour la lire au lieu d'un contenu à trois colonnes codé en dur —
  il n'a plus besoin d'être touché quand une chaîne s'ajoute. Deux entrées `STATUS_PLANNED`
  ajoutées (agriculture, LNG) comme rappel du prochain travail de cadrage, pas comme code.
  `docs/NOUVELLE_CHAINE.md` écrit : les sept fichiers que touche toujours une nouvelle
  chaîne, dans l'ordre, avec le rappel que le sujet réel de chaque projet est un piège
  d'unité de cotation, pas juste "un nouvel arb". Vérifié en conditions réelles : app lancée
  en local, les trois dashboards et les deux sections "à construire" s'affichent, 90 tests
  toujours verts.
- **2026-08-06 (3)** — Projet C implémenté : `chains/products.py` (conversion volume/masse,
  décomposition exacte de la variation de fret en parts marché / réglage / croisé, illusion
  des jours ouverts, profil saisonnier, inversion du moteur TCE pour la variante C-2),
  20 golden tests, dashboard 6 sections, pipeline synthétique. `signals/worldscale.py` est
  réutilisé tel quel : il refusait déjà de convertir des points en $/t sans flat rate daté,
  ce qui en fait le module le mieux vieilli du repo.
  Le portefeuille a maintenant sa signature : **les trois projets reposent sur une unité de
  cotation qui n'est pas l'unité économique** — tonne sèche contre tonne humide, kcal contre
  tonne, gallon contre tonne. Ce n'est pas une coïncidence, c'est là que se cachent les
  erreurs que personne ne corrige.
  `DEMANDE_DONNEES.md` écrit : liste par priorité, fichiers à produire, entitlements à
  tester, message prêt à envoyer.
- **2026-08-06 (2)** — Projet B implémenté : `chains/coal.py` (arb ARA, couche ETS avec
  montée en charge et conversion de change, base énergétique CV, MCO à contrôles, test de
  rupture et statistiques de régime), 22 golden tests, dashboard 6 sections, pipeline
  synthétique. Le générateur synthétique de B **impose** la rupture 2022 : c'est écrit en
  tête du module, et la page affiche un avertissement plus dur que celle du projet A.
  Résultat de méthode obtenu au passage : sur le jeu synthétique, dont la vraie pente
  post-rupture est 0,15, la régression sans contrôle donne 0,71 et avec contrôle TTF 0,18.
  Sans le contrôle, on conclurait l'inverse de la vérité. C'est la justification du test de
  biais de variable omise ajouté à la suite.
- **2026-08-06** — Projet A défini et implémenté (moteur + 16 golden tests + dashboard
  6 sections + pipeline synthétique de bout en bout). `data_dictionary.csv` écrit pour les
  trois chaînes avec `exchange_code`, `bbg_ticker` et `verified`. Aucun ticker Bloomberg
  écrit : aucun n'a encore été vu.
