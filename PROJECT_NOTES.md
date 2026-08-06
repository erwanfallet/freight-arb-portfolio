# Carnet de méthode

Ce fichier n'est pas la vitrine. La vitrine, c'est `README.md` et le dashboard. Ici on
garde les règles qu'on s'impose et l'historique des décisions, y compris les mauvaises.

## Ce qui a changé le 6 août 2026

L'itération précédente pré-enregistrait trois hypothèses (H1 attention mal allouée,
H2 mesure biaisée, H3 seuil erroné) avec critères de falsification fixés avant toute
donnée. La discipline était bonne. Le format était le problème :

1. **H2 et H3 étaient indéfendables faute de données.** H2 exige le fret réellement payé
   au chargement (donnée de fixture privée), H3 exige des départs physiques par AIS.
   Aucun des deux n'est accessible de l'extérieur. Un critère de falsification qu'on ne
   peut pas exécuter n'est pas de la rigueur, c'est une intention.
2. **H1 était exécutable mais invendable.** Son résultat est une affirmation sur le modèle
   mental de l'interlocuteur (« votre critère fret/valeur alloue mal l'attention »), pas
   un chiffre de P&L. Ce n'est pas ce qui déclenche une réponse.
3. **Cinq chaînes simultanées** diluaient tout. Une chaîne, une identité, un dashboard.

Les modules correspondants sont dans `archive/h1_h2_h3/`, avec leurs tests. Rien n'est
supprimé, rien n'est mis en avant.

**Ce qu'on garde de cette itération, sans réserve :**

- le contrat de données, et notamment le refus codé du forward-fill
- `voyage/` en entier — le TCE et l'indifférence armateur C3\*/C5\* redeviennent utiles
- l'idée qu'un critère doit être fixé avant de voir le résultat

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

- **2026-08-06** — Repositionnement complet. Projet A défini et implémenté (moteur +
  16 golden tests + dashboard 6 sections + pipeline synthétique de bout en bout). H1/H2/H3
  archivés. `data_dictionary.csv` réécrit pour les trois chaînes avec `exchange_code`,
  `bbg_ticker` et `verified`. Aucun ticker Bloomberg écrit : aucun n'a encore été vu.
