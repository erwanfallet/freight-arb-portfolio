# Itération abandonnée — H1 / H2 / H3

Conservée pour traçabilité. Ne pas remettre dans le paquet sans raison.

## Ce que c'était

Un pré-enregistrement de trois hypothèses avec critères de falsification fixés avant
toute donnée :

- **H1 — attention mal allouée.** Le ratio fret/valeur cargaison serait le mauvais
  critère pour dire où le risque fret compte ; ce qui compterait, c'est la part des
  changements de signe de l'arb attribuables à un mouvement de fret. Critère de mort :
  Spearman ρ ≥ 0,7 entre les deux classements sur 4 chaînes ou plus.
- **H2 — mesure biaisée.** Le fret observé à la décision serait un estimateur biaisé
  (malédiction du vainqueur) du fret réellement payé au chargement, le biais croissant
  avec la largeur de l'arb.
- **H3 — seuil erroné.** La règle « arb > 0 donc on expédie » serait fausse, le vrai seuil
  étant `arb > 0 + valeur d'attente`, croissant avec la volatilité du fret.

## Pourquoi c'est archivé

**H2 et H3 sont indéfendables faute de données.** H2 demande le fret réellement payé au
chargement, c'est-à-dire de la donnée de fixture privée. H3 demande des départs physiques
par AIS. Ni l'un ni l'autre n'est accessible depuis l'extérieur d'une maison de négoce.
Un critère de falsification qu'on ne peut pas exécuter n'est pas de la rigueur.

**H1 est exécutable mais son résultat n'est pas vendable.** Il produit une affirmation sur
le modèle mental du lecteur, pas un chiffre. Ce n'est pas ce à quoi un trader répond.

**Le périmètre était trop large.** Cinq chaînes simultanées, là où chaque projet lisible
en tient une.

## Ce qui reste réutilisable

`switching.py` contient une attribution de changement de signe par contrefactuel : on
rejoue le basculement en figeant tour à tour le prix puis le fret, et on regarde lequel
des deux contrefactuels aurait évité le basculement. La logique est propre et
indépendante de H1. Elle pourrait resservir dans un projet où l'on veut attribuer un
franchissement de seuil d'arb à une jambe plutôt qu'à l'autre.

`ranking.py` dépend de `scipy` — c'est la seule raison pour laquelle `scipy` figurait dans
les dépendances du moteur. Il en a été retiré : `pip install -e ".[archive]"` pour rejouer
ces modules.

## Ce qu'on en a gardé ailleurs

L'idée qu'un critère doit être fixé **avant** de voir le résultat. Elle vit maintenant
dans `PROJECT_NOTES.md`, appliquée à un test qu'on peut réellement exécuter : la part de
la variance du premium 65-62 expliquée par le différentiel de fret.
