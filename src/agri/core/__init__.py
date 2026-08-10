"""Socle technique commun aux pages agri — et réutilisable par les pages métaux.

Codé une fois, appelé partout. Contient les conversions d'unités, les règles de
rééchantillonnage qui décident de la validité d'un test, la boîte à outils
statistique, et le solveur de point de bascule qui est le livrable de chaque page.

Rien ici ne connaît une commodité en particulier : `chains/` fait les modèles,
`core/` fait l'arithmétique et la statistique.
"""
