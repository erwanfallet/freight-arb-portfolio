# Ajouter une nouvelle chaîne au portefeuille

Gabarit suivi par A (minerai de fer), B (charbon), C (distillat). Une nouvelle chaîne —
agriculture, LNG, ou autre — touche toujours les sept mêmes fichiers, dans cet ordre.

## 0. Avant d'écrire une ligne de code

Trouve le **piège d'unité** — c'est le sujet réel de chaque projet de ce portefeuille,
pas un détail technique en plus :

| Projet | Unité de cotation | Unité économique |
|---|---|---|
| A | tonne sèche (dmt) | tonne humide (wmt), celle que le fret paie |
| B | tonne, référence 6000 kcal | kcal réel de la cargaison |
| C | gallon (jambe US) | tonne (jambe EU), via une densité |

Sans piège de cette nature, il n'y a pas de projet — juste un arb qu'un praticien connaît
déjà. Vérifie aussi qu'une **série de flux physiques officielle et gratuite** existe pour
valider la thèse (douanes, EIA, Eurostat, GACC...) : c'est ce qui distingue "l'arb a l'air
ouvert" de "la cargaison est partie", le fil rouge du portefeuille.

## 1. `src/freight/chains/<id>.py` — le moteur

Docstring de tête obligatoire, structurée comme dans `ironore.py`/`coal.py`/`products.py` :
- `THÈSE` — l'affirmation en 3-5 lignes, avec l'identité mathématique centrale
- le piège d'unité, explicité
- `HYPOTHÈSES` — chaque paramètre nommé `<LETTRE>-H<n>` (ex. `D-H1`), avec sa valeur par
  défaut et, si le biais a un signe connu, lequel

Fonctions pures, testables sans I/O : décomposition, test décisif, sensibilités.

## 2. `src/freight/ingest/fixture_<id>.py` — le générateur synthétique

Un jeu de données fabriqué à la main qui **impose** le phénomène que la thèse prédit (pas
juste du bruit plausible) — comme la rupture 2022 du projet B. Tickers préfixés `SYNTH_`.
C'est ce qui permet au dashboard de tourner avant d'avoir la moindre donnée réelle.

## 3. `tests/test_<id>.py` — les golden tests

Chaque valeur attendue calculée à la main dans le commentaire qui la précède, pas généré
par le code lui-même relu après coup. Au moins un test qui porte l'argument central du
projet (voir les trois exemples cités dans `README.md`, section "État des tests").

## 4. `app/pages/N_<Nom>.py` — le dashboard

Numérotation qui suit l'ordre d'apparition dans `src/freight/portfolio.py`. Six sections,
gabarit S1-S6 des trois pages existantes : état actuel, décomposition terme par terme, le
test qui décide, la validation de flux, un angle secondaire, sensibilités. `REAL_TICKERS`
vide en tête de fichier — tant qu'il est vide, la page tourne en synthétique.

## 5. `scripts/smoke_<id>.py`

Fait tourner le pipeline complet sur le jeu synthétique, sans Streamlit — sert de
vérification rapide et de `make smoke`.

## 6. `data_dictionary.csv`

Une ligne par série, `verified=non` tant qu'un ticker n'a pas été vu sur un terminal. Pas
de ticker Bloomberg écrit sans avoir été confirmé via `CTM` puis `DES`.

## 7. `src/freight/portfolio.py`

Le seul endroit qui fait apparaître le projet sur la page d'accueil. Ajoute un `Project`
avec `status=STATUS_READY`, le nombre de golden tests, et le chemin du dashboard —
`app/Home.py` n'a besoin d'aucune autre modification.

---

Deux entrées existent déjà dans `portfolio.py` avec `status=STATUS_PLANNED` — agriculture
et LNG — comme rappel de ce qui reste à cadrer avant de coder : la thèse en cinq lignes,
et le piège d'unité qui la rend non triviale.
