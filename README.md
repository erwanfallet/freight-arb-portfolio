# Arbitrages physiques — le fret comme terme décisif

Un portefeuille, pas trois projets isolés : chaque chaîne ajoutée suit le même gabarit
(`docs/NOUVELLE_CHAINE.md`), reconstruit sa marge depuis les séries brutes, et la
confronte à une **série de flux physiques officielle et gratuite**. La question n'est
jamais seulement « l'arb est-il ouvert », c'est **« la cargaison est-elle effectivement
partie »**.

| | Projet | Secteur | Thèse | Statut |
|---|---|---|---|---|
| **A** | Minerai de fer | Vrac sec — minerais | Le premium 65-62 % Fe est en partie un spread de fret Capesize | Moteur + dashboard prêts, en attente des 4 séries |
| **B** | Charbon Atlantique | Vrac sec — énergie | L'arb API2 − API4 a perdu sa contrainte contraignante en 2022 | Moteur + dashboard prêts, en attente des séries |
| **C** | Distillat transatlantique | Tankers — produits raffinés | Le volume n'est pas la masse, et les points Worldscale ne sont pas un coût | Moteur + dashboard prêts, variantes C-2/C-3 codées |
| **D** | À définir | Vrac sec — agriculture | — | Secteur identifié, rien codé |
| **E** | À définir | Gaz (LNG) | — | Secteur identifié, rien codé |

Cette table est un résumé pour humains — la source de vérité est
**`src/freight/portfolio.py`**, que lit `app/Home.py`. Ajouter un projet là-bas suffit à
le faire apparaître sur la plateforme ; voir **`docs/NOUVELLE_CHAINE.md`** pour le gabarit
exact des sept fichiers que touche une nouvelle chaîne.

**`DEMANDE_DONNEES.md`** est le document à ouvrir devant le terminal : la liste des séries
par projet et par priorité, les fichiers exacts à produire, les entitlements à tester, et un
message prêt à envoyer. `FICHE_DONNEES.md` en est la version longue et raisonnée.

---

## Projet A — la thèse en cinq lignes

Les indices minerai 62 % Fe et 65 % Fe sont tous deux cotés **CFR Chine**. Le 65 % est
essentiellement brésilien (Tubarão → Qingdao, ~11 000 nm, route C3), le 62 %
essentiellement australien (Port Hedland → Qingdao, ~1 600 nm, route C5). Le fret est
donc déjà à l'intérieur des deux prix, et le premium haute teneur contient mécaniquement
le différentiel C3 − C5. On décompose, et on regarde combien il reste quand la distance
est payée.

```
premium_observé = P65_CFR − P62_CFR
fair_value_fret = C3/(1 − h_BR) − C5/(1 − h_AU)
résidu          = premium_observé − fair_value_fret
```

Le résidu s'appelle un résidu. Il contient la qualité, la valeur-en-usage, la tension
physique **et** la base FFA-vs-index. Aucune donnée publique ne permet de séparer ces
quatre termes, donc on ne le rebaptise pas « tension physique » pour rendre la conclusion
plus vendable.

### Le piège d'unité, qui est le cœur technique

Les indices minerai sont cotés en USD par tonne métrique **sèche** (dmt). Le fret se paie
sur le poids **embarqué**, donc humide (wmt, poids du connaissement) :

```
fret_par_dmt = fret_par_wmt / (1 − humidité)
```

Fines brésiliennes ~9 % d'humidité, Pilbara Blend ~8 %. Sur l'exemple chiffré du golden
test — P65 120, P62 100, C3 20, C5 10 — la part fret du premium passe de **50,0 %** sans
correction à **55,5 %** avec. Ignorer la conversion sous-estime systématiquement le fret.

C'est le même mouvement que `TC_par_t_zinc = TC_par_dmt_conc / (grade × recovery)` :
l'unité de cotation n'est pas l'unité économique, et faire la conversion proprement est la
moitié du travail.

---

## Projet B — la thèse en cinq lignes

```
arb_ARA = API2 (CIF ARA) − API4 (FOB Richards Bay) − fret(C4) − financement − ETS
```

Le manuel dit que cet arb ne peut pas rester grand ouvert : la concurrence et le fret le
ramènent vers zéro. Depuis 2022, le charbon sud-africain part vers l'Inde plutôt que vers
l'Europe, donc la cargaison marginale de Richards Bay n'est plus cotée sur Rotterdam.
L'équation a perdu son terme contraignant. Le prix CFR Inde étant sous licence, on ne
prouve pas l'égalité en prix : on montre la réorientation en **flux**, et on le présente
comme le résultat plus faible qu'il est.

### Le contrôle qui décide de tout

2022 est aussi l'année du choc gazier européen. Attribuer le décrochage à l'Inde sans
contrôler par le TTF, c'est se tromper de mécanisme. Sur le jeu synthétique — dont la
vraie pente post-rupture est **0,15** par construction — la régression donne :

| | coefficient sur le fret, après rupture |
|---|---|
| sans contrôle | **0,71** — on conclurait que le fret contraint encore |
| avec contrôle TTF | **0,18** — proche de la vérité, le fret ne contraint plus |

Le contrôle n'est pas un raffinement d'économètre : sans lui, la conclusion est inversée.

### Les deux couches techniques

**Pouvoir calorifique.** API2 et API4 sont tous deux des références 6 000 kcal/kg NAR :
l'arb de référence est neutre en CV **par construction** et reste juste. Le problème est
qu'il a cessé de décrire la cargaison physique, dont le CV réel a dérivé vers
~5 700-5 800. Le fret se paie à la tonne, le charbon se vend au kcal : à 5 750 kcal, le
fret par tonne-équivalent-6 000 vaut 1,0435× le fret affiché.

**ETS maritime.** Depuis 2024, un voyage dont une extrémité est hors UE — Richards Bay →
Rotterdam — est couvert à 50 % des émissions, avec une montée en charge de 40 % en 2024,
70 % en 2025 et 100 % à partir de 2026. Couverture effective : 20 %, 35 %, 50 %. Le quota
est coté en EUR et l'arb en USD, donc le change est un terme du calcul.

**Ordre de grandeur, à ne pas survendre :** avec des paramètres réalistes, ce terme vaut
de l'ordre de 0,2 $/t en 2024 et jusqu'à ~0,9 $/t à pleine montée en charge. Sur un arb de
quelques dollars, c'est significatif sans être dominant. C'est un terme que **personne
n'intègre**, ce qui n'est pas la même chose qu'un terme qui décide de tout.

---

## Projet C — la thèse en cinq lignes

```
arb = P_ARA($/t) − P_USGC($/t) − fret($/t) − pont_spec − financement − pertes
      avec  P_USGC($/t) = P_USGC($/gal) × 42 × bbl_par_tonne
      et    fret($/t)   = WS/100 × flat_rate(route, année)
```

Depuis 2022 l'Europe a perdu le diesel russe : le flux transatlantique s'est inversé et
allongé en tonne-mille. Deux termes de cette ligne ne sont pas ce qu'ils paraissent.

### Le volume n'est pas la masse

La jambe américaine se cote en $/gallon, l'européenne en $/tonne. La conversion passe par
une densité, ~7,45 bbl/t. Sur une jambe à ~780 $/t, passer de 7,45 à 7,50 déplace le prix de
**5,25 $/t** — souvent plus que l'arb tout entier. **Le facteur de conversion que tout le
monde traite comme une constante pèse autant que le signal.**

### Les points Worldscale ne sont pas un coût

Le flat rate est réinitialisé chaque 1er janvier. La décomposition est une identité exacte :

```
Δfret = [ ΔWS·FR_prev  +  WS_prev·ΔFR  +  ΔWS·ΔFR ] / 100
          └ marché ┘      └ réglage ┘     └ croisé ┘
```

À points WS constants, WS 150 et un flat rate qui passe de 20 à 24 donnent **+6,00 $/t** de
coût, dont zéro de marché. Sur le jeu synthétique, la plus grosse marche vaut **11,78 $/t**
dont **11,96 de réglage** et **−0,13 de marché**, pour un arb dont la moyenne est de
−1,02 $/t : le réglage pèse dix fois le signal, et il est invisible pour qui regarde les
points.

`signals/worldscale.py` refuse par construction de convertir des points en $/t sans flat
rate daté — il lève plutôt que de se rabattre sur l'année précédente.

### Si les flat rates manquent, le projet tient quand même

- **C-3** : les deux jambes prix restent gratuites et exchange-traded, l'arb se construit.
- **C-2** : on inverse le moteur TCE pour reconstruire le fret depuis l'économie du voyage
  — distances, consommation cubique, soutes, jours de port. `voyage/tce.py` cesse enfin
  d'être de l'infrastructure morte. Un test vérifie que la boucle
  freight → TCE → freight ferme exactement.

---

## Structure

```
src/freight/
  portfolio.py           registre du portefeuille — seule source lue par app/Home.py ;
                        un Project de plus ici suffit à le faire apparaître sur la plateforme
  chains/ironore.py     projet A : humidité, décomposition, variance expliquée,
                        épisodes de résidu négatif, effet de couverture, portage
  chains/coal.py        projet B : arb ARA, couche ETS avec montée en charge et change,
                        base énergétique CV, MCO à contrôles, test de rupture
  chains/products.py    projet C : conversion volume/masse, décomposition Worldscale,
                        illusion des jours ouverts, inversion TCE
  ingest/contract.py    contrat de données — rien n'entre sans contrat rempli
  ingest/loader.py      exports bruts -> format long canonique (date, ticker, valeur)
  ingest/series.py      format long -> séries, + tableau de couverture
  ingest/fixture.py     GÉNÉRATEUR SYNTHÉTIQUE projet A, tickers préfixés SYNTH_
  ingest/fixture_coal.py  idem projet B — la rupture 2022 y est IMPOSÉE à la main
  ingest/fixture_products.py  idem projet C — les marches de flat rate sont IMPOSÉES
  ingest/audit.py       audit de couverture sur data/raw/
  voyage/               TCE, consommation, distances, indifférence armateur C3*/C5*
  backtest/             moteur d'exécution, attribution, sensibilité paramétrique
  signals/worldscale.py conversion points WS -> USD/t (en attente des flat rates)
  events/               gabarit de post-mortem d'événement
app/Home.py             page d'accueil du portefeuille, groupée par secteur, générée
                        depuis src/freight/portfolio.py — jamais éditée à la main
app/pages/               un dashboard Streamlit par chaîne prête
docs/NOUVELLE_CHAINE.md  gabarit des sept fichiers que touche une nouvelle chaîne
tests/                  golden tests, valeurs calculées à la main, aucune donnée requise
scripts/smoke_*.py    pipeline de bout en bout sur données synthétiques
data/raw/               exports bruts, immuables, jamais modifiés à la main
```

## Lancer

```bash
make install   # venv + install editable, moteur + dashboard
make test      # golden tests — aucune donnée nécessaire, à lancer n'importe quand
make smoke     # pipeline complet des deux projets sur les jeux SYNTHÉTIQUES
make app       # dashboard Streamlit
make audit     # une fois data/raw/ rempli et data_dictionary.csv complété
```

## Règles de méthode

- **Un trou de données est une information**, pas du bruit à lisser. Aucun forward-fill
  avant l'étape d'audit. `gap_policy` n'accepte que `none` — le refus est codé, pas
  seulement écrit.
- **Aucune série n'entre dans un calcul sans contrat rempli** : ticker, unité native,
  fréquence, source, plus un drapeau `verified` daté. Les colonnes `exchange_code` et
  `bbg_ticker` sont séparées volontairement : le code de contrat de la bourse est
  vérifiable publiquement, le ticker Bloomberg doit être confirmé sur le terminal via
  `CTM` puis `DES`. **Aucun ticker Bloomberg n'est écrit dans ce repo sans avoir été vu.**
- **L'alignement des calendriers est explicite.** La décomposition travaille sur
  l'intersection des quatre séries et affiche combien de dates ont survécu.
- **Le mode synthétique est signalé en rouge** dans le dashboard, et ses tickers sont
  préfixés `SYNTH_`. Aucun chiffre produit dans ce mode ne doit sortir du repo.

## État des tests

90 tests, tous verts, sans aucune donnée de marché : 16 pour le projet A, 22 pour B, 20 pour
C, le reste pour le socle. Chaque valeur attendue est calculée à la main dans le commentaire
qui la précède. Trois tests portent leur propre argument :

- `test_omitting_the_control_biases_the_freight_coefficient` — pourquoi le TTF est
  obligatoire dans le projet B
- `test_reset_moves_cost_with_zero_market_move` — le résultat Worldscale du projet C, dans
  sa forme la plus nue
- `test_implied_freight_round_trips_through_the_tce_engine` — si cette boucle ne ferme pas,
  toute la variante C-2 s'écroule
