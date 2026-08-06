# Arbitrages physiques — le fret comme terme décisif

Trois chaînes où le fret n'est pas un détail de coût mais le terme qui décide.
Une règle commune : reconstruire la marge depuis les séries brutes, puis la confronter à
une **série de flux physiques officielle et gratuite**. La question n'est jamais seulement
« l'arb est-il ouvert », c'est **« la cargaison est-elle effectivement partie »**.

| | Projet | Thèse | Statut |
|---|---|---|---|
| **A** | Minerai de fer | Le premium 65-62 % Fe est en partie un spread de fret Capesize | Moteur + dashboard prêts, en attente des 4 séries |
| **B** | Charbon Atlantique | L'arb API2 − API4 a perdu sa contrainte contraignante en 2022 | À cadrer après A |
| **C** | Produits transatlantiques | Le fret coté en points Worldscale n'est pas un coût | Bloqué sur les flat rates Worldscale |

Voir `FICHE_DONNEES.md` pour la liste exacte des séries à obtenir, avec pour chacune la
source, le statut de vérification et le repli gratuit.

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

## Structure

```
src/freight/
  chains/ironore.py     projet A : humidité, décomposition, variance expliquée,
                        épisodes de résidu négatif, effet de couverture, portage
  ingest/contract.py    contrat de données — rien n'entre sans contrat rempli
  ingest/loader.py      exports bruts -> format long canonique (date, ticker, valeur)
  ingest/series.py      format long -> séries, + tableau de couverture
  ingest/fixture.py     GÉNÉRATEUR SYNTHÉTIQUE, tickers préfixés SYNTH_
  ingest/audit.py       audit de couverture sur data/raw/
  voyage/               TCE, consommation, distances, indifférence armateur C3*/C5*
  backtest/             moteur d'exécution, attribution, sensibilité paramétrique
  signals/worldscale.py conversion points WS -> USD/t (en attente des flat rates)
  events/               gabarit de post-mortem d'événement
app/                    dashboard Streamlit, une page par chaîne
tests/                  golden tests, valeurs calculées à la main, aucune donnée requise
scripts/smoke_ironore.py  pipeline de bout en bout sur données synthétiques
archive/h1_h2_h3/       itération abandonnée, conservée pour traçabilité — voir son README
data/raw/               exports bruts, immuables, jamais modifiés à la main
```

## Lancer

```bash
make install   # venv + install editable, moteur + dashboard
make test      # golden tests — aucune donnée nécessaire, à lancer n'importe quand
make smoke     # pipeline complet sur le jeu SYNTHÉTIQUE
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

48 tests, tous verts, sans aucune donnée de marché. Les 16 tests de `test_ironore.py`
comparent le moteur à des valeurs calculées à la main dans les commentaires.
