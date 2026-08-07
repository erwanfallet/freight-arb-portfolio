# Demande de données — la liste, les fichiers, les accès

Document opérationnel. À ouvrir devant le terminal, ou à envoyer à qui te donne l'accès.

**État du code :** les trois projets sont écrits, testés et branchés. 90 tests verts, aucune
donnée de marché requise pour les faire tourner. Les trois dashboards s'affichent déjà de
bout en bout sur des jeux synthétiques. **Il ne manque que les colonnes de chiffres.**

---

## 1. En une page

| Priorité | Projet | Séries | Licence nécessaire |
|---|---|---|---|
| **P0** | A — minerai de fer | 4 | **aucune** |
| **P0** | B — charbon | 6 | **aucune** |
| P1 | C — distillat | 4 | aucune pour l'arb, **Worldscale pour le résultat principal** |
| P2 | tous | 4 séries de flux | aucune, tout est public |

Traduction : **avec un terminal Bloomberg standard, sans aucun entitlement particulier, A et
B se font intégralement.** C se fait à 70 %, le reste dépend d'un seul verrou.

---

## 2. Trois vérifications à faire avant de télécharger quoi que ce soit

Chacune prend cinq minutes et peut changer le plan.

| # | Vérification | Pourquoi c'est décisif |
|---|---|---|
| **V1** | Volume et open interest du contrat **SGX M65F** (minerai 65 % Fe) | Si le contrat est trop peu traité, le premium 65-62 calculé est un artefact de cotation et non un prix de marché. Bascule alors sur un premium **62-58** (FEF contre le contrat SGX 58 % Fe) : dix minutes de code maintenant, deux semaines perdues si on le découvre à la fin |
| **V2** | Le contrat **TC2 / TC14** est-il coté en **USD par tonne** ou en **points Worldscale** ? | En USD/t : le verrou Worldscale du projet C disparaît. En points WS : il faut la table de flat rates, et c'est le seul vrai blocage des trois projets |
| **V3** | La fonction **`BALTIC`** répond-elle sur le terminal ? | Si oui : routes spot C3, C5, C4, TC14 en direct au lieu des FFA. Le projet A perd un caveat (la base FFA-vs-index) et le projet B trouve son fret C4 |

Note les réponses ici :

```
V1 — M65F volume / OI :
V2 — TC2 coté en :
V3 — BALTIC répond :
```

---

## 3. Les fichiers à produire

Un fichier par projet, déposé dans `data/raw/`. **Format large accepté** — c'est le format
naturel d'un export Bloomberg.

```
data/raw/ironore.csv
data/raw/coal.csv
data/raw/products.csv
```

Contenu attendu : une colonne `date`, puis une colonne par série, nommée avec le ticker
exact utilisé.

```csv
date,SGX_FEF_M1,SGX_M65F_M1,SGX_C3F_M1,SGX_C5F_M1
2024-01-02,138.25,153.40,20.15,10.05
2024-01-03,137.10,152.05,20.40,10.10
```

### Trois règles, et elles comptent plus que la vitesse

1. **Ne rebouche aucun trou.** Jour férié, cotation suspendue, série morte : le pipeline
   sait les gérer et t'annonce combien de dates survivent à l'intersection des calendriers.
   Un trou comblé à la main est indétectable ensuite.
2. **N'applique aucune conversion d'unité.** Dépose l'unité native et déclare-la dans
   `data_dictionary.csv`. Une conversion faite dans Excel ne laisse aucune trace, et les
   trois projets reposent précisément sur des conversions d'unité faites correctement.
3. **Le plus long historique disponible.** Idéalement 2019 pour A et B, 2021 pour C. Sans
   plusieurs régimes, il n'y a rien à montrer.

Ensuite : remplir `data_dictionary.csv` (colonnes `exchange_code`, `bbg_ticker`, `verified`,
`verified_date`) puis `REAL_TICKERS` en tête de la page Streamlit correspondante.

---

## 4. Projet A — minerai de fer. **4 séries, aucune licence.**

Fichier : `data/raw/ironore.csv`

| # | Série | Code bourse | Unité | Fréq. | Prio |
|---|---|---|---|---|---|
| A1 | Minerai 62 % Fe CFR Chine | SGX **FEF**, contrat le plus proche | USD/dmt | quotidien | P0 |
| A2 | Minerai 65 % Fe CFR Chine | SGX **M65F** | USD/dmt | quotidien | P0 |
| A3 | Fret route C3 Tubarão → Qingdao | FFA SGX **C3F** | USD/wmt | quotidien | P0 |
| A4 | Fret route C5 Australie-Occ. → Qingdao | FFA SGX **C5F** | USD/wmt | quotidien | P0 |
| A5 | Capesize 5TC (repli) | SGX **CWF** | USD/jour | quotidien | P2 |

**Attention aux unités, c'est le cœur du projet :** A1 et A2 sont en tonne **sèche** (dmt),
A3 et A4 en tonne **humide** (wmt). Ne les harmonise surtout pas — la conversion est
justement ce que le code fait, et l'ignorer est l'erreur que le projet démontre.

Repli gratuit si le terminal manque : Barchart, racines `KW3` (C3F), `KWD` (C5F),
`KWC` (5TC).

---

## 5. Projet B — charbon Atlantique. **6 séries, aucune licence.**

Fichier : `data/raw/coal.csv`

| # | Série | Code bourse | Unité | Fréq. | Prio |
|---|---|---|---|---|---|
| B1 | API2 CIF ARA | ICE **ATW** | USD/t | quotidien | P0 |
| B2 | API4 FOB Richards Bay | ICE **AFR** | USD/t | quotidien | P0 |
| B3 | Fret C4 Richards Bay → Rotterdam | route Baltic C4 | USD/t | quotidien | P0 |
| B4 | **TTF gaz Europe** | ICE TTF | EUR/MWh | quotidien | **P0** |
| B5 | Prix EUA | ICE EUA | EUR/tCO2 | quotidien | P1 |
| B6 | EURUSD | n'importe quelle source | — | quotidien | P1 |
| B7 | Fret C7 Bolivar → Rotterdam | EEX C7 | USD/t | quotidien | P2 |

**B4 est P0 et ce n'est pas négociable.** La thèse du projet est que l'arb ARA a perdu sa
contrainte en 2022 à cause de la réorientation vers l'Inde. Mais 2022 est aussi le choc
gazier européen. Sur le jeu synthétique, où je connais la vraie pente (0,15), la régression
**sans** contrôle TTF donne 0,71 et **avec** contrôle 0,18 : sans le TTF, la conclusion est
inversée. Une série gratuite décide de la validité du projet.

**B3 est le point faible connu.** L'existence d'un futures C4 liquide n'est pas confirmée.
Si C4 n'est pas disponible : repli sur le Capesize 5TC plus un ratio de route, ce qui
dégrade la précision du niveau de fret et doit être déclaré dans le dashboard.

**B5-B6 servent la couche ETS.** Le quota est en EUR et l'arb en USD, donc le change est un
terme du calcul. Ordre de grandeur du terme ETS : ~0,2 $/t en 2024, ~0,9 $/t à pleine montée
en charge. Réel, à ne pas survendre.

---

## 6. Projet C — distillat transatlantique. **4 séries, plus un verrou.**

Fichier : `data/raw/products.csv`

| # | Série | Source / code | Unité | Fréq. | Prio |
|---|---|---|---|---|---|
| C1 | ULSD spot US Gulf Coast | **EIA**, gratuit, historique long | USD/gal | quotidien | P0 |
| C2 | Gasoil ARA | ICE Low Sulphur Gasoil futures | USD/t | quotidien | P0 |
| C3 | Fret route TC14 USGC → Continent | ICE / CME, racine Barchart `IT2` pour TC2 | points WS **ou** USD/t | quotidien | P0 |
| C4 | **Flat rates Worldscale** de la route | Worldscale Association | USD/t à WS100 | annuel | **le verrou** |
| C5 | VLSFO Rotterdam ou Houston | Ship & Bunker, vlsfo.com | USD/t | quotidien | P1 |
| C6 | TCE moyen MR | Clarksons, ou courtier | USD/jour | quotidien | P1 |

**C1 et C2 sont en unités différentes, et c'est le sujet.** La jambe américaine est en
$/gallon, l'européenne en $/tonne. La conversion passe par une densité (~7,45 bbl/t). Sur
une jambe à ~780 $/t, passer de 7,45 à 7,50 déplace le prix de **~5 $/t** — souvent plus que
l'arb tout entier. **Le facteur de conversion que tout le monde traite comme une constante
pèse autant que le signal.** C'est le résultat central du projet C.

**C4, le verrou.** Le fret tanker se cote en points WS, et `$/t = WS/100 × flat_rate`, le
flat rate étant réinitialisé chaque 1er janvier. Sur le jeu synthétique, la plus grosse
marche déplace le coût de **11,78 $/t** dont **11,96 attribuables au seul réglage** et
**−0,13 au marché** — pour un arb dont la moyenne est de −1,02 $/t. Le réglage pèse dix fois
le signal, et il est invisible pour qui regarde les points WS.

**Si C4 est indisponible**, deux issues déjà codées :

- **C-3** : l'arb se construit quand même avec C1 à C3. Seul le résultat Worldscale tombe.
- **C-2** : on ne paie pas le fret, on le **calcule**. C5 et C6 suffisent : le moteur TCE est
  inversé pour remonter au taux de fret qu'implique un TCE de marché, et on le compare au
  fret coté là où on a un point de comparaison. « Je n'ai pas acheté le fret, je l'ai
  reconstruit » est un argument plus fort qu'un abonnement.

Ce qu'il faut demander pour C4, en pratique : **la table de flat rates de la route,
année par année, même seulement pour les années passées.** Une valeur par an suffit.

---

## 7. Séries de flux — P2, gratuites, mais c'est le fil rouge du portefeuille

Ce qui distingue ces trois projets : chacun confronte la marge à une série de flux
physiques officielle. La question n'est pas seulement « l'arb est-il ouvert », c'est
**« la cargaison est-elle partie »**.

| Projet | Série | Source | Fréq. |
|---|---|---|---|
| A | Importations chinoises de minerai par origine | Douanes chinoises (GACC) | mensuel |
| B | Importations européennes de charbon sud-africain | Eurostat | mensuel |
| B | Importations indiennes de charbon par origine | Statistiques commerciales indiennes | mensuel |
| C | Exportations américaines de distillat par destination | EIA | mensuel |

Toutes gratuites. La mensualité est une limite de résolution **déclarée à l'avance** dans
chaque projet, pas une surprise à découvrir après coup.

---

## 8. Entitlements Bloomberg — ce qu'il faut tester, et ce que ça change

| Entitlement | Ce que ça débloque | Verdict |
|---|---|---|
| **Terminal standard, sans extra** | Tous les futures : SGX FEF, M65F, C3F, C5F, CWF ; ICE ATW, AFR, TTF, EUA, Gasoil | **Suffit pour A et B en entier** |
| **Baltic** (`BALTIC`) | Routes spot C3, C5, C4, TC14. Supprime la base FFA-vs-index du projet A, résout le fret C4 du projet B | Fort confort, pas bloquant |
| **Worldscale** | La table de flat rates | **Le seul verrou réel**, et il ne concerne que C |
| **Platts** | IODEX spot, primes | Inutile : les futures SGX couvrent A |
| **Argus** | EBOB, indices API sous-jacents | Inutile dans la version distillat de C |
| **Clarksons SIN** | TCE MR, distances, spécifications de flotte | Rendrait C-2 solide plutôt qu'approximatif |
| **Kpler / Vortexa** | Flux navire par navire, quotidien | Le plus gros gain possible : ferait sauter la limite de résolution mensuelle des trois projets |

**Procédure pour chaque ticker :** partir du code de contrat de la bourse, passer par le
menu des tables de contrats (`CTM`), confirmer avec `DES` que la série est bien celle
attendue, puis seulement écrire le ticker dans `data_dictionary.csv` avec
`verified = oui` et la date. **Aucun ticker Bloomberg n'est écrit dans ce repo sans avoir
été vu sur un écran.**

---

## 9. Message prêt à copier-coller

> Bonjour,
>
> J'ai besoin d'un export d'historiques de prix de clôture quotidiens, sur le plus long
> historique disponible, en CSV et **dans les unités natives, sans retraitement**.
>
> **Minerai de fer et fret sec (SGX)** : futures 62 % Fe (FEF), futures 65 % Fe (M65F), FFA
> Capesize routes C3 et C5, et le Capesize 5TC.
>
> **Charbon et énergie (ICE)** : API2 Rotterdam (ATW), API4 Richards Bay (AFR), TTF, EUA, et
> le fret route Baltic C4 si vous y avez accès.
>
> **Produits pétroliers** : ICE Low Sulphur Gasoil, et le fret route TC14 — en précisant si
> la cotation est en points Worldscale ou en USD par tonne.
>
> Deux questions annexes : est-ce que la fonction BALTIC est accessible sur le terminal, et
> est-ce que la table de flat rates Worldscale est disponible, même seulement pour les
> années passées ?
>
> Merci beaucoup.

Rien là-dedans n'est confidentiel ni exotique : ce sont des contrats à terme listés.

---

## 10. Que faire si une série manque

| Manque | Repli | Coût |
|---|---|---|
| M65F illiquide | Premium 62-58 avec le contrat SGX 58 % Fe | Moins élégant, mais liquide |
| C3F ou C5F | Capesize 5TC + ratio de route | Perte de précision sur le niveau, à déclarer |
| C4 | Capesize 5TC + ratio de route | Idem |
| Flat rates Worldscale | Variantes C-3 et C-2, déjà codées | Le résultat Worldscale tombe, le projet tient |
| TCE MR | Estimation depuis les soutes et un taux d'affrètement public | C-2 devient indicatif |
| Séries de flux | Rien — les sections de validation restent en attente | Les projets tournent, la validation manque |

**Aucun de ces replis n'empêche de livrer les trois projets.** Le seul résultat réellement
perdu en l'absence de licence est celui du reset Worldscale.
