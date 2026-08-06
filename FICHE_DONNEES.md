# Fiche données — ce qu'il faut obtenir, pour A, B et C

Objectif de ce document : que tu puisses aller demander un accès, ou fouiller une base que
tu découvres, **sans avoir à me redemander quoi chercher**.

Rappel de la règle : **aucun ticker Bloomberg n'est écrit ici.** Ce qui est écrit, c'est le
**code de contrat de la bourse**, qui est vérifiable publiquement. Sur le terminal, tu
passes du code bourse au ticker Bloomberg via le menu des tables de contrats (`CTM`), puis
`DES` pour confirmer que la série est la bonne, et seulement là tu l'écris dans
`data_dictionary.csv` avec `verified = oui` et la date.

---

## 0. Le format d'entrée, pour que ça marche du premier coup

Tu déposes des CSV dans `data/raw/`. Deux formats acceptés, détectés automatiquement.

**Format long** (préféré) :

```csv
date,ticker,valeur
2024-01-02,SGX_FEF_M1,138.25
2024-01-03,SGX_FEF_M1,137.10
```

**Format large** (un export Bloomberg typique) :

```csv
date,SGX_FEF_M1,SGX_M65F_M1,SGX_C3F_M1,SGX_C5F_M1
2024-01-02,138.25,153.40,20.15,10.05
```

Trois règles :

- **Ne rebouche jamais un trou à la main.** Un jour férié, une cotation suspendue, une
  série morte : le pipeline sait les gérer et il te dira combien de dates ont survécu à
  l'intersection des calendriers. Un trou comblé à la main est invisible et indétectable.
- **Ne convertis pas les unités avant de déposer.** Dépose l'unité native, déclare-la dans
  `data_dictionary.csv`, et laisse le code convertir. Une conversion faite dans Excel ne
  laisse aucune trace.
- **Un ticker = un nom stable.** Le même ticker dans le fichier et dans le dictionnaire.

Ensuite tu remplis `REAL_TICKERS` en tête de `app/pages/1_Iron_Ore_Premium.py` et le
dashboard bascule du mode synthétique aux vraies données.

---

## 1. Projet A — minerai de fer. **Quatre séries, et il démarre.**

C'est le projet le moins exigeant des trois. Rien ici n'a besoin de Platts, ni du Baltic,
ni d'Argus.

| # | Ce qu'il faut | Identification précise | Où | Statut |
|---|---|---|---|---|
| A1 | Prix 62 % Fe CFR Chine, quotidien, USD/dmt | SGX TSI Iron Ore 62 % Fe futures, code **FEF**, contrat le plus proche | SGX, ou Bloomberg via `CTM` | Code très largement utilisé, à confirmer |
| A2 | Prix 65 % Fe CFR Chine, quotidien, USD/dmt | SGX MB Iron Ore 65 % Fe futures, code **M65F** | idem | Existence confirmée |
| A3 | Fret route C3 Tubarão → Qingdao, USD/wmt | FFA Capesize route C3, code SGX **C3F** | SGX ; repli gratuit Barchart racine `KW3` | Confirmé |
| A4 | Fret route C5 Australie-Occidentale → Qingdao, USD/wmt | FFA Capesize route C5, code SGX **C5F** | SGX ; repli gratuit Barchart racine `KWD` | Confirmé |

**Le plus long historique possible**, idéalement 2019 à aujourd'hui : il faut couvrir à la
fois le régime de premium haute teneur large de 2019 et son effondrement de 2022-2024,
sinon la décomposition n'a qu'un seul régime à montrer et le projet perd son intérêt.

**La vérification à faire en premier, avant même de télécharger le reste :** volume et
open interest du **M65F**. Si le contrat est trop peu traité, la série est bruitée, le
premium calculé est un artefact de cotation, et il faut basculer sur un premium 62-58
(FEF contre le contrat SGX 58 % Fe) — moins élégant, mais liquide. Cette bascule coûte
dix minutes dans le code, et deux semaines si on la découvre à la fin.

### Séries de confort, pas bloquantes

| Ce qu'il faut | Pourquoi | Où |
|---|---|---|
| Importations chinoises de minerai par origine, mensuel | La validation de flux de S4 : la part brésilienne précède-t-elle l'écartement de C3 − C5 | Douanes chinoises (GACC), gratuit |
| Capesize 5TC | Repli de niveau si C3/C5 posent problème | Barchart racine `KWC` |
| Humidité par origine | Hypothèse A-H2, pas une série | Spécifications publiques des producteurs |

---

## 2. Projet B — charbon Atlantique. **Deux séries prix gratuites, un point faible.**

| # | Ce qu'il faut | Identification | Où | Statut |
|---|---|---|---|---|
| B1 | API2 CIF ARA, quotidien | ICE Rotterdam Coal futures, racine **ATW** | ICE ; gratuit sur investing.com et TradingView (`ICEEUR-ATW1!`) | Confirmé |
| B2 | API4 FOB Richards Bay, quotidien | ICE Richards Bay Coal futures, racine **AFR** | idem (`ICEEUR-AFR1!`) | Confirmé |
| B3 | Fret C4 Richards Bay → Rotterdam | Route Baltic C4 | **à chercher** | **Point faible.** L'existence d'un futures C4 liquide n'est pas confirmée |
| B4 | Fret C7 Bolivar → Rotterdam | EEX Baltic Capesize C7 futures | EEX | Existence confirmée |
| B5 | Prix EUA | ICE EUA futures | ICE ; gratuit sur investing.com | À confirmer |
| B6 | **TTF gaz Europe** | ICE TTF futures | ICE ; gratuit | **Non négociable, voir ci-dessous** |

**Pourquoi B6 est non négociable.** La thèse du projet B est que l'arb ARA a perdu sa
contrainte en 2022 parce que la cargaison marginale de Richards Bay part vers l'Inde. Mais
2022 est aussi l'année du choc gazier européen, qui a fait exploser la demande de charbon
en Europe. **Sans contrôler par le TTF, le résultat sera attribué au mauvais mécanisme**, et
la première personne compétente qui lit l'email le verra. C'est le vrai risque intellectuel
du projet B, et il se règle avec une série gratuite.

**Ce que le projet B a de neuf, et que personne n'intègre :** le coût ETS maritime européen
depuis 2024 rend le fret vers l'Europe structurellement plus cher que vers l'Inde à
distance égale. Terme récent, chiffrable (part du voyage en zone UE × facteur d'émission ×
prix EUA), et absent des modèles d'arb charbon publics. Ça peut devenir le résultat
principal plutôt qu'un détail.

**Flux :** importations européennes de charbon sud-africain (Eurostat, mensuel, gratuit) et
importations indiennes par origine (statistiques commerciales indiennes).

---

## 3. Projet C — produits transatlantiques. **Deux blocages, à résoudre avant de coder.**

### La vérification numéro un, et elle prend cinq minutes

Le contrat **TC2** existe chez ICE **et** chez CME (« TC2 37k mt »), et Barchart en a
l'historique sous la racine `IT2`. La question décisive :

> **Ce contrat est-il coté en USD par tonne, ou en points Worldscale ?**

- **En USD/t** → le problème Worldscale disparaît, le projet C se fait avec des données
  d'échange.
- **En points WS** → il faut les flat rates pour convertir, et là c'est bloqué : je n'ai
  trouvé aucune source publique gratuite de la table Worldscale. Elle est publiée chaque
  novembre par la Worldscale Association pour l'année suivante, et les ajustements passent
  par des notes d'abonnés.

### Le second blocage

Le prix **EBOB** (Eurobob oxy barges ARA) est une évaluation Argus, sous licence. Des
futures ICE réglés sur cette évaluation existent probablement, mais je n'ai pas vérifié le
code et je ne vais pas l'inventer.

### Les trois issues, par ordre de préférence

| Issue | Ce qu'il faut | Ce que devient le projet |
|---|---|---|
| **C-1** | Flat rates Worldscale + un prix EBOB | Le projet initial, avec son meilleur résultat : le reset de flat rate du 1er janvier déplace le seuil d'arb à points WS constants |
| **C-3** | **Rien de plus que du gratuit.** Bascule sur le distillat transatlantique : ICE Gasoil ARA contre CME ULSD/Heating Oil NY, fret TC14 | Même trade, sens inverse, deux jambes prix liquides et gratuites. Et une meilleure histoire depuis 2022 : l'Europe a perdu le diesel russe, le flux s'est inversé et allongé en tonne-mille |
| **C-2** | Distances, consommation, VLSFO, hire | On ne paie plus le fret, on le **calcule** : `voyage/tce.py` reconstruit le coût MR par le bas, et on le compare à la route cotée. Plus difficile, plus original, et ça donne enfin un usage au module |

Mon avis, si les flat rates ne tombent pas : **C-3 + C-2 combinés**. « Je n'ai pas acheté le
fret, je l'ai calculé » est un argument de crédibilité plus fort qu'un abonnement.

### Ce qui est déjà acquis pour C, gratuitement

| Ce qu'il faut | Où | Statut |
|---|---|---|
| Essence conventionnelle spot NY Harbor, quotidien | EIA | Confirmé, historique long. **Attention** : la série de futures RBOB de l'EIA s'arrête après le 5 avril 2024 |
| Importations US d'essence par pays d'origine, mensuel | EIA | Confirmé — la validation de flux, officielle et gratuite |
| Stocks essence PADD1, hebdomadaire | EIA | Confirmé |
| VLSFO Rotterdam / Singapour | Ship & Bunker, vlsfo.com, OilPriceAPI (palier gratuit) | Prix courants confirmés ; historique long propre à confirmer |

---

## 4. Quelle base débloque quoi

Tu ne sais pas encore à quoi tu as accès. Voilà la grille de lecture, pour que le jour où
on te propose un accès tu saches immédiatement s'il sert.

| Base | Ce que ça débloque | Priorité |
|---|---|---|
| **Bloomberg avec entitlement Baltic** | Les routes spot C3, C5, C4, TC2 en direct, au lieu des FFA. Supprime la base FFA-vs-index (A-H3), donc supprime un caveat du projet A. **Débloque C-1 côté fret.** | Haute |
| **Bloomberg sans entitlement particulier** | Tous les futures : SGX FEF, M65F, C3F, C5F, ICE ATW, AFR, EUA, TTF. **Suffit intégralement pour A et pour B.** | Suffisant pour démarrer |
| **Platts / S&P Global Commodity Insights** | IODEX 62 et 65 en évaluation spot, les primes aluminium, et les flat rates Worldscale ajustés | Moyenne — les futures SGX couvrent déjà A |
| **Argus Direct** | EBOB, et les indices API2/API4 sous-jacents | **Haute pour C-1**, inutile pour A et B |
| **Worldscale Association** | La table de flat rates. **Le seul verrou de C-1.** | Haute pour C uniquement |
| **Clarksons SIN** | Taux de time charter, distances, spécifications de flotte. **C'est ce qui rendrait C-2 solide** au lieu d'approximatif | Haute si on part sur C-2 |
| **Kpler ou Vortexa** | Flux navire par navire, en temps quasi réel. Remplacerait toutes les séries douanières mensuelles par du quotidien, et **ferait sauter la limite de résolution déclarée en S4** | Le plus gros gain des trois projets |
| **CEIC ou Wind** | Douanes chinoises proprement historisées, sans scraping | Confort réel sur A |
| **LSEG / Refinitiv Eikon, Datastream** | Couverture large, souvent des futures et des indices sans entitlement séparé | Bon substitut à Bloomberg |
| **Barchart / TradingView / investing.com, gratuit** | Historique quotidien des futures listés, sur quelques années | **Déjà suffisant pour prototyper A et B** |

### Si on te demande « de quoi tu as besoin exactement ? »

Réponse courte à donner telle quelle :

> Des séries de prix de clôture quotidiennes, sur le plus long historique disponible, pour
> quatre contrats à terme : SGX minerai de fer 62 % Fe (FEF) et 65 % Fe (M65F), et les FFA
> Capesize routes C3 et C5. En complément, les futures charbon ICE API2 (ATW) et API4 (AFR),
> plus TTF et EUA. En CSV, unités natives, sans retraitement.

C'est une demande banale, non confidentielle, et qui ne réclame aucun entitlement exotique.

---

## 5. Les trois vérifications à faire dès que tu as un terminal

1. **`BALTIC` répond-il ?** Si oui, tu as les routes spot, le projet C reprend sa forme
   initiale et le projet A perd un caveat.
2. **Le contrat TC2 est-il coté en USD/t ou en points WS ?** Décisif pour le projet C.
3. **Volume et open interest du M65F.** Décisif pour le projet A — c'est la seule chose qui
   peut forcer à changer d'indice.

Aucune de ces trois réponses n'empêche de commencer : le moteur du projet A est écrit, les
16 golden tests sont verts, et le dashboard tourne déjà de bout en bout sur données
synthétiques. Il ne manque que quatre colonnes de chiffres.
