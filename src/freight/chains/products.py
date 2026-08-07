"""Projet C — arb distillat transatlantique, et le fret qui n'est pas un coût.

THÈSE
-----
Acheter du diesel sur le Gulf Coast américain, l'expédier en MR vers Rotterdam, le vendre
contre le gasoil ARA. Depuis 2022 l'Europe a perdu le diesel russe : le flux
transatlantique s'est inversé et allongé en tonne-mille, ce qui en fait le trade produits
le plus regardé de la décennie.

    arb = P_ARA($/t) − P_USGC($/t) − fret($/t) − pont_de_spec − financement

Deux termes de cette ligne ne sont pas ce qu'ils paraissent.

1. LE VOLUME N'EST PAS LA MASSE
-------------------------------
Le diesel américain se cote en **$/gallon**, le gasoil européen en **$/tonne**. La
conversion passe par une densité :

    P($/t) = P($/gal) × 42 gal/bbl × bbl_par_tonne

`bbl_par_tonne` vaut environ 7,45 pour un diesel à 0,845 kg/l. Ce n'est **pas** une
constante universelle : elle dépend de la densité du lot, qui varie avec la spécification
et la température de référence.

Et l'ordre de grandeur est brutal. Sur une jambe à ~780 $/t, passer de 7,45 à 7,50 bbl/t
déplace le prix de **~5 $/t**, c'est-à-dire souvent plus que l'arb tout entier. **Le
facteur de conversion que tout le monde traite comme une constante est plus gros que le
signal.** C'est le résultat central de ce projet.

C'est le même piège que l'humidité sur le minerai de fer et le pouvoir calorifique sur le
charbon : le fret se paie sur une unité, la valeur se mesure sur une autre.

2. LES POINTS WORLDSCALE NE SONT PAS UN COÛT
--------------------------------------------
Le fret tanker se cote en points Worldscale. WS 100 correspond au *flat rate* de la route,
recalculé **chaque 1er janvier** par la Worldscale Association à partir de l'environnement
de coûts de l'année précédente. Donc :

    fret($/t) = WS/100 × flat_rate(route, année)

Conséquence directe : **le coût en $/t saute au 1er janvier même si les points WS n'ont pas
bougé d'un cheveu.** Un modèle qui utilise les points WS comme proxy de coût se trompe de
timing chaque année, et les flat rates ont fortement bougé en 2022-2023 avec les soutes.

La décomposition exacte d'une variation de fret :

    Δfret = [ ΔWS·FR_prev + WS_prev·ΔFR + ΔWS·ΔFR ] / 100
              └ marché ┘   └ réglage ┘   └ croisé ┘

Au 1er janvier, ΔWS peut être nul et Δfret ne pas l'être. Le terme « réglage » est
invisible pour qui regarde les points.

Le module `signals/worldscale.py` refuse par construction de convertir des points en $/t
sans flat rate daté. Ce module s'appuie dessus.

SI LES FLAT RATES SONT INDISPONIBLES
------------------------------------
Deux issues, toutes deux implémentées ici :

- **C-3** : les deux jambes prix restent exchange-traded et gratuites (EIA pour le Gulf
  Coast, ICE Gasoil pour l'ARA), donc l'arb se construit quand même. Seul le résultat
  Worldscale tombe.
- **C-2** : on ne paie pas le fret, on le **calcule**. `implied_freight_from_tce` inverse
  le moteur TCE existant : à partir d'un TCE de marché MR, des distances, des soutes et
  des jours de port, on remonte au taux de fret que ce TCE implique. On compare ensuite au
  fret coté là où on en a. « Je n'ai pas acheté le fret, je l'ai reconstruit » est un
  argument plus fort qu'un abonnement.

HYPOTHÈSES
----------
C-H1  Densité du diesel : 7,45 bbl/t par défaut. **Paramétrée et balayée** — c'est le
      terme le plus sensible du projet, il ne peut pas être un nombre en dur.
C-H2  Pont de spécification USGC ULSD 15 ppm contre ICE Gasoil 10 ppm : traité comme une
      **fourchette explicite**, jamais comme un chiffre unique. Un pont bricolé
      transformerait l'arb en bruit.
C-H3  Pertes en transit et surestaries : paramètre unique, non modélisé finement.
C-H4  Financement de la valeur cargaison sur la durée de traversée.
C-H5  Les flat rates utilisés sont ceux de la route et de l'année de la cotation. Le
      module lève si un couple route/année manque, plutôt que de se rabattre en silence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from freight.signals.worldscale import FlatRateTable

# C-H1 — densité par défaut du distillat, en barils par tonne.
DEFAULT_BBL_PER_TONNE = 7.45
GALLONS_PER_BARREL = 42.0


# ------------------------------------------------------------------ volume vs masse
def usd_per_gallon_to_usd_per_tonne(
    price_per_gallon: pd.Series | float, bbl_per_tonne: float = DEFAULT_BBL_PER_TONNE
) -> pd.Series | float:
    """Convertit un prix en $/gal en $/t (C-H1).

        $/t = $/gal × 42 × bbl_par_tonne

    `bbl_par_tonne` n'est pas une constante physique universelle : c'est une hypothèse de
    densité. Elle est donc un argument, jamais une valeur en dur.
    """
    if bbl_per_tonne <= 0:
        raise ValueError(f"bbl_per_tonne doit être > 0, reçu {bbl_per_tonne}")
    return price_per_gallon * GALLONS_PER_BARREL * bbl_per_tonne


def density_sensitivity(
    price_per_gallon: float, bbl_grid: tuple[float, ...] = (7.35, 7.40, 7.45, 7.50, 7.55)
) -> pd.DataFrame:
    """Tableau de sensibilité du prix converti à l'hypothèse de densité.

    Existe pour rendre visible, en une ligne, que le terme le plus incertain du projet est
    aussi l'un des plus gros.
    """
    base = usd_per_gallon_to_usd_per_tonne(price_per_gallon, DEFAULT_BBL_PER_TONNE)
    rows = []
    for bbl in bbl_grid:
        converted = usd_per_gallon_to_usd_per_tonne(price_per_gallon, bbl)
        rows.append(
            {
                "bbl_par_tonne": bbl,
                "prix ($/t)": round(float(converted), 2),
                "écart au défaut ($/t)": round(float(converted - base), 2),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ Worldscale
def freight_usd_per_tonne(
    ws_points: pd.Series, flat_rates: pd.Series
) -> pd.Series:
    """fret($/t) = WS/100 × flat_rate, les deux séries alignées par date.

    `flat_rates` doit être une série journalière en escalier : la même valeur toute
    l'année civile, une marche au 1er janvier. La construire depuis une
    `FlatRateTable` avec `flat_rate_step_series`.
    """
    aligned = pd.concat({"ws": ws_points, "fr": flat_rates}, axis=1).dropna()
    if aligned.empty:
        raise ValueError("aucune date commune entre les points WS et les flat rates")
    out = aligned["ws"] / 100.0 * aligned["fr"]
    out.name = "freight_usd_t"
    return out


def flat_rate_step_series(
    index: pd.DatetimeIndex, route: str, table: FlatRateTable
) -> pd.Series:
    """Série journalière en escalier des flat rates, depuis la table datée.

    Lève si une année de l'index n'a pas de flat rate — c'est voulu. Un repli silencieux
    sur l'année précédente est exactement l'erreur que ce projet dénonce.
    """
    values = [table.flat_rate(route, d.year) for d in index]
    return pd.Series(values, index=index, name=f"flat_rate_{route}")


def decompose_freight_change(ws_points: pd.Series, flat_rates: pd.Series) -> pd.DataFrame:
    """Décomposition exacte de la variation de fret en $/t.

        Δfret = [ ΔWS·FR_prev + WS_prev·ΔFR + ΔWS·ΔFR ] / 100
                  └ marché ┘   └ réglage ┘   └ croisé ┘

    Les trois termes somment exactement à Δfret : c'est une identité algébrique, pas une
    approximation. Le terme « réglage » est celui qu'un modèle en points WS ne voit pas.
    """
    aligned = pd.concat({"ws": ws_points, "fr": flat_rates}, axis=1).dropna().sort_index()
    if len(aligned) < 2:
        raise ValueError("il faut au moins deux observations pour décomposer une variation")

    ws_prev = aligned["ws"].shift(1)
    fr_prev = aligned["fr"].shift(1)
    d_ws = aligned["ws"].diff()
    d_fr = aligned["fr"].diff()

    out = pd.DataFrame(index=aligned.index)
    out["freight_usd_t"] = aligned["ws"] / 100.0 * aligned["fr"]
    out["d_freight"] = out["freight_usd_t"].diff()
    out["market_component"] = d_ws * fr_prev / 100.0
    out["reset_component"] = ws_prev * d_fr / 100.0
    out["cross_component"] = d_ws * d_fr / 100.0
    return out.dropna()


def january_reset_effect(decomposition: pd.DataFrame) -> pd.DataFrame:
    """Isole les dates où le flat rate change — en pratique le premier jour ouvré de
    chaque année — et chiffre le saut de coût attribuable au seul réglage.

    Retourne une ligne par réinitialisation : date, saut total, part réglage, part
    marché, et la variation de points WS ce jour-là. Quand cette dernière est petite et
    que la part réglage est grande, on a le résultat du projet sous les yeux.
    """
    reset_rows = decomposition[decomposition["reset_component"].abs() > 1e-12]
    if reset_rows.empty:
        return pd.DataFrame(
            columns=["date", "saut_total", "part_reglage", "part_marche", "part_croisee"]
        )
    return pd.DataFrame(
        {
            "date": reset_rows.index,
            "saut_total": reset_rows["d_freight"].round(3).to_numpy(),
            "part_reglage": reset_rows["reset_component"].round(3).to_numpy(),
            "part_marche": reset_rows["market_component"].round(3).to_numpy(),
            "part_croisee": reset_rows["cross_component"].round(3).to_numpy(),
        }
    )


# ------------------------------------------------------------------------------- arb
def reconstruct_arb(
    price_destination_usd_t: pd.Series,
    price_origin_usd_per_gallon: pd.Series,
    freight_usd_t: pd.Series,
    *,
    bbl_per_tonne: float = DEFAULT_BBL_PER_TONNE,
    spec_bridge_usd_t: float = 0.0,
    voyage_days: float = 16.0,
    annual_rate: float = 0.06,
    losses_usd_t: float = 0.0,
) -> pd.DataFrame:
    """Arb USGC -> ARA, terme par terme, avec la conversion de volume explicite.

    Alignement par intersection, aucun forward-fill. Retourne un DataFrame avec
    p_dest, p_origin_gal, p_origin_t, spread_naif, freight, spec, financing, losses,
    arb, is_open.

    `spread_naif` est le spread avant fret et pont de spec : c'est le chiffre que l'on
    regarde quand on croit voir un arb ouvert, et la comparaison avec `arb` est l'objet
    de `open_days_comparison`.
    """
    aligned = pd.concat(
        {
            "p_dest": price_destination_usd_t,
            "p_origin_gal": price_origin_usd_per_gallon,
            "freight": freight_usd_t,
        },
        axis=1,
    ).dropna()
    if aligned.empty:
        raise ValueError(
            "aucune date commune aux trois séries — vérifier les calendriers, "
            "ne pas combler les trous"
        )
    aligned = aligned.sort_index()

    out = aligned.copy()
    out["p_origin_t"] = usd_per_gallon_to_usd_per_tonne(aligned["p_origin_gal"], bbl_per_tonne)
    out["spread_naif"] = out["p_dest"] - out["p_origin_t"]
    out["spec"] = float(spec_bridge_usd_t)
    out["financing"] = out["p_origin_t"] * annual_rate * (voyage_days / 365.0)
    out["losses"] = float(losses_usd_t)
    out["arb"] = (
        out["spread_naif"] - out["freight"] - out["spec"] - out["financing"] - out["losses"]
    )
    out["is_open"] = out["arb"] > 0
    out["looks_open_naif"] = out["spread_naif"] > 0
    return out


@dataclass(frozen=True)
class OpenDaysComparison:
    n_obs: int
    days_looking_open: int
    days_really_open: int

    @property
    def share_looking_open(self) -> float:
        return self.days_looking_open / self.n_obs if self.n_obs else float("nan")

    @property
    def share_really_open(self) -> float:
        return self.days_really_open / self.n_obs if self.n_obs else float("nan")

    @property
    def illusion_share(self) -> float:
        """Part des jours « ouverts en apparence » qui ne survivent pas au coût réel."""
        if self.days_looking_open == 0:
            return float("nan")
        return 1.0 - self.days_really_open / self.days_looking_open


def open_days_comparison(arb_frame: pd.DataFrame) -> OpenDaysComparison:
    """Combien de jours l'arb a l'air ouvert sur le spread nu, et combien le restent une
    fois payés le fret, le pont de spec et le financement.
    """
    return OpenDaysComparison(
        n_obs=len(arb_frame),
        days_looking_open=int(arb_frame["looks_open_naif"].sum()),
        days_really_open=int(arb_frame["is_open"].sum()),
    )


def monthly_profile(series: pd.Series) -> pd.DataFrame:
    """Profil moyen par mois calendaire. Le distillat transatlantique est saisonnier
    (chauffage européen, campagnes de maintenance), donc un arb qui ne survit que
    quelques mois par an n'est pas le même objet qu'un arb permanent.
    """
    frame = pd.DataFrame({"valeur": series})
    frame["mois"] = frame.index.month
    grouped = frame.groupby("mois")["valeur"].agg(["mean", "std", "count"])
    grouped.index.name = "mois"
    return grouped.round(3)


# ------------------------------------------------------------------- C-2 : fret calculé
def implied_freight_from_tce(
    target_tce_usd_per_day: float,
    total_days: float,
    total_voyage_costs_usd: float,
    cargo_t: float,
    commission: float,
) -> float:
    """Inverse le moteur TCE : quel taux de fret produit le TCE visé ?

        TCE = (cargo · F · (1 − c) − coûts) / jours
        =>  F = (TCE · jours + coûts) / (cargo · (1 − c))

    C'est la voie C-2. Si les flat rates Worldscale sont indisponibles, on ne renonce pas
    au projet : on reconstruit le fret depuis l'économie du voyage — distances,
    consommation, prix des soutes, jours de port — et on le compare à la route cotée
    partout où on dispose d'un point de comparaison.

    Même forme algébrique que `voyage/indifference.fair_value_c3`, et c'est normal : les
    deux résolvent une équation de TCE en le taux de fret.
    """
    if total_days <= 0:
        raise ValueError("total_days doit être > 0")
    if cargo_t <= 0:
        raise ValueError("cargo_t doit être > 0")
    if not 0.0 <= commission < 1.0:
        raise ValueError(f"commission doit être dans [0, 1), reçu {commission}")
    return (target_tce_usd_per_day * total_days + total_voyage_costs_usd) / (
        cargo_t * (1.0 - commission)
    )


def freight_model_vs_quoted(
    modelled: pd.Series, quoted: pd.Series
) -> dict[str, float]:
    """Écart entre le fret reconstruit et le fret coté.

    Un écart systématiquement d'un côté n'est pas une erreur de modèle : c'est ce que la
    route intègre au-delà du coût de voyage — attente, positionnement, pouvoir de
    négociation. C'est le résultat de la variante C-2, pas son échec.
    """
    aligned = pd.concat({"modelled": modelled, "quoted": quoted}, axis=1).dropna()
    if aligned.empty:
        raise ValueError("aucune date commune entre le fret modélisé et le fret coté")
    diff = aligned["modelled"] - aligned["quoted"]
    return {
        "n_obs": float(len(aligned)),
        "ecart_moyen_usd_t": float(diff.mean()),
        "ecart_median_usd_t": float(diff.median()),
        "ecart_moyen_pct": float(100.0 * (diff / aligned["quoted"]).mean()),
        "correlation": float(np.corrcoef(aligned["modelled"], aligned["quoted"])[0, 1]),
    }
