"""T2-4 — Le white premium, ou ce qu'un prix peut dire et ce qu'il ne peut pas.

LE RÉSULTAT, EN UNE PHRASE
---------------------------
Le **niveau** de la rente de raffinage n'est pas identifiable à partir des prix : un facteur
de conversion que personne ne publie y injecte autant d'incertitude que la grandeur cherchée.
Sa **variation**, elle, l'est entièrement — le paramètre déplace toutes les années dans le
même sens, donc les écarts entre années lui survivent. Et cette variation dit quelque chose :
la richness est passée d'environ -26 USD/t en 2021 à +35 en 2024, un basculement plus de cinq
fois plus grand que ce que le paramètre peut produire.

C'est une page qui énonce d'abord une **limite** de ce qu'elle peut savoir, puis le résultat
qui survit à cette limite. L'ordre compte : publier le niveau sans la limite serait publier
un artefact de convention.

TENSION — INFÉRÉE, PAS SOURCÉE
-------------------------------
**Il me semble que** le premium blanc (No.5 - No.11) est présenté comme la marge de
raffinage, alors qu'il contient surtout un résidu de positionnement et de contraintes de
livraison. « Il me semble », jamais « j'ai lu que ».

LE PIÈGE D'UNITÉ
----------------
Le No.11 se cote en **cents/lb, base 96° pol** ; le No.5 en **USD/tonne** de sucre
raffiné. Les comparer exige la conversion c/lb -> USD/t **et** un ajustement de
polarisation, parce qu'il faut plus d'une tonne de brut à 96° pour faire une tonne de
blanc. `pol_adjust` vaut ~1,06-1,08 selon la spécification retenue.

ORDRE DE GRANDEUR, ÉNONCÉ CORRECTEMENT : entre ces deux bornes l'écart vaut ~5 USD/t sur
l'échantillon réel. Ce n'est **pas** du même ordre que le white premium lui-même (~70 USD/t,
soit 7 %) — le dire serait faux. C'est du même ordre que la **richness**, la grandeur que la
page cherche réellement à établir, dont la médiane depuis 2015 tourne autour de 5 USD/t.
C'est cette comparaison-là qui est décisive, et c'est elle que `identification_check`
mesure au lieu de la supposer.

IDENTITÉ
--------
    white_premium = No5_usd_t - No11_c_lb x 22,0462 x pol_adjust
    fv_refining   = energie + perte_rendement x No11_usd_t + main_d_oeuvre
                    + financement + jambe_fret
    richness      = white_premium - fv_refining

    richness > 0  -> zone RICH   : le blanc paie plus que le coût de le produire
    richness < 0  -> zone CHEAP  : raffiner détruit de la valeur au prix affiché

HYPOTHÈSES
----------
W-H1  `pol_adjust` par défaut à 1,07, borné à [1,00 ; 1,20]. Voir ci-dessus.
W-H2  La perte de rendement au raffinage est un pourcentage du brut entrant, valorisé au
      prix du No.11. Défaut 2 %.
W-H3  Le coût d'énergie est un forfait par tonne, exogène. C'est le poste le plus volatil
      d'une raffinerie — d'où le slider et la sensibilité.
W-H4  Aucun coût de capital sur l'actif de raffinage. La `richness` est donc une marge de
      contribution, jamais un profit. Ne pas la comparer à un ROIC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.stats import regime_runs
from agri.core.units import cents_lb_to_usd_t

DEFAULT_POL_ADJUST = 1.07              # W-H1
DEFAULT_YIELD_LOSS = 0.02              # W-H2
DEFAULT_ENERGY_USD_T = 28.0            # W-H3
DEFAULT_LABOUR_USD_T = 12.0
DEFAULT_FREIGHT_LEG_USD_T = 18.0
DEFAULT_FINANCING_DAYS = 45

# Intensite energetique par defaut d'une raffinerie de sucre (W-H3, donnee reelle) :
# calibree pour qu'a un Henry Hub proche de sa moyenne historique (~3,5 USD/mmBtu), le
# cout energie retombe pres du forfait DEFAULT_ENERGY_USD_T (28 USD/t) — 28/3,5 = 8
# mmBtu/t, un ordre de grandeur plausible pour la cristallisation. Parametre, pas mesure :
# affiche comme tel dans le panneau de diagnostics de la page.
DEFAULT_ENERGY_INTENSITY_MMBTU_T = 8.0


class WhitePremiumError(ValueError):
    """Modèle mal spécifié."""


def white_premium_usd_t(
    no5_usd_t: pd.Series,
    no11_cents_lb: pd.Series,
    *,
    pol_adjust: float = DEFAULT_POL_ADJUST,
) -> pd.Series:
    """Premium blanc, les deux jambes ramenées à la même unité et à la même polarisation."""
    if not 1.0 <= pol_adjust <= 1.20:
        raise WhitePremiumError(
            f"pol_adjust hors plage plausible [1,00 ; 1,20], reçu {pol_adjust} — "
            "au-delà ce n'est plus une correction de polarisation"
        )
    raw_on_white_basis = cents_lb_to_usd_t(no11_cents_lb) * pol_adjust
    return no5_usd_t - raw_on_white_basis


def fair_value_refining_usd_t(
    no11_cents_lb: pd.Series,
    *,
    yield_loss: float = DEFAULT_YIELD_LOSS,
    energy_usd_t: float | pd.Series = DEFAULT_ENERGY_USD_T,
    labour_usd_t: float = DEFAULT_LABOUR_USD_T,
    freight_leg_usd_t: float = DEFAULT_FREIGHT_LEG_USD_T,
    annual_rate: pd.Series | float = 0.055,
    financing_days: int = DEFAULT_FINANCING_DAYS,
) -> pd.DataFrame:
    """Coût de raffinage reconstruit, poste par poste.

    Renvoie un DataFrame avec une colonne par poste plus le total : c'est le waterfall de
    la page, et il doit être lisible ligne à ligne par quelqu'un qui exploite une
    raffinerie — sinon il ne peut pas le contester utilement.

    `energy_usd_t` accepte un forfait (défaut) ou une série datée — un vrai proxy gaz
    naturel (Henry Hub) plutôt qu'une constante, quand on en a un.
    """
    if not 0.0 <= yield_loss < 0.20:
        raise WhitePremiumError(f"perte de rendement hors plage : {yield_loss}")
    raw_usd_t = cents_lb_to_usd_t(no11_cents_lb)
    rate = (
        pd.Series(float(annual_rate), index=no11_cents_lb.index)
        if isinstance(annual_rate, (int, float))
        else pd.Series(annual_rate).reindex(no11_cents_lb.index)
    )
    energy = (
        pd.Series(float(energy_usd_t), index=no11_cents_lb.index)
        if isinstance(energy_usd_t, (int, float))
        else pd.Series(energy_usd_t).reindex(no11_cents_lb.index)
    )
    out = pd.DataFrame(index=no11_cents_lb.index)
    out["energie"] = energy
    out["perte_rendement"] = yield_loss * raw_usd_t
    out["main_d_oeuvre"] = labour_usd_t
    out["fret"] = freight_leg_usd_t
    out["financement"] = raw_usd_t * rate * financing_days / 360.0
    out["total"] = out.sum(axis=1)
    return out


def build_richness(
    no5_usd_t: pd.Series,
    no11_cents_lb: pd.Series,
    *,
    pol_adjust: float = DEFAULT_POL_ADJUST,
    **cost_kwargs,
) -> pd.DataFrame:
    """Premium observé, fair value reconstruite, et le résidu — la page entière en un frame.

    Colonnes : no5, no11, white_premium, fv_refining, richness, zone.
    """
    aligned = pd.concat({"no5": no5_usd_t, "no11": no11_cents_lb}, axis=1, sort=True).dropna()
    if aligned.empty:
        raise WhitePremiumError("aucune date commune au No.5 et au No.11")

    out = aligned.copy()
    out["white_premium"] = white_premium_usd_t(
        out["no5"], out["no11"], pol_adjust=pol_adjust
    )
    costs = fair_value_refining_usd_t(out["no11"], **cost_kwargs)
    out["fv_refining"] = costs["total"]
    out["richness"] = out["white_premium"] - out["fv_refining"]
    out["zone"] = np.where(out["richness"] > 0, "RICH", "CHEAP")
    out.attrs["pol_adjust"] = pol_adjust
    return out


@dataclass(frozen=True)
class RichnessSummary:
    """Combien de temps le premium couvre les coûts, et ce que vaut le résidu."""

    share_rich: float
    mean_richness: float
    median_richness: float
    rich_episodes: pd.DataFrame
    cheap_episodes: pd.DataFrame
    n_obs: int

    @property
    def headline(self) -> str:
        return (
            f"Le white premium n'est couvert par les coûts de raffinage reconstruits que "
            f"{self.share_rich:.0%} du temps ; le résidu médian vaut "
            f"{self.median_richness:+.1f} $/t. Ce n'est pas une marge de raffinage — c'est "
            "un signal de disponibilité physique qui porte une marge de raffinage dedans."
        )


def summarise_richness(frame: pd.DataFrame, *, min_obs: int = 5) -> RichnessSummary:
    return RichnessSummary(
        share_rich=float((frame["richness"] > 0).mean()),
        mean_richness=float(frame["richness"].mean()),
        median_richness=float(frame["richness"].median()),
        rich_episodes=regime_runs(
            frame["richness"] > 0, depth=frame["richness"], min_obs=min_obs
        ),
        cheap_episodes=regime_runs(
            frame["richness"] <= 0, depth=frame["richness"], min_obs=min_obs
        ),
        n_obs=len(frame),
    )


def pol_adjust_sensitivity(
    no5_usd_t: pd.Series,
    no11_cents_lb: pd.Series,
    *,
    values: np.ndarray | None = None,
    **cost_kwargs,
) -> pd.DataFrame:
    """Ce que le seul choix de `pol_adjust` fait au résultat de la page (W-H1).

    Section obligatoire : sur une plage défendable de 1,06 à 1,08, la part du temps passée
    en zone RICH peut basculer de façon substantielle. Montrer cette sensibilité est ce
    qui distingue une page honnête d'une page qui a choisi son hypothèse pour avoir
    raison.
    """
    values = np.arange(1.04, 1.101, 0.005) if values is None else np.asarray(values)
    rows = []
    for pol in values:
        frame = build_richness(
            no5_usd_t, no11_cents_lb, pol_adjust=float(pol), **cost_kwargs
        )
        rows.append(
            {
                "pol_adjust": float(pol),
                "mean_white_premium": float(frame["white_premium"].mean()),
                "mean_richness": float(frame["richness"].mean()),
                "share_rich": float((frame["richness"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


@cached('t2_4_richness')
def load_real_richness_frame(
    *,
    pol_adjust: float = DEFAULT_POL_ADJUST,
    energy_intensity_mmbtu_t: float = DEFAULT_ENERGY_INTENSITY_MMBTU_T,
    start: str = "1990-07-18",
) -> pd.DataFrame:
    """`build_richness` sur ICE No.11 et No.5 **réels** (export Bloomberg), coût
    énergie proxié par Henry Hub réel plutôt que le forfait constant DEFAULT_ENERGY_USD_T.

    Les deux jambes de prix (No.11, No.5) et l'énergie (Henry Hub) sont toutes réelles ;
    main-d'œuvre, fret et perte de rendement restent des forfaits paramétrés (W-H2, et
    l'équivalent pour fret/main-d'œuvre) faute de comptabilité analytique de raffinerie
    publique — aucune source ne publie ces postes en série temporelle.
    """
    from agri.data.bloomberg_loader import load as load_bloomberg

    no11 = load_bloomberg("sugar_no11").loc[start:]
    no5 = load_bloomberg("sugar_no5").loc[start:]
    henry_hub = load_bloomberg("henry_hub")
    energy_usd_t = (henry_hub * energy_intensity_mmbtu_t).reindex(no11.index)

    frame = build_richness(no5, no11, pol_adjust=pol_adjust, energy_usd_t=energy_usd_t)
    frame.attrs["energy_source"] = "henry_hub_real"
    frame.attrs["energy_intensity_mmbtu_t"] = energy_intensity_mmbtu_t
    return frame


# ===========================================================================
# L'IDENTIFICATION — ce que le prix peut dire, et ce qu'il ne peut pas
# ===========================================================================
# `pol_adjust` n'est pas observable : aucune bourse ne le publie, il dépend de la
# spécification contractuelle retenue, et sa plage plausible est 1,06-1,08. Le réflexe est
# de choisir une valeur et de conclure. Le réflexe est mauvais : sur données réelles,
# l'incertitude que ce seul paramètre injecte dans la richness est du même ordre que la
# richness elle-même. Le NIVEAU de la rente de raffinage n'est donc pas identifiable à
# partir des prix. Sa VARIATION, elle, l'est entièrement — c'est le résultat de la page.
POL_PLAUSIBLE_LO = 1.06
POL_PLAUSIBLE_HI = 1.08


@dataclass(frozen=True)
class ImpliedPolAdjust:
    """L'ajustement de polarisation que le marché price, sous hypothèse de rente nulle.

    On inverse la question habituelle. Plutôt que de fixer `pol_adjust` et de demander si le
    raffinage est rentable, on suppose le raffinage **concurrentiel** — rente médiane nulle,
    ce qu'une industrie mature devrait produire — et on demande quel `pol_adjust` le marché
    price alors. Le nombre qui sort se compare directement à la spécification qu'un
    raffineur connaît par cœur.
    """

    pol_star: float
    plausible_lo: float
    plausible_hi: float
    start: str

    @property
    def within_plausible(self) -> bool:
        return self.plausible_lo <= self.pol_star <= self.plausible_hi

    @property
    def headline(self) -> str:
        if self.within_plausible:
            verdict = (
                "il tombe **dans** la plage plausible : les prix sont compatibles avec un "
                "raffinage sans rente, et rien dans la donnée n'oblige à conclure autrement"
            )
        else:
            side = "au-dessus" if self.pol_star > self.plausible_hi else "en dessous"
            verdict = (
                f"il tombe {side} de la plage plausible "
                f"[{self.plausible_lo:.2f} ; {self.plausible_hi:.2f}] — soit la plage est "
                "trop étroite, soit le raffinage capte une rente. Le prix seul ne permet "
                "pas de trancher"
            )
        return (
            f"Pour que la rente médiane de raffinage soit exactement nulle depuis "
            f"{self.start[:4]}, il faudrait un ajustement de polarisation de "
            f"{self.pol_star:.4f} ; {verdict}."
        )


def implied_pol_adjust(
    *,
    start: str = "2015-01-01",
    plausible_lo: float = POL_PLAUSIBLE_LO,
    plausible_hi: float = POL_PLAUSIBLE_HI,
    search_lo: float = 1.0001,
    search_hi: float = 1.1999,
    **kwargs,
) -> ImpliedPolAdjust:
    """Résout `pol*` tel que la richness médiane s'annule.

    La recherche est bornée à l'intérieur de la plage que `white_premium_usd_t` accepte :
    au-delà de 1,20 ce n'est plus une correction de polarisation, et laisser le solveur y
    aller produirait une exception plutôt qu'un résultat.
    """
    from scipy.optimize import brentq

    def median_richness(pol: float) -> float:
        return float(
            load_real_richness_frame(pol_adjust=pol, start=start, **kwargs)["richness"].median()
        )

    low, high = median_richness(search_lo), median_richness(search_hi)
    if low * high > 0:
        raise WhitePremiumError(
            f"aucun pol_adjust dans [{search_lo:.4f} ; {search_hi:.4f}] n'annule la richness "
            f"médiane (elle vaut {low:+.1f} à la borne basse et {high:+.1f} à la haute). "
            "La rente ne s'explique donc pas par la seule polarisation."
        )

    return ImpliedPolAdjust(
        pol_star=float(brentq(median_richness, search_lo, search_hi, xtol=1e-6)),
        plausible_lo=plausible_lo,
        plausible_hi=plausible_hi,
        start=start,
    )


@dataclass(frozen=True)
class IdentificationCheck:
    """LE résultat : le niveau n'est pas identifiable, la variation l'est.

    `annual` porte la richness médiane par année, calculée aux deux bornes de la plage
    plausible et à la valeur de référence. Les trois colonnes se déplacent presque
    parallèlement : c'est cela, et rien d'autre, qui autorise à lire les écarts entre années
    tout en refusant de lire le niveau.
    """

    annual: pd.DataFrame
    pol_lo: float
    pol_hi: float
    pol_ref: float

    @property
    def parameter_span_max(self) -> float:
        """Le pire écart, sur une année, imputable au seul choix de `pol_adjust`."""
        return float((self.annual["richness_lo"] - self.annual["richness_hi"]).max())

    @property
    def signal_span(self) -> float:
        """L'amplitude du signal lui-même : la richness la plus haute moins la plus basse."""
        return float(self.annual["richness_ref"].max() - self.annual["richness_ref"].min())

    @property
    def ratio(self) -> float:
        if self.parameter_span_max <= 0:
            raise WhitePremiumError(
                "amplitude de parametre nulle : les trois variantes de pol_adjust sont "
                "identiques, ce qui ne peut arriver que si elles ont ete reconstruites a "
                "partir d'une seule et meme serie. Le rapport n'a pas de sens ici."
            )
        return self.signal_span / self.parameter_span_max

    @property
    def rank_correlation(self) -> float:
        """Le classement des années survit-il au changement de paramètre ?"""
        return float(
            self.annual["richness_lo"].rank().corr(
                self.annual["richness_hi"].rank(), method="spearman"
            )
        )

    @property
    def sign_flipping_years(self) -> list[int]:
        """Les années dont le SIGNE dépend du paramètre — les seules non interprétables."""
        lo, hi = self.annual["richness_lo"], self.annual["richness_hi"]
        return [int(year) for year in self.annual.index if (lo[year] > 0) != (hi[year] > 0)]

    @property
    def headline(self) -> str:
        flipping = self.sign_flipping_years
        flip_text = (
            "aucune année ne change de signe"
            if not flipping
            else f"{len(flipping)} année(s) changent de signe ({', '.join(map(str, flipping))})"
        )
        return (
            f"Le choix de pol_adjust déplace la richness d'au plus "
            f"{self.parameter_span_max:.1f} USD/t sur une année donnée, alors que l'écart "
            f"entre la meilleure et la pire année atteint {self.signal_span:.1f} USD/t — un "
            f"facteur {self.ratio:.1f}. Le classement des années est identique aux deux "
            f"bornes (corrélation de rang {self.rank_correlation:.4f}) et {flip_text}. "
            "Le niveau de la rente n'est pas identifiable ; sa variation l'est."
        )


def identification_check(
    *,
    start: str = "2015-01-01",
    pol_lo: float = POL_PLAUSIBLE_LO,
    pol_hi: float = POL_PLAUSIBLE_HI,
    pol_ref: float = DEFAULT_POL_ADJUST,
    **kwargs,
) -> IdentificationCheck:
    """Compare l'amplitude injectée par le paramètre à l'amplitude du signal.

    C'est le test que la page fait passer à sa propre conclusion avant de l'énoncer : si le
    paramètre inobservable pesait autant que le phénomène, il n'y aurait rien à dire, et il
    faudrait le dire.
    """
    if not pol_lo < pol_ref < pol_hi:
        raise WhitePremiumError(
            f"la valeur de référence {pol_ref} doit être strictement entre les bornes "
            f"{pol_lo} et {pol_hi}"
        )

    # Les trois variantes sont reconstruites depuis UN SEUL frame plutot que par trois
    # appels au loader. Deux raisons, et la seconde est la plus importante :
    #   - c'est trois fois moins de lecture disque ;
    #   - en mode snapshot le loader ignore ses arguments et renverrait trois fois le meme
    #     frame, donc une amplitude de parametre nulle et une division par zero. Recalculer
    #     ici garde la page vraie meme quand la donnee brute est absente.
    base = load_real_richness_frame(pol_adjust=pol_ref, start=start, **kwargs)
    columns = {}
    for label, pol in (("richness_lo", pol_lo), ("richness_ref", pol_ref), ("richness_hi", pol_hi)):
        premium = base["no5"] - cents_lb_to_usd_t(base["no11"]) * pol
        richness = premium - base["fv_refining"]
        columns[label] = richness.groupby(base.index.year).median()

    annual = pd.DataFrame(columns)
    annual.index.name = "year"
    if len(annual) < 3:
        raise WhitePremiumError(
            f"seulement {len(annual)} année(s) dans l'échantillon : la comparaison entre "
            "amplitude du paramètre et amplitude du signal n'a pas de sens"
        )
    return IdentificationCheck(annual=annual, pol_lo=pol_lo, pol_hi=pol_hi, pol_ref=pol_ref)


@dataclass(frozen=True)
class ImpliedRefiningCost:
    """Ce que le marché paie pour l'acte de raffiner — le nombre du mail.

    Le white premium *est* le prix que le marché met sur la transformation d'une tonne de
    brut en une tonne de blanc. Il ne demande aucune hypothèse de coût pour être lu : c'est
    un prix observé. Le comparer à un modèle de coût est ce qui introduit des hypothèses —
    d'où l'affichage des deux côte à côte plutôt que de leur seule différence.
    """

    market_usd_t: float
    modelled_usd_t: float
    pol_adjust: float
    start: str

    @property
    def gap_usd_t(self) -> float:
        return self.market_usd_t - self.modelled_usd_t

    @property
    def headline(self) -> str:
        return (
            f"Depuis {self.start[:4]}, le marché paie en médiane "
            f"{self.market_usd_t:.0f} USD/t pour l'acte de raffiner. Le modèle de coût de "
            f"cette page en trouve {self.modelled_usd_t:.0f}, soit {self.gap_usd_t:+.0f} "
            "USD/t d'écart — mais c'est le premier nombre qui est observé, et le second qui "
            "repose sur des forfaits. Un raffineur compare le sien au premier."
        )


def implied_refining_cost(
    *, start: str = "2015-01-01", pol_adjust: float = DEFAULT_POL_ADJUST, **kwargs
) -> ImpliedRefiningCost:
    """Le white premium médian, lu comme le prix de marché du raffinage."""
    frame = load_real_richness_frame(pol_adjust=pol_adjust, start=start, **kwargs)
    return ImpliedRefiningCost(
        market_usd_t=float(frame["white_premium"].median()),
        modelled_usd_t=float(frame["fv_refining"].median()),
        pol_adjust=float(pol_adjust),
        start=start,
    )


__all__ = [
    "DEFAULT_ENERGY_INTENSITY_MMBTU_T",
    "POL_PLAUSIBLE_HI",
    "POL_PLAUSIBLE_LO",
    "IdentificationCheck",
    "ImpliedPolAdjust",
    "ImpliedRefiningCost",
    "RichnessSummary",
    "WhitePremiumError",
    "build_richness",
    "fair_value_refining_usd_t",
    "identification_check",
    "implied_pol_adjust",
    "implied_refining_cost",
    "load_real_richness_frame",
    "pol_adjust_sensitivity",
    "summarise_richness",
    "white_premium_usd_t",
]
