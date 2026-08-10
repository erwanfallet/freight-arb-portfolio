"""T3-2 — Sucre : pourquoi le mix ne suit pas la parité.

THÈSE
-----
L'élasticité du mix à la parité n'a de sens que **conditionnellement au niveau de
couverture d'entrée de saison**. Le mix n'est pas un ratio de prix : c'est la solution d'un
programme sous contraintes, et la contrainte qui mord change d'une région à l'autre et
d'une saison à l'autre.

LE DÉSACCORD, OUVERT ET CHIFFRÉ ENTRE DEUX MAISONS
----------------------------------------------------
- **Hedgepoint** (févr. 2026) : récolte CS 2026/27 à 630 Mt, production sucre stable
  ~40,5 Mt, mix à ~48,6 % contre 50,6 % la saison précédente. Le mix devrait tomber vers
  **46 %** pour réduire matériellement l'excédent, mais **les limites d'usine et le sucre
  déjà vendu à terme** rendent la chose difficile. Excédent mondial élargi à 3,4 Mt.
- **Czarnikow** (juin 2026) : les mills entrent **beaucoup moins couverts** que les quatre
  saisons précédentes ; les opportunités de pricing 2026/27 sont restées sous BRL 2000/t,
  sous le coût de production ; UNICA montre le mix au plus bas depuis 2022/23 ; production
  révisée à 39,5 Mt.
- Czarnikow (juil. 2026) : la parité éthanol/essence au plus bas de la décennie sans que la
  demande réagisse assez ; l'éthanol de maïs ajoute structurellement de l'offre.

Hedgepoint dit contraintes ; Czarnikow observe que le déblocage vient précisément d'un
niveau de couverture inhabituellement bas. **Les deux s'accordent sur le fait que le degré
de couverture préalable est le paramètre discriminant** — variable observable et rarement
modélisée. C'est là que la page se place.

POSTURE
-------
On ne tranche pas entre les deux maisons. On mesure l'élasticité **conditionnelle**, et on
laisse l'insider dire si la conditionnalité correspond à ce qu'il voit.

LE PIÈGE D'UNITÉ
----------------
La parité exige d'empiler quatre conversions : des kg d'ATR par litre d'éthanol, des BRL
par litre, des BRL par kg, puis des cents par livre via le taux de change **et** la
livre-kilogramme. Une seule erreur dans la chaîne déplace la parité de plusieurs cents et
inverse le signal. Et les coefficients Consecana sont **révisés** : les figer dans le code
fausse tout l'historique (mode de défaillance connu).

MODÈLE
------
    sugar_eq_hydrous_brl_kg = P_hydrous_brl_l x (ATR_sucre / ATR_hydrate)
    sugar_eq_hydrous_c_lb   = sugar_eq_hydrous_brl_kg x 100 / (USDBRL x 2,20462)
    parity_gap_c_lb         = NY11_c_lb x pol_factor - sugar_eq_hydrous_c_lb

Estimation en panel, par quinzaine UNICA x région :

    dmix = a_r + b1 parity_gap_{t-1}
                + b2 (parity_gap_{t-1} x hedge_ratio_entry_r)
                + b3 cap_utilisation
                + b4 (dist_port_r x parity_gap_{t-1}) + e

**L'objet d'intérêt est b2**, pas b1.

HYPOTHÈSES
----------
G-H1  Coefficients Consecana paramétrés, jamais figés (ils sont révisés chaque saison).
G-H2  `pol_factor` ajuste le NY11 (96° pol) vers la qualité VHP effectivement produite.
G-H3  Le taux de couverture d'entrée de saison est constant sur la saison, par région.
      Faux en toute rigueur — les mills continuent de pricer — mais c'est la variable telle
      que les deux maisons la décrivent.
G-H4  CEPEA cote parfois TTC, parfois HT selon la série. Le paramètre `strip_vat` existe
      pour ça et doit être décidé série par série, pas par défaut.
G-H5  L'année électorale brésilienne influence les prix à la pompe indépendamment du
      marché. Traité comme variable de contrôle possible, jamais comme du bruit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.data.snapshot import cached

from agri.core.stats import HacRegression, hac_ols
from agri.core.fmt import fr, fr_pct
from agri.core.units import LB_PER_TONNE, strip_vat

# G-H1 — barème Consecana, kg d'ATR par unité produite. À VÉRIFIER chaque saison.
ATR_SUGAR_VHP_PER_KG = 1.0495
ATR_ETHANOL_ANHYDROUS_PER_L = 1.7651
ATR_ETHANOL_HYDROUS_PER_L = 1.6913

KG_PER_LB = LB_PER_TONNE / 1000.0        # 2,20462 lb par kg
DEFAULT_POL_FACTOR = 0.98                # G-H2


class SugarMixError(ValueError):
    """Modèle mal spécifié."""


# ===========================================================================
# La parité
# ===========================================================================
def hydrous_sugar_equivalent_cents_lb(
    hydrous_brl_l: pd.Series,
    usdbrl: pd.Series,
    *,
    atr_sugar: float = ATR_SUGAR_VHP_PER_KG,
    atr_hydrous: float = ATR_ETHANOL_HYDROUS_PER_L,
    vat_rate: float | None = None,
) -> pd.Series:
    """Prix de l'éthanol hydraté converti en équivalent sucre, en cents/lb.

    Chaîne de conversion, à lire dans l'ordre parce que chaque étape peut être fausse
    séparément :
        BRL/litre  --(ATR_sucre / ATR_hydrate)-->  BRL/kg de sucre équivalent
        BRL/kg     --(/ USDBRL)-->                 USD/kg
        USD/kg     --(/ 2,20462)-->                USD/lb
        USD/lb     --(x 100)-->                    cents/lb

    `vat_rate` retire la TVA quand la série CEPEA est cotée TTC (G-H4). Laisser à None
    quand elle est déjà HT — le module ne devine pas.
    """
    if atr_hydrous <= 0 or atr_sugar <= 0:
        raise SugarMixError("les coefficients Consecana doivent être > 0")
    aligned = pd.concat({"hydrous": hydrous_brl_l, "fx": usdbrl}, axis=1).dropna()
    if aligned.empty:
        raise SugarMixError("aucune date commune à l'hydraté et au change")
    if (aligned["fx"] <= 0).any():
        raise SugarMixError("USDBRL doit être > 0 — vérifier le sens de cotation")

    price = aligned["hydrous"]
    if vat_rate is not None:
        price = strip_vat(price, vat_rate)

    brl_per_kg = price * (atr_sugar / atr_hydrous)
    return (brl_per_kg * 100.0 / (aligned["fx"] * KG_PER_LB)).rename("sugar_eq_c_lb")


def parity_gap_cents_lb(
    ny11_cents_lb: pd.Series,
    hydrous_brl_l: pd.Series,
    usdbrl: pd.Series,
    *,
    pol_factor: float = DEFAULT_POL_FACTOR,
    **conversion_kwargs,
) -> pd.DataFrame:
    """Écart de parité sucre/éthanol, en cents/lb. Positif = le sucre paie mieux.

    Colonnes : ny11, ny11_adjusted, sugar_eq_ethanol, parity_gap, sugar_favoured.
    """
    equivalent = hydrous_sugar_equivalent_cents_lb(
        hydrous_brl_l, usdbrl, **conversion_kwargs
    )
    frame = pd.concat({"ny11": ny11_cents_lb, "sugar_eq_ethanol": equivalent}, axis=1).dropna()
    if frame.empty:
        raise SugarMixError("aucune date commune au NY11 et à l'équivalent éthanol")
    frame["ny11_adjusted"] = frame["ny11"] * pol_factor
    frame["parity_gap"] = frame["ny11_adjusted"] - frame["sugar_eq_ethanol"]
    frame["sugar_favoured"] = frame["parity_gap"] > 0
    frame.attrs["pol_factor"] = pol_factor
    return frame


def consecana_sensitivity(
    ny11_cents_lb: pd.Series,
    hydrous_brl_l: pd.Series,
    usdbrl: pd.Series,
    *,
    atr_sugar_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """G-H1 : ce qu'un changement de barème Consecana fait à la parité.

    Section obligatoire. Les coefficients sont révisés ; figés dans le code, ils faussent
    tout l'historique — et l'erreur est silencieuse, puisque la parité reste plausible.
    """
    values = (
        np.array([1.0300, 1.0495, 1.0700])
        if atr_sugar_values is None
        else np.asarray(atr_sugar_values)
    )
    rows = []
    for atr in values:
        frame = parity_gap_cents_lb(
            ny11_cents_lb, hydrous_brl_l, usdbrl, atr_sugar=float(atr)
        )
        rows.append(
            {
                "atr_sugar": float(atr),
                "mean_parity_gap": float(frame["parity_gap"].mean()),
                "share_sugar_favoured": float(frame["sugar_favoured"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Le mix sous contraintes
# ===========================================================================
@dataclass(frozen=True)
class MillConstraints:
    """Les trois contraintes du programme du mill, par région."""

    region: str
    crystallisation_cap: float       # mix maximal atteignable techniquement
    presold_share: float             # mix minimal imposé par le sucre déjà vendu
    logistics_cost_c_lb: float       # coût d'acheminement vers le port
    hedge_ratio_entry: float         # taux de couverture d'entrée de saison

    def __post_init__(self) -> None:
        if not 0.0 <= self.presold_share <= self.crystallisation_cap <= 1.0:
            raise SugarMixError(
                f"{self.region} : contraintes incohérentes — il faut "
                f"0 <= pré-vendu ({self.presold_share}) <= capacité "
                f"({self.crystallisation_cap}) <= 1"
            )
        if not 0.0 <= self.hedge_ratio_entry <= 1.0:
            raise SugarMixError(f"{self.region} : taux de couverture hors [0, 1]")


def optimal_mix(parity_gap_c_lb: float, constraints: MillConstraints) -> dict:
    """Programme du mill : maximiser le revenu sous les trois contraintes.

    Le programme est linéaire en `mix`, donc la solution est **toujours à une borne** —
    c'est précisément ce qui rend le mix insensible à la parité sur de larges plages, et
    c'est l'argument de Hedgepoint. Le signal de prix ne déplace le mix que lorsqu'il
    change la borne active.

    Renvoie le mix retenu et **quelle contrainte est saturée**, qui est l'information que
    la page cherche à afficher par région et par quinzaine.
    """
    net_gap = parity_gap_c_lb - constraints.logistics_cost_c_lb
    if net_gap > 0:
        return {
            "mix": constraints.crystallisation_cap,
            "binding": "capacité de cristallisation",
            "net_gap": net_gap,
        }
    return {
        "mix": constraints.presold_share,
        "binding": "sucre déjà vendu",
        "net_gap": net_gap,
    }


def binding_constraint_panel(
    parity: pd.Series, regions: list[MillConstraints]
) -> pd.DataFrame:
    """S5 — quelle contrainte est saturée, par quinzaine et par région."""
    rows = []
    for date, gap in parity.dropna().items():
        for region in regions:
            solution = optimal_mix(float(gap), region)
            rows.append(
                {
                    "date": date,
                    "region": region.region,
                    "parity_gap": float(gap),
                    "mix_predicted": solution["mix"],
                    "binding": solution["binding"],
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# L'élasticité conditionnelle — le livrable
# ===========================================================================
@dataclass(frozen=True)
class MixElasticity:
    """b1 et b2 : l'élasticité du mix à la parité, et sa conditionnalité au hedge."""

    regression: HacRegression
    beta_parity: float
    beta_interaction: float
    interaction_pvalue: float
    n_obs: int
    n_regions: int

    def elasticity_at_hedge(self, hedge_ratio: float) -> float:
        """Points de mix par point de parité, à un niveau de couverture donné.

        C'est la **courbe** qui est le livrable, pas un nombre : la spec le dit
        explicitement, et une courbe est plus difficile à balayer d'un revers de main
        qu'un point unique.
        """
        return self.beta_parity + self.beta_interaction * hedge_ratio

    @property
    def hedging_matters(self) -> bool:
        return self.interaction_pvalue < 0.05

    def headline(self, low: float = 0.30, high: float = 0.60) -> str:
        at_low = self.elasticity_at_hedge(low)
        at_high = self.elasticity_at_hedge(high)
        if not self.hedging_matters:
            return (
                f"L'élasticité du mix à la parité vaut {self.beta_parity:+.4f} point de mix "
                f"par cent/lb, et je ne trouve pas d'effet significatif du niveau de "
                f"couverture d'entrée (p = {self.interaction_pvalue:.3f}). Sur cet "
                "échantillon, la thèse de la conditionnalité au hedge ne tient pas."
            )
        if abs(at_low) < 1e-12:
            return "élasticité nulle à couverture basse — résultat non interprétable"
        # points de mix par 100 pts de parité, plutôt que l'inverse (cents necessaires
        # pour 1 pt de mix) : l'inverse explose en un nombre absurde quand l'élasticité
        # est petite, alors que la pente elle-même reste lisible à n'importe quelle echelle
        ratio = abs(at_low) / abs(at_high) if at_high else float("inf")
        return (
            f"100 points d'écart de parité déplacent le mix de {at_high * 100:+.2f} point "
            f"quand les mills entrent couverts à {high:.0%}, contre {at_low * 100:+.2f} "
            f"point quand ils entrent à {low:.0%} — {ratio:.1f}x plus sensible en dessous. "
            "Le mix de 46 % que réclame le bilan mondial n'est atteignable cette saison "
            "que parce que la couverture d'entrée est basse."
        )


def estimate_mix_elasticity(panel: pd.DataFrame) -> MixElasticity:
    """Panel par quinzaine x région, effets fixes région, erreurs HAC.

    `panel` doit contenir : region, d_mix, parity_gap_lag, hedge_ratio_entry,
    cap_utilisation, dist_port.

    Les effets fixes région sont introduits par indicatrices explicites plutôt que par
    démeanage : sur une dizaine de régions, la lisibilité vaut plus que l'élégance, et on
    voit les niveaux régionaux dans la sortie.
    """
    required = {
        "region",
        "d_mix",
        "parity_gap_lag",
        "hedge_ratio_entry",
        "cap_utilisation",
        "dist_port",
    }
    missing = required - set(panel.columns)
    if missing:
        raise SugarMixError(f"colonnes manquantes dans le panel : {sorted(missing)}")

    clean = panel.dropna(subset=sorted(required - {"region"}))
    if len(clean) < 60:
        raise SugarMixError(f"panel trop court : n={len(clean)}")

    design = pd.DataFrame(index=clean.index)
    design["parity"] = clean["parity_gap_lag"]
    design["parity_x_hedge"] = clean["parity_gap_lag"] * clean["hedge_ratio_entry"]
    design["cap_utilisation"] = clean["cap_utilisation"]
    design["parity_x_dist"] = clean["parity_gap_lag"] * clean["dist_port"]

    dummies = pd.get_dummies(clean["region"], prefix="region", drop_first=True).astype(float)
    design = pd.concat([design, dummies], axis=1)

    regression = hac_ols(clean["d_mix"], design)
    return MixElasticity(
        regression=regression,
        beta_parity=float(regression.params["parity"]),
        beta_interaction=float(regression.params["parity_x_hedge"]),
        interaction_pvalue=float(regression.pvalues["parity_x_hedge"]),
        n_obs=len(clean),
        n_regions=clean["region"].nunique(),
    )


def mix_required_for_surplus(
    *, target_surplus_mt: float, current_surplus_mt: float, cane_mt: float, atr_yield_t_per_mt: float = 0.140
) -> float:
    """Le déplacement de mix nécessaire pour ramener l'excédent mondial à une cible.

    Répond directement à Hedgepoint : « il faudrait 46 % » devient « il faudrait retirer
    X Mt, soit Y points de mix, et voici si les contraintes le permettent ».
    """
    if cane_mt <= 0 or atr_yield_t_per_mt <= 0:
        raise SugarMixError("le broyage et le rendement ATR doivent être > 0")
    tonnes_to_remove = current_surplus_mt - target_surplus_mt
    sugar_capacity_mt = cane_mt * atr_yield_t_per_mt
    return tonnes_to_remove / sugar_capacity_mt


# ===========================================================================
# LE PLANCHER QUI N'EN EST PAS UN — sur NY11 et USDBRL réels
# ===========================================================================
# Ni le mix UNICA ni l'éthanol CEPEA ne sont dans l'export : la régression en panel
# ci-dessus reste donc sur jeu synthétique. Mais les deux séries qui suffisent à trancher
# une affirmation sourcée y sont, elles — le NY11 et l'USDBRL. C'est ce que fait cette
# section, et elle produit le résultat le plus contre-intuitif de la page.
CENTS_LB_TO_USD_T = LB_PER_TONNE / 100.0     # 22,0462

# Ancrage SOURCÉ : Czarnikow (juin 2026) note que les opportunités de pricing 2026/27 sont
# restées « sous BRL 2000/t, sous le coût de production ». C'est un nombre publié, daté et
# attribuable — pas une hypothèse de la page. Il est exposé en paramètre parce qu'il est
# régional et évolutif, pas parce qu'il serait incertain.
CZARNIKOW_COST_BRL_T = 2000.0


@cached('t3_2_parity')
def load_real_parity_frame(start: str | None = "2000-01-01") -> pd.DataFrame:
    """NY11 et USDBRL réels, plus le sucre exprimé en BRL par tonne.

    Colonnes : ny11 (c/lb), usdbrl, sugar_brl_t.

    C'est la seule jambe de la parité que l'export permet de construire : le prix de
    l'éthanol hydraté (CEPEA) n'y est pas. La page ne fait donc pas semblant de calculer un
    parity gap — elle exploite ce que ces deux séries suffisent à établir.
    """
    from agri.data.bloomberg_loader import load

    frame = pd.concat(
        {"ny11": load("sugar_no11"), "usdbrl": load("usdbrl")}, axis=1, sort=True
    ).dropna()
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if frame.empty:
        raise SugarMixError(f"aucune date commune au NY11 et à l'USDBRL après {start}")
    frame["sugar_brl_t"] = frame["ny11"] * CENTS_LB_TO_USD_T * frame["usdbrl"]
    return frame


def indifference_hydrous_brl_l(
    ny11_cents_lb: pd.Series,
    usdbrl: pd.Series,
    *,
    pol_factor: float = DEFAULT_POL_FACTOR,
    atr_sugar: float = ATR_SUGAR_VHP_PER_KG,
    atr_hydrous: float = ATR_ETHANOL_HYDROUS_PER_L,
) -> pd.Series:
    """Le prix de l'éthanol hydraté qui rend une usine indifférente, en BRL/litre.

    C'est `hydrous_sugar_equivalent_cents_lb` inversée : au lieu de partir d'un prix
    d'éthanol pour le comparer au sucre, on part du sucre et on demande **à quel prix
    d'éthanol le moulin arrête de faire du sucre**. Un nombre que toute salle de marché
    brésilienne cote, et qui ne demande que les deux séries dont on dispose.

        hydraté* = NY11 x pol_factor x USDBRL x 2,20462 x (ATR_hydraté / ATR_sucre) / 100
    """
    if atr_sugar <= 0 or atr_hydrous <= 0:
        raise SugarMixError("les coefficients Consecana doivent être > 0")
    aligned = pd.concat({"ny11": ny11_cents_lb, "fx": usdbrl}, axis=1, sort=True).dropna()
    if aligned.empty:
        raise SugarMixError("aucune date commune au NY11 et à l'USDBRL")
    if (aligned["fx"] <= 0).any():
        raise SugarMixError("USDBRL doit être > 0 — vérifier le sens de cotation")

    return (
        aligned["ny11"]
        * pol_factor
        * aligned["fx"]
        * KG_PER_LB
        * (atr_hydrous / atr_sugar)
        / 100.0
    ).rename("hydrous_indifference_brl_l")


@dataclass(frozen=True)
class ProductionCostCheck:
    """L'affirmation de Czarnikow, confrontée aux prix.

    Une maison publie « le pricing est resté sous le coût de production ». C'est une
    affirmation vérifiable, et la vérifier vaut mieux que la citer.
    """

    frame: pd.DataFrame
    cost_brl_t: float
    start: str

    @property
    def share_below(self) -> float:
        return float((self.frame["sugar_brl_t"] < self.cost_brl_t).mean())

    @property
    def last_brl_t(self) -> float:
        return float(self.frame["sugar_brl_t"].iloc[-1])

    @property
    def is_below_now(self) -> bool:
        return self.last_brl_t < self.cost_brl_t

    @property
    def headline(self) -> str:
        verdict = "sous" if self.is_below_now else "au-dessus du"
        return (
            f"Au dernier cours, le sucre vaut {fr(self.last_brl_t, 0)} BRL/t — {verdict} le "
            f"coût de production de {fr(self.cost_brl_t, 0)} BRL/t que Czarnikow nomme en "
            f"juin 2026. Depuis {self.start[:4]}, le prix a passé {fr_pct(self.share_below)} "
            "du temps sous ce seuil."
        )


def production_cost_check(
    frame: pd.DataFrame, *, cost_brl_t: float = CZARNIKOW_COST_BRL_T
) -> ProductionCostCheck:
    """Confronte le prix du sucre en BRL au coût de production publié."""
    if "sugar_brl_t" not in frame.columns:
        raise SugarMixError("le frame doit contenir 'sugar_brl_t' — voir load_real_parity_frame")
    if cost_brl_t <= 0:
        raise SugarMixError("le coût de production doit être > 0")
    return ProductionCostCheck(
        frame=frame, cost_brl_t=float(cost_brl_t), start=str(frame.index.min().date())
    )


@dataclass(frozen=True)
class MovingFloor:
    """LE résultat : le « plancher de coût brésilien » coté en cents/lb n'est pas un plancher.

    Un coût de production réel se libelle dans la monnaie où les charges sont payées, donc
    en BRL. Traduit en cents/lb pour un lecteur new-yorkais, il devient une fonction du taux
    de change — et se met à bouger de plusieurs cents sans qu'aucun coût brésilien n'ait
    changé. Un trader qui voit « du support vers 18 cents parce que c'est le coût brésilien »
    regarde une série de change déguisée en niveau structurel.
    """

    frame: pd.DataFrame
    cost_brl_t: float

    @property
    def floor_min(self) -> float:
        return float(self.frame["floor_c_lb"].min())

    @property
    def floor_max(self) -> float:
        return float(self.frame["floor_c_lb"].max())

    @property
    def floor_range(self) -> float:
        return self.floor_max - self.floor_min

    @property
    def floor_last(self) -> float:
        return float(self.frame["floor_c_lb"].iloc[-1])

    @property
    def fx_range(self) -> tuple[float, float]:
        return float(self.frame["usdbrl"].min()), float(self.frame["usdbrl"].max())

    @property
    def headline(self) -> str:
        low, high = self.fx_range
        return (
            f"Le même coût de {fr(self.cost_brl_t, 0)} BRL/t se traduit par un plancher NY11 "
            f"allant de {fr(self.floor_min, 1)} à {fr(self.floor_max, 1)} c/lb sur la "
            f"période — {fr(self.floor_range, 1)} cents d'amplitude, produits uniquement par "
            f"un USDBRL passé de {fr(low, 2)} à {fr(high, 2)}. Aucun coût brésilien n'a "
            f"bougé. Au dernier change, le plancher est à {fr(self.floor_last, 1)} c/lb."
        )


def moving_floor(
    frame: pd.DataFrame, *, cost_brl_t: float = CZARNIKOW_COST_BRL_T
) -> MovingFloor:
    """Le prix NY11 qui atteint exactement le coût de production, jour par jour.

        plancher_c_lb = coût_BRL_t / (22,0462 x USDBRL)

    Le résultat est une **série**, pas un niveau — et c'est tout le propos.
    """
    if "usdbrl" not in frame.columns:
        raise SugarMixError("le frame doit contenir 'usdbrl'")
    if cost_brl_t <= 0:
        raise SugarMixError("le coût de production doit être > 0")

    out = frame.copy()
    out["floor_c_lb"] = cost_brl_t / (CENTS_LB_TO_USD_T * out["usdbrl"])
    out["below_floor"] = out["ny11"] < out["floor_c_lb"]
    return MovingFloor(frame=out, cost_brl_t=float(cost_brl_t))


def floor_variance_decomposition(frame: pd.DataFrame) -> pd.Series:
    """Part du mouvement du plancher imputable au change plutôt qu'au sucre.

    Le plancher en cents/lb est **exactement** proportionnel à 1/USDBRL : sa variation est
    donc entièrement du change, par construction. Ce qui mérite d'être décomposé, c'est
    l'écart entre le prix et son plancher — c'est-à-dire ce qu'un moulin regarde vraiment.

        ln(sucre_BRL) = ln(NY11) + ln(USDBRL) + const
    """
    changes = np.log(frame[["ny11", "usdbrl"]]).diff().dropna()
    if len(changes) < 30:
        raise SugarMixError(f"échantillon trop court pour décomposer : n={len(changes)}")

    var_sugar = float(changes["ny11"].var())
    var_fx = float(changes["usdbrl"].var())
    covariance = float(changes["ny11"].cov(changes["usdbrl"]))
    total = var_sugar + var_fx + 2 * covariance
    if total <= 0:
        raise SugarMixError("variance totale nulle ou négative — décomposition impossible")

    return pd.Series(
        {
            "share_sugar": var_sugar / total,
            "share_fx": var_fx / total,
            "share_covariance": 2 * covariance / total,
            "correlation": float(changes["ny11"].corr(changes["usdbrl"])),
        }
    )


__all__ = [
    "ATR_ETHANOL_ANHYDROUS_PER_L",
    "ATR_ETHANOL_HYDROUS_PER_L",
    "ATR_SUGAR_VHP_PER_KG",
    "CENTS_LB_TO_USD_T",
    "CZARNIKOW_COST_BRL_T",
    "MillConstraints",
    "MixElasticity",
    "MovingFloor",
    "ProductionCostCheck",
    "SugarMixError",
    "binding_constraint_panel",
    "consecana_sensitivity",
    "estimate_mix_elasticity",
    "floor_variance_decomposition",
    "hydrous_sugar_equivalent_cents_lb",
    "indifference_hydrous_brl_l",
    "load_real_parity_frame",
    "mix_required_for_surplus",
    "moving_floor",
    "optimal_mix",
    "parity_gap_cents_lb",
    "production_cost_check",
]
