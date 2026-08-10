"""T2-5 — L'usine comme option sur la marge.

CE QUE CETTE PAGE EXPLIQUE
---------------------------
Un signal de curtailment du type `consecutive_below(margin, 0, N=4)` — celui qui tourne
sur les pages **zinc** et **lithium** — suppose sans le dire que s'arrêter et redémarrer
est gratuit. Ça ne l'est pas : sur un four zinc ou une ligne de conversion de spodumène,
le redémarrage se compte en semaines de production perdue et en réactifs. La règle coûte
alors de l'argent des deux côtés — elle déclenche des arrêts que le coût de redémarrage
ne justifie pas, et elle fait redémarrer avant que la marge ne couvre ce redémarrage.

Et il y a un problème plus dur, visible en croisant deux sections de la page zinc
elle-même : la sensibilité y montre qu'il faut **376 $/t de crédit acide pour équilibrer**,
et que ce crédit est le levier non modélisé le plus important. L'incertitude sur le
**signe** de la marge dépasse donc largement le seuil de 0 $/t que la règle N=4 teste. La
section sensibilité invalide la section curtailment, sur la même page.

LE LIVRABLE — ON INVERSE LA QUESTION
-------------------------------------
Plutôt que de proposer une meilleure règle (que personne n'a demandée), on rend la règle
existante **contestable** : sur un chemin de marge donné, `marge < 0 pendant 4 mois`
arrête et redémarre à des niveaux précis, donc elle *est* équivalente à une bande, donc
elle suppose un coût d'aller-retour précis.

    « Votre règle arrête en médiane à M_off et redémarre à M_on. Une frontière d'exercice
      qui produirait la même bande suppose un coût d'arrêt-redémarrage de X. Est-ce que X
      ressemble au vôtre ? »

C'est une question à laquelle seul quelqu'un qui exploite l'actif peut répondre, et elle
ne demande pas au lecteur d'accepter un modèle — juste de comparer un nombre au sien.

LE CONTREFACTUEL QU'IL FAUT TOUJOURS AFFICHER
----------------------------------------------
Avant de comparer deux règles d'arrêt, il faut vérifier que s'arrêter vaut quoi que ce
soit : `run_always_on_policy` donne le P&L d'une usine qui ne s'arrête jamais. Si la
meilleure règle ne bat pas ce contrefactuel sur la période, toute la discussion sur la
frontière d'exercice est théorique — et la page doit le dire au lieu de comparer deux
règles également inutiles.

L'IDÉE TECHNIQUE
----------------
La règle optimale n'est pas un seuil : c'est une **bande d'hystérésis** `[M_off, M_on]`
avec `M_off < 0 < M_on`, dont la largeur est fixée par les coûts d'arrêt et de redémarrage
et par la volatilité de la marge. Un opérateur rationnel continue à produire à marge
négative si le coût d'arrêt-redémarrage dépasse la perte, et ne redémarre pas dès que la
marge repasse à zéro.

MODÈLE
------
Marge en Ornstein-Uhlenbeck — les marges de transformation sont moyenne-réversives,
contrairement aux prix :

    dM = kappa (theta - M) dt + sigma dW

Calibration par MCO sur `M_{t+1} = a + b M_t + e` :
    kappa = -ln(b)/dt,  theta = a/(1-b),  sigma = sd(e) x sqrt(2 kappa / (1 - b^2))

Valorisation par programmation dynamique à deux états :

    V_on(M)  = max( M - c_fix + d E[V_on(M')] ,  -K_off + d E[V_off(M')] )
    V_off(M) = max( -c_idle    + d E[V_off(M')] ,  -K_on  + d E[V_on(M')]  )

Résolution par itération de la valeur sur une grille de M, puis extraction de la frontière
`M_off*` (on arrête) et `M_on*` (on redémarre).

RÉSULTAT ATTENDU — CONTRE-INTUITIF, ET C'EST L'INTÉRÊT
--------------------------------------------------------
Une usine dont la marge est souvent négative peut valoir **plus** qu'une usine dont la
marge est stablement positive, si la volatilité et la flexibilité sont suffisantes. La
valeur d'option croît avec sigma à moyenne égale.

HYPOTHÈSES
----------
O-H1  Marge OU. Les marges de transformation reviennent à la moyenne ; le tester (ADF+KPSS)
      avant de calibrer, et refuser de calibrer si le test dit racine unitaire.
O-H2  Coûts d'arrêt et de redémarrage forfaitaires, exprimés en jours de marge moyenne.
      C'est le paramètre le plus incertain de la page — d'où deux sliders et une
      sensibilité dédiée.
O-H3  Pas de délai technique entre la décision et l'effet. Un vrai redémarrage prend des
      jours à des semaines, ce qui **élargit** la bande d'hystérésis : biais conservateur.
O-H4  Pas de contrainte de contrat d'approvisionnement ni d'engagement de livraison. Une
      usine réelle ne s'arrête pas librement — la frontière calculée est donc une borne.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agri.core.stats import StationarityVerdict, adf_kpss, regime_runs

STATE_ON = 0
STATE_OFF = 1


class PlantOptionError(ValueError):
    """Modèle mal spécifié ou calibration refusée."""


# ===========================================================================
# Calibration Ornstein-Uhlenbeck
# ===========================================================================
@dataclass(frozen=True)
class OUParams:
    """Paramètres OU, avec le verdict de stationnarité qui autorise leur usage."""

    kappa: float           # vitesse de retour à la moyenne, par unité de temps
    theta: float           # niveau de long terme
    sigma: float           # volatilité instantanée
    dt: float
    stationarity: StationarityVerdict
    n_obs: int

    @property
    def half_life(self) -> float:
        """Demi-vie du retour à la moyenne, dans l'unité de temps de `dt`."""
        return np.log(2.0) / self.kappa if self.kappa > 0 else float("inf")

    @property
    def summary(self) -> str:
        return (
            f"kappa = {self.kappa:.4f}/période (demi-vie {self.half_life:.1f} périodes), "
            f"theta = {self.theta:.2f}, sigma = {self.sigma:.2f} | "
            f"stationnarité : {self.stationarity.verdict}"
        )


def calibrate_ou(margin: pd.Series, *, dt: float = 1.0, strict: bool = True) -> OUParams:
    """Calibre un OU par MCO sur `M_{t+1} = a + b M_t + e` (O-H1).

    `strict=True` **refuse** de calibrer si ADF et KPSS ne concluent pas à la
    stationnarité. Calibrer un OU sur une marche aléatoire produit un kappa proche de zéro
    et une valeur d'option absurde, sans jamais planter — d'où le refus explicite.
    """
    clean = pd.Series(margin).dropna().astype(float)
    if len(clean) < 50:
        raise PlantOptionError(f"au moins 50 observations sont nécessaires, reçu {len(clean)}")

    verdict = adf_kpss(clean)
    if strict and verdict.verdict != "stationary":
        raise PlantOptionError(
            f"la marge n'est pas stationnaire au sens conjoint ADF+KPSS "
            f"(verdict : {verdict.verdict}). Calibrer un OU dessus produirait un kappa "
            "proche de zéro et une valeur d'option absurde. Passer strict=False pour "
            "forcer, en affichant l'avertissement dans la page."
        )

    y = clean.to_numpy()[1:]
    x = clean.to_numpy()[:-1]
    design = np.column_stack([np.ones(len(x)), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ np.array([a, b])

    if not 0.0 < b < 1.0:
        raise PlantOptionError(
            f"coefficient autorégressif hors (0, 1) : b = {b:.4f}. Au-delà de 1 la série "
            "est explosive, à zéro ou en dessous elle n'est pas un OU."
        )

    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma = float(np.std(residuals, ddof=2)) * np.sqrt(2.0 * kappa / (1.0 - b**2))
    return OUParams(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        dt=dt,
        stationarity=verdict,
        n_obs=len(clean),
    )


# ===========================================================================
# Programmation dynamique — la frontière d'exercice
# ===========================================================================
@dataclass(frozen=True)
class HysteresisBand:
    """La vraie règle d'arrêt-redémarrage, et ce qu'elle vaut."""

    m_off: float
    m_on: float
    grid: np.ndarray
    value_on: np.ndarray
    value_off: np.ndarray
    n_iterations: int
    converged: bool

    @property
    def width(self) -> float:
        return self.m_on - self.m_off

    @property
    def is_degenerate(self) -> bool:
        """`M_on < M_off` — la bande est inversée, donc il n'y a pas de politique d'arrêt.

        Ce cas est REEL, pas un bug de solveur : il apparait des que le cout de maintien a
        l'arret devient cher devant le cout de redemarrage. Un actif qui coute 2 par
        periode a l'arret et 3 a redemarrer ne doit jamais rester arrete — la
        programmation dynamique le dit en rendant la region « redemarrer » plus large que
        la region « arreter », donc en les faisant se chevaucher.

        Lu naivement, ce cas produit une largeur de bande NEGATIVE qui se propage
        silencieusement dans une sensibilite ou une interpolation. On le nomme ici pour
        que l'aval refuse de le traiter comme une bande ordinaire.
        """
        return self.m_on < self.m_off

    def option_value_at(self, margin: float) -> float:
        """Valeur de l'usine en marche, à un niveau de marge donné."""
        return float(np.interp(margin, self.grid, self.value_on))

    @property
    def headline(self) -> str:
        if self.is_degenerate:
            return (
                f"Pas de politique d'arrêt à ces coûts : la région « redémarrer » "
                f"(au-dessus de {self.m_on:+.2f}) recouvre la région « arrêter » "
                f"(sous {self.m_off:+.2f}). Un actif qui coûte plus cher à laisser à "
                "l'arrêt qu'à redémarrer ne doit jamais s'arrêter — l'optimum est de "
                "tourner en continu, et une règle de curtailment n'a rien à optimiser ici."
            )
        return (
            f"La frontière optimale n'est pas un seuil mais une bande : on arrête à "
            f"{self.m_off:+.2f} et on ne redémarre qu'à {self.m_on:+.2f}, soit "
            f"{self.width:.2f} d'hystérésis. Une règle « marge < 0 » arrête trop tôt et "
            "redémarre trop tôt, deux fois par cycle."
        )


def solve_hysteresis(
    ou: OUParams,
    *,
    cost_restart: float,
    cost_shutdown: float,
    cost_idle: float,
    cost_fixed: float = 0.0,
    discount_rate: float = 0.08,
    grid_points: int = 401,
    grid_span_sigmas: float = 4.0,
    max_iterations: int = 5_000,
    tolerance: float = 1e-8,
) -> HysteresisBand:
    """Itération de la valeur sur deux états, et extraction de la bande d'hystérésis.

    L'espérance conditionnelle sous l'OU est calculée par quadrature gaussienne discrète :
    depuis `M`, la marge suivante est normale de moyenne `theta + (M - theta) e^{-kappa dt}`
    et d'écart-type `sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa))`.
    """
    if ou.kappa <= 0:
        raise PlantOptionError("kappa doit être > 0 pour une marge moyenne-réversive")
    if min(cost_restart, cost_shutdown, cost_idle) < 0:
        raise PlantOptionError("les coûts de transition et de maintien doivent être >= 0")

    span = grid_span_sigmas * ou.sigma / np.sqrt(2.0 * ou.kappa)
    grid = np.linspace(ou.theta - span, ou.theta + span, grid_points)

    decay = np.exp(-ou.kappa * ou.dt)
    conditional_mean = ou.theta + (grid - ou.theta) * decay
    conditional_std = ou.sigma * np.sqrt((1.0 - decay**2) / (2.0 * ou.kappa))
    if conditional_std <= 0:
        raise PlantOptionError("écart-type conditionnel nul — paramètres OU dégénérés")

    # matrice de transition : ligne = état courant sur la grille, colonne = état suivant
    difference = grid[None, :] - conditional_mean[:, None]
    weights = np.exp(-0.5 * (difference / conditional_std) ** 2)
    transition = weights / weights.sum(axis=1, keepdims=True)

    discount = np.exp(-discount_rate * ou.dt)
    value_on = np.maximum(grid - cost_fixed, 0.0)
    value_off = np.zeros_like(grid)

    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        expected_on = transition @ value_on
        expected_off = transition @ value_off

        new_on = np.maximum(
            grid - cost_fixed + discount * expected_on,
            -cost_shutdown + discount * expected_off,
        )
        new_off = np.maximum(
            -cost_idle + discount * expected_off,
            -cost_restart + discount * expected_on,
        )
        gap = max(np.max(np.abs(new_on - value_on)), np.max(np.abs(new_off - value_off)))
        value_on, value_off = new_on, new_off
        if gap < tolerance:
            converged = True
            break

    # frontières : plus haute marge où l'on choisit d'arrêter, plus basse où l'on redémarre
    stop = (-cost_shutdown + discount * (transition @ value_off)) >= (
        grid - cost_fixed + discount * (transition @ value_on)
    )
    restart = (-cost_restart + discount * (transition @ value_on)) >= (
        -cost_idle + discount * (transition @ value_off)
    )
    m_off = float(grid[stop].max()) if stop.any() else float(grid[0])
    m_on = float(grid[restart].min()) if restart.any() else float(grid[-1])

    return HysteresisBand(
        m_off=m_off,
        m_on=m_on,
        grid=grid,
        value_on=value_on,
        value_off=value_off,
        n_iterations=iteration,
        converged=converged,
    )


# ===========================================================================
# Comparaison à la règle heuristique
# ===========================================================================
@dataclass(frozen=True)
class RuleComparison:
    """Ce que la règle « marge < 0 pendant N mois » coûte face à la frontière optimale."""

    n_shutdowns_heuristic: int
    n_shutdowns_optimal: int
    false_shutdowns: int
    heuristic_threshold: float
    consecutive_periods: int

    @property
    def headline(self) -> str:
        return (
            f"La règle « marge < {self.heuristic_threshold:g} pendant "
            f"{self.consecutive_periods} périodes » déclenche "
            f"{self.n_shutdowns_heuristic} arrêts contre {self.n_shutdowns_optimal} pour la "
            f"frontière optimale, dont {self.false_shutdowns} que la frontière n'aurait pas "
            "faits. Chacun paie un arrêt et un redémarrage pour rien."
        )


def compare_to_heuristic(
    margin: pd.Series,
    band: HysteresisBand,
    *,
    threshold: float = 0.0,
    consecutive_periods: int = 4,
) -> RuleComparison:
    """Backteste la règle heuristique contre la frontière d'hystérésis.

    Le contraste est le produit de la page : une règle à seuil arrête sur des creux
    passagers que la bande absorbe, et chaque arrêt évitable paie un `K_off` plus un
    `K_on`.
    """
    clean = pd.Series(margin).dropna().astype(float)
    below = clean < threshold
    heuristic_episodes = regime_runs(below, depth=clean, min_obs=consecutive_periods)

    optimal_episodes = regime_runs(clean < band.m_off, depth=clean, min_obs=1)
    n_heuristic = len(heuristic_episodes)
    n_optimal = len(optimal_episodes)
    return RuleComparison(
        n_shutdowns_heuristic=n_heuristic,
        n_shutdowns_optimal=n_optimal,
        false_shutdowns=max(0, n_heuristic - n_optimal),
        heuristic_threshold=threshold,
        consecutive_periods=consecutive_periods,
    )


# ===========================================================================
# Simulateur de politique d'exploitation — ce que la règle COÛTE réellement
# ===========================================================================
@dataclass(frozen=True)
class PolicyResult:
    """P&L complet d'une politique d'arrêt-redémarrage, poste par poste.

    Les trois postes sont séparés parce qu'ils ne se compensent pas de la même façon :
    l'exploitation dépend du marché, les coûts de switch dépendent de la *fréquence* des
    décisions, et le coût de maintien à l'arrêt dépend de leur *durée*. Une règle peut
    être bonne sur un poste et mauvaise sur un autre.
    """

    label: str
    operating_pnl: float
    switching_cost: float
    idle_cost: float
    n_stops: int
    n_starts: int
    periods_on: int
    periods_off: int
    stop_margins: list[float]
    start_margins: list[float]
    state: pd.Series

    @property
    def total_pnl(self) -> float:
        return self.operating_pnl - self.switching_cost - self.idle_cost

    @property
    def effective_m_off(self) -> float:
        """Niveau de marge auquel la règle arrête RÉELLEMENT, en médiane.

        Pour une règle à persistance (« sous le seuil pendant N périodes »), ce niveau
        n'est pas le seuil : le temps d'attente laisse la marge descendre. C'est cette
        différence qui permet de comparer une règle de persistance à une bande.
        """
        return float(np.median(self.stop_margins)) if self.stop_margins else float("nan")

    @property
    def effective_m_on(self) -> float:
        return float(np.median(self.start_margins)) if self.start_margins else float("nan")

    @property
    def effective_width(self) -> float:
        return self.effective_m_on - self.effective_m_off


def simulate_operating_policy(
    margin: pd.Series,
    *,
    label: str,
    stop_rule,
    start_rule,
    cost_restart: float,
    cost_shutdown: float,
    cost_idle: float,
    start_on: bool = True,
) -> PolicyResult:
    """Fait tourner une politique d'exploitation sur un chemin de marge réel.

    `stop_rule(i, values, state_history)` et `start_rule(...)` renvoient un booléen. On
    passe l'historique complet plutôt que la valeur courante seule, parce qu'une règle de
    persistance a besoin de regarder en arrière — et c'est précisément ce qui la distingue
    d'une bande.
    """
    values = pd.Series(margin).dropna().astype(float)
    if values.empty:
        raise PlantOptionError("marge vide")

    on = start_on
    operating = switching = idle = 0.0
    n_stops = n_starts = periods_on = periods_off = 0
    stop_margins: list[float] = []
    start_margins: list[float] = []
    states: list[bool] = []

    array = values.to_numpy()
    for i in range(len(array)):
        if on:
            operating += array[i]
            periods_on += 1
            if stop_rule(i, array):
                switching += cost_shutdown
                stop_margins.append(float(array[i]))
                n_stops += 1
                on = False
        else:
            idle += cost_idle
            periods_off += 1
            if start_rule(i, array):
                switching += cost_restart
                start_margins.append(float(array[i]))
                n_starts += 1
                on = True
        states.append(on)

    return PolicyResult(
        label=label,
        operating_pnl=operating,
        switching_cost=switching,
        idle_cost=idle,
        n_stops=n_stops,
        n_starts=n_starts,
        periods_on=periods_on,
        periods_off=periods_off,
        stop_margins=stop_margins,
        start_margins=start_margins,
        state=pd.Series(states, index=values.index, name=f"on_{label}"),
    )


def _persistence_rules(threshold: float, n_periods: int):
    """Règle « sous le seuil pendant N périodes consécutives », et sa symétrique."""

    def stop_rule(i: int, values: np.ndarray) -> bool:
        if i + 1 < n_periods:
            return False
        return bool(np.all(values[i + 1 - n_periods : i + 1] < threshold))

    def start_rule(i: int, values: np.ndarray) -> bool:
        if i + 1 < n_periods:
            return False
        return bool(np.all(values[i + 1 - n_periods : i + 1] > threshold))

    return stop_rule, start_rule


def _band_rules(m_off: float, m_on: float):
    """Règle de bande : instantanée des deux côtés, l'hystérésis est dans les niveaux."""

    def stop_rule(i: int, values: np.ndarray) -> bool:
        return bool(values[i] < m_off)

    def start_rule(i: int, values: np.ndarray) -> bool:
        return bool(values[i] > m_on)

    return stop_rule, start_rule


def _never_stop_rules():
    """Politique dégénérée : on ne s'arrête jamais (voir `HysteresisBand.is_degenerate`)."""
    return (lambda i, v: False), (lambda i, v: False)


def run_heuristic_policy(
    margin: pd.Series, *, threshold: float = 0.0, n_periods: int = 4, **costs
) -> PolicyResult:
    """La règle utilisée sur les pages zinc et lithium : `consecutive_below(margin, 0, N)`."""
    stop_rule, start_rule = _persistence_rules(threshold, n_periods)
    return simulate_operating_policy(
        margin, label=f"heuristique N={n_periods}", stop_rule=stop_rule,
        start_rule=start_rule, **costs,
    )


def run_band_policy(margin: pd.Series, band: HysteresisBand, **costs) -> PolicyResult:
    """La frontière d'exercice calibrée.

    **Refuse de tourner sur une bande dégénérée.** Appliquer `M_on < M_off` tel quel
    ferait osciller l'usine — elle s'arrêterait sous `M_off` puis redémarrerait
    immédiatement puisque le même niveau est déjà au-dessus de `M_on`, en payant un
    aller-retour à chaque période. Et lui substituer une politique de repli (« ne jamais
    s'arrêter », par exemple) reviendrait à inventer une règle que le modèle n'a pas
    produite, puis à la comparer comme si c'était sa recommandation.

    Le cas se produit quand les coûts de switch deviennent petits devant la volatilité
    conditionnelle de la marge : il n'y a alors pas de bande propre, seulement du
    chattering. C'est un résultat sur le problème, pas un incident de calcul.
    """
    if band.is_degenerate:
        raise PlantOptionError(
            f"bande dégénérée (M_off={band.m_off:+.2f} > M_on={band.m_on:+.2f}) : à ces "
            "coûts de switch, la friction est trop faible devant la volatilité de la "
            "marge pour qu'une frontière d'exercice existe. Aucune politique de bande "
            "n'est simulable — et lui substituer un repli inventé fausserait la comparaison."
        )
    stop_rule, start_rule = _band_rules(band.m_off, band.m_on)
    return simulate_operating_policy(
        margin, label="bande d'hystérésis", stop_rule=stop_rule, start_rule=start_rule, **costs,
    )


def run_always_on_policy(margin: pd.Series, **costs) -> PolicyResult:
    """Contrefactuel : l'usine ne s'arrête jamais.

    Indispensable pour savoir si la flexibilité vaut quelque chose **du tout** sur cette
    période : si la meilleure règle ne bat pas « ne jamais s'arrêter », toute la
    discussion sur la frontière d'exercice est théorique.
    """
    return simulate_operating_policy(
        margin, label="jamais d'arrêt", stop_rule=lambda i, v: False,
        start_rule=lambda i, v: False, **costs,
    )


# ===========================================================================
# LE LIVRABLE — inverser la question
# ===========================================================================
@dataclass(frozen=True)
class ImpliedSwitchingCost:
    """Le coût de redémarrage que la règle à seuil suppose implicitement.

    C'est le chiffre qui rend la règle contestable. « Marge < 0 pendant 4 mois » n'est
    pas une hypothèse neutre : sur un chemin de marge donné, elle arrête et redémarre à
    des niveaux précis, donc elle est équivalente à une bande, donc elle suppose un coût
    de switch précis. Un exploitant sait si ce coût ressemble au sien.
    """

    effective_m_off: float
    effective_m_on: float
    effective_width: float
    implied_switching_cost: float
    searched_lo: float
    searched_hi: float
    converged: bool
    n_stops_observed: int

    @property
    def headline(self) -> str:
        if not self.converged:
            return (
                f"La règle arrête en médiane à {self.effective_m_off:+.2f} et redémarre à "
                f"{self.effective_m_on:+.2f} (largeur {self.effective_width:.2f}), mais aucun "
                f"coût de switch dans [{self.searched_lo:g}, {self.searched_hi:g}] ne "
                "reproduit cette bande — la règle n'est équivalente à aucune frontière "
                "d'exercice rationnelle sur cette plage."
            )
        return (
            f"La règle arrête en médiane à {self.effective_m_off:+.2f} et redémarre à "
            f"{self.effective_m_on:+.2f}. Une frontière d'exercice qui produirait la même "
            f"bande suppose un coût d'arrêt-redémarrage de **{self.implied_switching_cost:,.2f} "
            f"par unité de marge**. C'est le chiffre que la règle suppose sans le dire."
        )


def implied_switching_cost(
    margin: pd.Series,
    ou: OUParams,
    *,
    threshold: float = 0.0,
    n_periods: int = 4,
    cost_idle: float = 0.0,
    restart_share: float = 0.67,
    lo: float = 1e-4,
    hi: float = 1e4,
    n_grid: int = 24,
    **solve_kwargs,
) -> ImpliedSwitchingCost:
    """Quel coût de redémarrage rendrait la règle `marge < seuil pendant N` optimale ?

    Méthode : on fait tourner la règle sur le chemin de marge réel pour relever la bande
    qu'elle implémente **de fait** (médiane des marges d'arrêt et de redémarrage), puis on
    cherche le coût de switch dont la frontière d'exercice calibrée reproduit cette
    largeur de bande. La largeur croît de façon monotone avec le coût de switch, donc une
    recherche sur grille logarithmique suffit et reste lisible.

    `restart_share` répartit le coût total entre redémarrage et arrêt — un redémarrage
    coûte typiquement plus cher qu'un arrêt sur un actif thermique, d'où 2/3 par défaut.
    Le résultat est renvoyé en coût **total** de l'aller-retour, qui est la grandeur
    qu'un exploitant connaît.
    """
    heuristic = run_heuristic_policy(
        margin, threshold=threshold, n_periods=n_periods,
        cost_restart=0.0, cost_shutdown=0.0, cost_idle=cost_idle,
    )
    if heuristic.n_stops == 0 or not heuristic.start_margins:
        return ImpliedSwitchingCost(
            effective_m_off=heuristic.effective_m_off,
            effective_m_on=heuristic.effective_m_on,
            effective_width=float("nan"),
            implied_switching_cost=float("nan"),
            searched_lo=lo, searched_hi=hi, converged=False,
            n_stops_observed=heuristic.n_stops,
        )

    target_width = heuristic.effective_width
    grid = np.geomspace(lo, hi, n_grid)
    costs_kept: list[float] = []
    widths: list[float] = []
    for total_cost in grid:
        band = solve_hysteresis(
            ou,
            cost_restart=total_cost * restart_share,
            cost_shutdown=total_cost * (1.0 - restart_share),
            cost_idle=cost_idle,
            **solve_kwargs,
        )
        # Les bandes dégénérées (M_on < M_off) portent une largeur négative : les laisser
        # entrer dans l'interpolation ferait passer une courbe monotone par des points qui
        # ne représentent aucune bande. On les écarte, et si tout est dégénéré on le dit.
        if band.is_degenerate:
            continue
        costs_kept.append(float(total_cost))
        widths.append(band.width)

    if not widths:
        return ImpliedSwitchingCost(
            effective_m_off=heuristic.effective_m_off,
            effective_m_on=heuristic.effective_m_on,
            effective_width=target_width,
            implied_switching_cost=float("nan"),
            searched_lo=lo, searched_hi=hi, converged=False,
            n_stops_observed=heuristic.n_stops,
        )

    widths_array = np.asarray(widths)
    grid = np.asarray(costs_kept)
    if target_width < widths_array.min() or target_width > widths_array.max():
        return ImpliedSwitchingCost(
            effective_m_off=heuristic.effective_m_off,
            effective_m_on=heuristic.effective_m_on,
            effective_width=target_width,
            implied_switching_cost=float("nan"),
            searched_lo=lo, searched_hi=hi, converged=False,
            n_stops_observed=heuristic.n_stops,
        )

    implied = float(np.interp(target_width, widths_array, grid))
    return ImpliedSwitchingCost(
        effective_m_off=heuristic.effective_m_off,
        effective_m_on=heuristic.effective_m_on,
        effective_width=target_width,
        implied_switching_cost=implied,
        searched_lo=lo, searched_hi=hi, converged=True,
        n_stops_observed=heuristic.n_stops,
    )


@dataclass(frozen=True)
class PolicyComparison:
    """Les trois politiques sur le même chemin de marge, en P&L complet.

    `band` vaut None quand la frontière est dégénérée : on ne compare alors que la règle
    à seuil et le contrefactuel, en le disant.
    """

    heuristic: PolicyResult
    band: PolicyResult | None
    always_on: PolicyResult
    cost_restart: float
    cost_shutdown: float
    band_error: str = ""

    @property
    def band_is_available(self) -> bool:
        return self.band is not None

    @property
    def gap_vs_band(self) -> float:
        """Ce que la règle à seuil coûte face à la frontière calibrée."""
        if self.band is None:
            return float("nan")
        return self.band.total_pnl - self.heuristic.total_pnl

    @property
    def heuristic_flexibility_value(self) -> float:
        """Ce que la règle à seuil gagne face à une usine qui ne s'arrête jamais.

        Toujours calculable, même sans bande — et c'est la première chose à regarder :
        si elle est négative, s'arrêter détruit de la valeur sur cette période.
        """
        return self.heuristic.total_pnl - self.always_on.total_pnl

    @property
    def flexibility_value(self) -> float:
        """Ce que la meilleure règle disponible vaut face au contrefactuel."""
        if self.band is None:
            return self.heuristic_flexibility_value
        return self.band.total_pnl - self.always_on.total_pnl

    @property
    def headline(self) -> str:
        if self.band is None:
            return (
                f"Pas de frontière d'exercice à ces coûts — {self.band_error} "
                f"Reste comparable : la règle à seuil bat « ne jamais s'arrêter » de "
                f"{self.heuristic_flexibility_value:+,.1f} par unité, sur "
                f"{self.heuristic.n_stops} arrêts."
            )
        if self.flexibility_value <= 0:
            return (
                f"Sur cette période, aucune règle d'arrêt ne bat « ne jamais s'arrêter » "
                f"({self.flexibility_value:+,.1f} pour la meilleure) : la marge ne reste "
                "jamais assez longtemps assez bas pour que le coût de redémarrage se "
                "rentabilise. La frontière d'exercice est théorique ici, et il faut le dire."
            )
        return (
            f"La frontière calibrée bat la règle à seuil de {self.gap_vs_band:+,.1f} par "
            f"unité, et bat « ne jamais s'arrêter » de {self.flexibility_value:+,.1f}. "
            f"L'écart vient de {self.heuristic.n_stops} arrêts contre "
            f"{self.band.n_stops} : chaque arrêt évitable paie un aller-retour de "
            f"{self.cost_restart + self.cost_shutdown:,.2f}."
        )

    def to_frame(self) -> pd.DataFrame:
        results = [self.heuristic, self.always_on]
        if self.band is not None:
            results.insert(1, self.band)
        rows = []
        for result in results:
            rows.append(
                {
                    "politique": result.label,
                    "P&L total": result.total_pnl,
                    "exploitation": result.operating_pnl,
                    "coûts de switch": -result.switching_cost,
                    "coût de maintien": -result.idle_cost,
                    "arrêts": result.n_stops,
                    "redémarrages": result.n_starts,
                    "périodes à l'arrêt": result.periods_off,
                }
            )
        return pd.DataFrame(rows)


def compare_policies(
    margin: pd.Series,
    band: HysteresisBand,
    *,
    cost_restart: float,
    cost_shutdown: float,
    cost_idle: float,
    threshold: float = 0.0,
    n_periods: int = 4,
) -> PolicyComparison:
    """Les trois politiques sur le même chemin, en P&L complet et comparable.

    Si la bande est dégénérée, la comparaison se poursuit sans elle plutôt que de
    s'interrompre : la règle à seuil contre le contrefactuel reste une information utile.
    """
    costs = dict(cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle)
    band_result: PolicyResult | None
    band_error = ""
    try:
        band_result = run_band_policy(margin, band, **costs)
    except PlantOptionError as error:
        band_result = None
        band_error = str(error)

    return PolicyComparison(
        heuristic=run_heuristic_policy(margin, threshold=threshold, n_periods=n_periods, **costs),
        band=band_result,
        always_on=run_always_on_policy(margin, **costs),
        cost_restart=cost_restart,
        cost_shutdown=cost_shutdown,
        band_error=band_error,
    )


def switching_cost_sensitivity(
    margin: pd.Series,
    ou: OUParams,
    *,
    cost_grid: np.ndarray | None = None,
    cost_idle: float = 0.0,
    restart_share: float = 0.67,
    threshold: float = 0.0,
    n_periods: int = 4,
    **solve_kwargs,
) -> pd.DataFrame:
    """Largeur de bande et écart de P&L en fonction du coût de switch.

    C'est la sensibilité qui décide : elle montre à partir de quel coût de redémarrage la
    règle à seuil devient réellement coûteuse, et donc si le débat vaut la peine d'être
    porté à un exploitant.
    """
    grid = (
        np.geomspace(0.01, 100.0, 15) if cost_grid is None else np.asarray(cost_grid)
    )
    rows = []
    for total_cost in grid:
        restart = total_cost * restart_share
        shutdown = total_cost * (1.0 - restart_share)
        band = solve_hysteresis(
            ou, cost_restart=restart, cost_shutdown=shutdown, cost_idle=cost_idle, **solve_kwargs
        )
        comparison = compare_policies(
            margin, band, cost_restart=restart, cost_shutdown=shutdown,
            cost_idle=cost_idle, threshold=threshold, n_periods=n_periods,
        )
        rows.append(
            {
                "switching_cost": float(total_cost),
                "m_off": band.m_off,
                "m_on": band.m_on,
                # Une largeur negative ne veut rien dire : sur une bande degeneree on
                # affiche NaN et on marque la ligne, plutot que de laisser un nombre
                # negatif se faire lire comme une bande etroite.
                "band_width": float("nan") if band.is_degenerate else band.width,
                "degenerate": band.is_degenerate,
                "gap_vs_heuristic": comparison.gap_vs_band,
                "flexibility_value": comparison.flexibility_value,
                "n_stops_heuristic": comparison.heuristic.n_stops,
                "n_stops_band": (
                    comparison.band.n_stops if comparison.band is not None else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def volatility_sensitivity(
    ou: OUParams,
    *,
    sigma_multipliers: np.ndarray | None = None,
    **solve_kwargs,
) -> pd.DataFrame:
    """La démonstration contre-intuitive : la valeur de l'usine **croît** avec sigma.

    À moyenne de marge égale, une usine dont la marge est plus volatile vaut plus, parce
    que la flexibilité d'arrêt tronque la queue basse. C'est ce qui donne un chiffre à un
    débat qui se tient d'habitude en slogans.
    """
    multipliers = (
        np.array([0.5, 0.75, 1.0, 1.5, 2.0])
        if sigma_multipliers is None
        else np.asarray(sigma_multipliers)
    )
    rows = []
    for multiplier in multipliers:
        scaled = OUParams(
            kappa=ou.kappa,
            theta=ou.theta,
            sigma=ou.sigma * float(multiplier),
            dt=ou.dt,
            stationarity=ou.stationarity,
            n_obs=ou.n_obs,
        )
        band = solve_hysteresis(scaled, **solve_kwargs)
        rows.append(
            {
                "sigma_multiplier": float(multiplier),
                "sigma": scaled.sigma,
                "m_off": band.m_off,
                "m_on": band.m_on,
                "band_width": band.width,
                "value_at_theta": band.option_value_at(ou.theta),
            }
        )
    return pd.DataFrame(rows)


def real_board_crush_margin(*, start: str = "1990-07-18") -> pd.Series:
    """Marge de trituration board, entièrement réelle — CBOT soja/tourteau/huile.

    Contrairement au proxy énergie de T2-4 ou au roll omis de T1-2, ici les trois jambes
    de `board_crush_usd_bu` sont réelles sans aucun terme paramétré : c'est la marge que
    n'importe quel desk board lit sur son écran.
    """
    from agri.core.units import board_crush_usd_bu
    from agri.data.bloomberg_loader import load as load_bloomberg

    bean = load_bloomberg("cbot_soybean")
    meal = load_bloomberg("cbot_soymeal")
    oil = load_bloomberg("cbot_soyoil")
    frame = pd.concat({"bean": bean, "meal": meal, "oil": oil}, axis=1, sort=True).dropna()
    frame = frame.loc[start:]
    return board_crush_usd_bu(frame["bean"], frame["meal"], frame["oil"]).rename("board_crush_usd_bu")


@dataclass(frozen=True)
class RealMarginDiagnostic:
    """Ce que le test de stationnarité dit de la vraie marge — un résultat, pas un raté.

    La calibration OU suppose une marge moyenne-réversive. Le vérifier sur la vraie série
    plutôt que de le supposer est le sujet même de `core.stats.adf_kpss` : ici, il rend un
    verdict défavorable sur toute fenêtre testée (1990-2026 complet, et chaque sous-période
    depuis 2005), ce qui est en soi une découverte — la marge de crush réelle traverse de
    vraies ruptures de régime (Covid 2020, guerre en Ukraine 2022, révisions de mandat RVO)
    qu'un OU homogène sur toute la période ne peut pas représenter.
    """

    stationarity: StationarityVerdict
    window_start: str
    window_end: str
    n_obs: int

    @property
    def headline(self) -> str:
        return (
            f"Sur {self.window_start} → {self.window_end} ({self.n_obs} observations), "
            f"le verdict de stationnarité conjoint est « {self.stationarity.verdict} » : "
            "la marge de crush réelle ne se comporte pas comme un OU homogène sur cette "
            "période. Ce n'est pas un échec de calibration — c'est la preuve que le "
            "régime a changé au moins une fois (Covid, guerre en Ukraine, RVO), ce "
            "qu'aucun modèle à paramètres fixes ne peut absorber."
        )


def diagnose_real_margin_stationarity(margin: pd.Series) -> RealMarginDiagnostic:
    """Teste — plutôt que suppose — la stationnarité de la vraie marge (O-H1 appliquée
    à des données réelles, pas seulement à un jeu synthétique construit pour la vérifier)."""
    verdict = adf_kpss(margin, alpha=0.05)
    return RealMarginDiagnostic(
        stationarity=verdict,
        window_start=str(margin.index.min().date()),
        window_end=str(margin.index.max().date()),
        n_obs=len(margin),
    )


def calibrate_real_ou_indicative(margin: pd.Series) -> OUParams:
    """Calibration OU **indicative** sur donnée réelle non stationnaire (`strict=False`).

    À afficher systématiquement à côté de `diagnose_real_margin_stationarity` : les
    paramètres qui en sortent décrivent le régime moyen de la fenêtre choisie, pas une
    dynamique stable — c'est un résultat illustratif, pas une frontière d'exercice à
    suivre telle quelle.
    """
    return calibrate_ou(margin, strict=False)


__all__ = [
    "HysteresisBand",
    "ImpliedSwitchingCost",
    "OUParams",
    "PlantOptionError",
    "PolicyComparison",
    "PolicyResult",
    "RealMarginDiagnostic",
    "RuleComparison",
    "calibrate_ou",
    "calibrate_real_ou_indicative",
    "compare_policies",
    "compare_to_heuristic",
    "diagnose_real_margin_stationarity",
    "implied_switching_cost",
    "real_board_crush_margin",
    "run_always_on_policy",
    "run_band_policy",
    "run_heuristic_policy",
    "simulate_operating_policy",
    "solve_hysteresis",
    "switching_cost_sensitivity",
    "volatility_sensitivity",
]
