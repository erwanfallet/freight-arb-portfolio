from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.sugar_mix import (  # noqa: E402
    CENTS_LB_TO_USD_T,
    CZARNIKOW_COST_BRL_T,
    DEFAULT_POL_FACTOR,
    floor_variance_decomposition,
    indifference_hydrous_brl_l,
    load_real_parity_frame,
    moving_floor,
    production_cost_check,
)
from agri.core.fmt import fr, fr_pct  # noqa: E402
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (
    snapshot_banner,  # noqa: E402
    SHUT_COLOR,
    Scope,
    diagnostic_note,
    finding,
    kpi_banner,
    mail_question,
    page_header,
    regime_chart,
    scope_note,
    section,
    show,
)

st.set_page_config(page_title="T3-2 — Sucre : le plancher qui bouge", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# En-tête et périmètre
# ===========================================================================
page_header(
    code="T3-2",
    title="Le « plancher de coût brésilien » est une série de change",
    subtitle=(
        "Un coût de production se libelle en réaux ; traduit en cents par livre pour un "
        "lecteur new-yorkais, il se met à bouger de vingt cents sans qu'aucun coût "
        "brésilien n'ait changé"
    ),
    scope=Scope(
        unit_trap=(
            "Le NY11 cote en **cents par livre**, en USD. Un coût de production brésilien "
            "se libelle en **BRL par tonne**, parce que c'est la monnaie dans laquelle les "
            "charges sont payées. Passer de l'un à l'autre demande deux conversions — livre "
            "vers tonne, puis USD vers BRL — et la seconde n'est pas une constante : c'est "
            "un prix de marché qui bouge tous les jours. Un « niveau structurel » exprimé "
            "dans la mauvaise monnaie cesse d'être structurel."
        ),
        conversion=(
            "sucre_BRL_t   = NY11_c_lb x 22,0462 x USDBRL\n"
            "plancher_c_lb = coût_BRL_t / (22,0462 x USDBRL)\n"
            "hydraté*_BRL_l = NY11_c_lb x pol_factor x USDBRL x 2,20462 x (ATR_h / ATR_s) / 100"
        ),
        proxies=[
            "coût de production : le chiffre de BRL 2 000/t vient de Czarnikow (juin 2026), "
            "il est **sourcé et daté** — mais il est régional et évolutif, d'où le slider",
            "pol_factor : ajuste le NY11 (96° pol) vers la qualité VHP réellement produite",
        ],
        out_of_scope=[
            "l'élasticité conditionnelle du mix, qui est la thèse d'origine du projet : elle "
            "demande le mix UNICA par quinzaine et région et l'éthanol CEPEA, **aucun des "
            "deux n'étant dans l'export** — voir S5, où la spécification est laissée telle "
            "quelle plutôt que simulée",
            "coûts de transport intérieur et distance au port, qui font partie du programme "
            "du moulin mais pas de l'identité de prix testée ici",
        ],
        frequency_note=(
            "NY11 et USDBRL sont quotidiens. UNICA publie par quinzaine — c'est pourquoi la "
            "partie mix de S5 changerait de fréquence, et pas seulement de données."
        ),
        data_warnings=[
            "Les coefficients Consecana sont **révisés chaque saison** (G-H1). Les figer "
            "fausserait tout l'historique, et l'erreur serait silencieuse puisque la parité "
            "resterait plausible.",
        ],
    ),
)

# ===========================================================================
# Paramètres
# ===========================================================================
st.sidebar.markdown("### Paramètres")
cost_brl_t = st.sidebar.slider(
    "Coût de production (BRL/tonne)", 1200.0, 3000.0, CZARNIKOW_COST_BRL_T, 50.0,
    help="Czarnikow, juin 2026 : les opportunités de pricing sont restées sous BRL 2 000/t.",
)
pol_factor = st.sidebar.slider("pol_factor (NY11 → VHP)", 0.94, 1.00, DEFAULT_POL_FACTOR, 0.005)
window_start = st.sidebar.selectbox(
    "Fenêtre", ["2015-01-01", "2010-01-01", "2000-01-01"], index=0
)

frame = load_real_parity_frame(window_start)
cost_check = production_cost_check(frame, cost_brl_t=cost_brl_t)
floor = moving_floor(frame, cost_brl_t=cost_brl_t)
decomposition = floor_variance_decomposition(frame)
hydrous = indifference_hydrous_brl_l(frame["ny11"], frame["usdbrl"], pol_factor=pol_factor)

kpi_banner(
    {
        "NY11 (dernier)": f"{fr(frame['ny11'].iloc[-1], 2)} c/lb",
        "USDBRL": fr(frame["usdbrl"].iloc[-1], 3),
        "Sucre en BRL": f"{fr(cost_check.last_brl_t, 0)} BRL/t",
        "Plancher implicite": f"{fr(floor.floor_last, 1)} c/lb",
        "Amplitude du plancher": f"{fr(floor.floor_range, 1)} c/lb",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Deux maisons, un désaccord daté, et ce sur quoi elles s'accordent",
    "**Hedgepoint** (février 2026) voit une récolte centre-sud à 630 Mt et un mix qui devrait "
    "tomber vers 46 % pour réduire matériellement l'excédent — mais qui n'y arrivera pas, "
    "parce que **les limites d'usine et le sucre déjà vendu à terme** l'en empêchent. "
    "**Czarnikow** (juin 2026) observe l'inverse du côté de la contrainte : les moulins "
    "entrent dans la saison **beaucoup moins couverts** que les quatre précédentes, et les "
    "opportunités de pricing 2026/27 sont restées **sous BRL 2 000/t, c'est-à-dire sous le "
    "coût de production**.\n\n"
    "Hedgepoint dit contraintes ; Czarnikow observe que le déblocage vient précisément d'un "
    "niveau de couverture inhabituellement bas. Mais les deux maisons s'accordent sur un "
    "point : **le degré de couverture préalable est la variable qui discrimine**. C'est "
    "observable, et c'est rarement modélisé.\n\n"
    "Cette page ne tranche pas entre elles. Elle fait deux choses plus utiles : elle vérifie "
    "l'affirmation chiffrée de Czarnikow sur les prix réels, et elle montre que la façon "
    "dont ce chiffre est habituellement traduit pour un lecteur new-yorkais est trompeuse.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "L'affirmation de Czarnikow, confrontée aux prix",
    "« Le pricing est resté sous le coût de production » est une affirmation vérifiable, et "
    "la vérifier vaut mieux que la citer. Le NY11 et l'USDBRL suffisent : le sucre en réaux "
    "par tonne est le produit des deux, à la conversion livre-tonne près.",
    formula="sucre_BRL_t = NY11_c_lb x 22,0462 x USDBRL",
)
finding(cost_check.headline)
show(
    regime_chart(
        floor.frame.assign(sous_cout=floor.frame["sugar_brl_t"] < cost_brl_t),
        "sugar_brl_t",
        regime_col="sous_cout",
        regime_color=SHUT_COLOR,
        title="Sucre exprimé en réaux par tonne",
        y_title="BRL/tonne",
        zero_line=False,
        reference_lines={f"coût {fr(cost_brl_t, 0)} BRL/t": cost_brl_t},
        annotations={"2026-01-01": "fenêtre Czarnikow"},
    )
)
scope_note(
    "Zones ombrées : le prix est sous le coût de production retenu. L'affirmation de "
    "Czarnikow porte sur la campagne 2026/27 ; le graphe montre qu'elle s'inscrit dans un "
    "régime plus large, pas dans un accident ponctuel."
)

# ===========================================================================
# S3 — LE RÉSULTAT
# ===========================================================================
section(
    "S3",
    "Le même coût, traduit en cents par livre, cesse d'être un niveau",
    "Un desk new-yorkais ne raisonne pas en réaux. Le coût brésilien lui est donc rapporté "
    "en cents par livre — « il y a du support vers 18 cents, c'est le coût brésilien » — et "
    "il l'entend comme un niveau structurel, un plancher que le marché aurait du mal à "
    "traverser durablement.\n\n"
    "Or cette traduction divise par le taux de change. Le coût en réaux peut rester "
    "**strictement constant** : son expression en cents bouge de tout ce que bouge l'USDBRL. "
    "Sur la période, le même coût de production produit un plancher allant de "
    f"{fr(floor.floor_min, 1)} à {fr(floor.floor_max, 1)} c/lb. Ce n'est pas un plancher "
    "qui s'est déplacé parce que les coûts brésiliens ont changé — **aucun coût n'a changé "
    "dans ce calcul, il est fixé par construction**. C'est un plancher qui s'est déplacé "
    "parce que le real s'est déprécié.",
    formula="plancher_c_lb = coût_BRL_t / (22,0462 x USDBRL)",
)
finding(floor.headline)
show(
    regime_chart(
        floor.frame,
        "floor_c_lb",
        regime_col="below_floor",
        regime_color=SHUT_COLOR,
        title="Plancher NY11 impliqué par un coût de production CONSTANT en réaux",
        y_title="cents par livre",
        zero_line=False,
        reference_lines={f"NY11 dernier : {fr(frame['ny11'].iloc[-1], 1)}": float(frame["ny11"].iloc[-1])},
    )
)
scope_note(
    "Zones ombrées : les périodes où le NY11 était effectivement sous le plancher. La "
    "courbe elle-même n'est **que** l'inverse de l'USDBRL, remise à l'échelle — c'est "
    "exactement le propos."
)

c1, c2, c3 = st.columns(3)
c1.metric("Plancher le plus bas", f"{fr(floor.floor_min, 1)} c/lb")
c2.metric("Plancher le plus haut", f"{fr(floor.floor_max, 1)} c/lb")
c3.metric("Amplitude", f"{fr(floor.floor_range, 1)} c/lb", delta="uniquement du change", delta_color="off")

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "Ce que le change fait au prix lui-même, et pourquoi ce n'est pas symétrique",
    "On pourrait objecter que le change bouge aussi le prix du sucre en réaux, donc que les "
    "deux effets se compensent. Ils ne se compensent pas de la même façon, et la "
    "décomposition de variance le dit.\n\n"
    f"Le mouvement du sucre exprimé en réaux vient à {fr_pct(decomposition['share_sugar'])} "
    f"du prix en dollars et à {fr_pct(decomposition['share_fx'])} du change, avec une "
    f"covariance de {fr_pct(decomposition['share_covariance'])} — la corrélation entre les "
    f"deux vaut {fr(decomposition['correlation'], 3)}, donc **négative** : quand le real se "
    "déprécie, le sucre en dollars tend à baisser. Le change amortit donc partiellement le "
    "prix reçu par le moulin.\n\n"
    "Mais le **plancher**, lui, ne bénéficie d'aucun amortissement : il est exactement "
    "proportionnel à l'inverse du change, par construction. C'est là qu'est l'asymétrie — "
    "le prix reçu est partiellement couvert par la corrélation, le seuil auquel on le "
    "compare ne l'est pas du tout.",
)
st.dataframe(
    pd.DataFrame(
        {
            "composante": ["prix du sucre (USD)", "change USDBRL", "covariance (×2)"],
            "part de la variance": [
                fr_pct(decomposition["share_sugar"], 1),
                fr_pct(decomposition["share_fx"], 1),
                fr_pct(decomposition["share_covariance"], 1),
            ],
        }
    ),
    width="stretch", hide_index=True,
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "Le nombre que le moulin regarde vraiment",
    "Un moulin brésilien n'arbitre pas contre un coût de production : il arbitre entre faire "
    "du sucre et faire de l'éthanol, sur la même canne. La grandeur qui commande cette "
    "décision est le prix d'éthanol auquel il devient indifférent — et elle se déduit du "
    "NY11 et du change seuls, via le barème Consecana, sans avoir besoin d'une série "
    "d'éthanol.\n\n"
    "C'est l'inversion de la chaîne de conversion habituelle : au lieu de partir d'un prix "
    "d'éthanol pour le comparer au sucre, on part du sucre et on demande à quel prix "
    "d'éthanol le moulin arrête d'en faire.",
    formula="hydraté* = NY11 x pol_factor x USDBRL x 2,20462 x (ATR_hydraté / ATR_sucre) / 100",
)
finding(
    f"Au dernier cours, un moulin est indifférent entre sucre et éthanol hydraté à "
    f"{fr(hydrous.iloc[-1], 2)} BRL par litre. Sur la période, ce seuil a évolué entre "
    f"{fr(hydrous.min(), 2)} et {fr(hydrous.max(), 2)} BRL/l."
)
show(
    regime_chart(
        hydrous.to_frame("hydrous_brl_l"),
        "hydrous_brl_l",
        title="Prix de l'éthanol hydraté qui rend le moulin indifférent",
        y_title="BRL par litre",
        zero_line=False,
    )
)
scope_note(
    "Le comparer au prix CEPEA effectif donnerait le parity gap, donc le signal de mix. "
    "CEPEA publie gratuitement — c'est la série la plus rentable à récupérer pour ce "
    "projet, et la seule qui manque pour boucler la thèse d'origine."
)

# ===========================================================================
# S6
# ===========================================================================
section(
    "S6",
    "Ce que cette page ne fait pas, et ce qu'il faudrait pour le faire",
    "La thèse d'origine du projet est plus ambitieuse que ce qui précède : elle porte sur "
    "l'**élasticité du mix à la parité, conditionnellement au taux de couverture d'entrée "
    "de saison** — c'est-à-dire précisément la variable sur laquelle Hedgepoint et "
    "Czarnikow convergent. Elle demande une estimation en panel par quinzaine UNICA et par "
    "région.\n\n"
    "Deux séries manquent, et aucune n'est payante : le **mix UNICA** par quinzaine et "
    "région, et l'**éthanol hydraté CEPEA**. La spécification est laissée ci-dessous telle "
    "qu'elle serait estimée, plutôt que simulée sur données fabriquées — un coefficient "
    "estimé sur un jeu synthétique n'apprend rien à personne, il vérifie seulement que le "
    "code retrouve ce qu'on y a mis.",
    formula=(
        "dmix = a_r + b1 parity_gap_{t-1}\n"
        "            + b2 (parity_gap_{t-1} x taux_de_couverture_entrée_r)\n"
        "            + b3 utilisation_capacité\n"
        "            + b4 (distance_port_r x parity_gap_{t-1}) + e\n"
        "\n"
        "L'objet d'intérêt est b2, jamais b1."
    ),
)
diagnostic_note(
    "Le moteur d'élasticité en panel existe et est testé (`estimate_mix_elasticity`), mais "
    "il tourne sur jeu synthétique. Il n'est volontairement pas affiché ici : une page qui "
    "montrerait un b2 fabriqué à côté de résultats mesurés sur données réelles inviterait à "
    "les lire de la même façon."
)

mail_question(
    f"Quand on traduit un coût de production de {fr(cost_brl_t, 0)} BRL/t en cents par "
    f"livre, on obtient un plancher qui a varié de {fr(floor.floor_min, 1)} à "
    f"{fr(floor.floor_max, 1)} c/lb depuis {window_start[:4]} — {fr(floor.floor_range, 1)} "
    "cents d'amplitude produits uniquement par l'USDBRL. Est-ce que votre équipe raisonne "
    "sur un plancher en cents, ou est-ce qu'elle le recalcule à chaque mouvement du real ? "
    "Et sur 2026/27, à quel taux de couverture d'entrée de saison vos moulins sont-ils "
    "réellement entrés ?",
    "Desks sucre de Sucden, Czarnikow, Alvean, Wilmar, ED&F Man ; trading et origination des groupes sucriers CS",
)
