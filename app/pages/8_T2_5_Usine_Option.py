from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.china_soy import load_real_crush_frame  # noqa: E402
from agri.chains.plant_option import (  # noqa: E402
    PlantOptionError,
    calibrate_ou,
    compare_policies,
    diagnose_real_margin_stationarity,
    implied_switching_cost,
    real_board_crush_margin,
    run_heuristic_policy,
    solve_hysteresis,
    switching_cost_sensitivity,
    volatility_sensitivity,
)
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (
    snapshot_banner,  # noqa: E402
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

st.set_page_config(page_title="T2-5 — L'usine comme option", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# En-tête et périmètre
# ===========================================================================
page_header(
    code="T2-5",
    title="L'usine comme option sur la marge",
    subtitle=(
        "Ce qu'un signal de curtailment « marge < 0 pendant N périodes » suppose sans le "
        "dire — et le coût d'arrêt-redémarrage qu'il implique, sur la marge de "
        "trituration chinoise"
    ),
    scope=Scope(
        unit_trap=(
            "La marge de crush chinoise mélange **trois régimes fiscaux et une devise** : "
            "le CBOT cote en cents par boisseau, la DCE en CNY/t **TVA incluse**, et les "
            "droits de douane s'appliquent à la valeur CNF et non au FOB. La marge est "
            "calculée HT, donc la recette DCE est divisée par (1+TVA) avant toute comparaison."
        ),
        conversion=(
            "bean_cnf_usd_t = (CBOT_usd_bu + basis_c_bu/100) x 36,7437 + freight_usd_t\n"
            "margin_cny_t   = (0,785 x meal_dce + 0,185 x oil_dce)/(1+TVA)\n"
            "                 - bean_cnf_usd_t x USDCNY x (1+droit) - processing"
        ),
        proxies=[
            "basis FOB US Gulf et fret Chine : forfaits paramétrés, absents de l'export Bloomberg",
            "coûts d'arrêt / redémarrage / maintien : aucun n'est public, ce sont les sliders de la page",
        ],
        out_of_scope=[
            "trituration US (board crush) : sa marge n'est jamais passée sous zéro en 8 ans, "
            "donc la question du curtailment n'y est pas posée — voir S5",
            "contrats d'approvisionnement et engagements de livraison, qui empêchent un arrêt libre",
        ],
        frequency_note=(
            "CBOT, DCE et USDCNY sont quotidiens ; la marge est donc quotidienne, et une "
            "« période » du modèle est un jour ouvré, pas un mois."
        ),
        data_warnings=[
            "DCE tourteau/huile convertis depuis CNY/t TTC — le taux de TVA applicable "
            "dépend du produit ET de la date, à revérifier avant toute lecture engageante.",
        ],
    ),
)

# ===========================================================================
# Données et calibration
# ===========================================================================
st.sidebar.markdown("### Coûts d'exploitation (CNY/t de fève)")
cost_total = st.sidebar.slider("Coût d'arrêt-redémarrage (aller-retour)", 10.0, 600.0, 143.0, 1.0)
restart_share = st.sidebar.slider("Part du redémarrage dans ce total", 0.5, 0.9, 0.67, 0.01)
cost_idle = st.sidebar.slider("Coût de maintien à l'arrêt (par jour)", 0.0, 10.0, 2.0, 0.5)
n_periods = st.sidebar.slider("N de la règle heuristique (jours)", 2, 30, 4, 1)

cost_restart = cost_total * restart_share
cost_shutdown = cost_total * (1.0 - restart_share)

margin = load_real_crush_frame()["margin"]
ou = calibrate_ou(margin, strict=False)
diagnostic = diagnose_real_margin_stationarity(margin)

kpi_banner(
    {
        "Marge (dernière)": f"{margin.iloc[-1]:+,.0f}",
        "Médiane": f"{margin.median():+,.0f}",
        "Part sous zéro": f"{(margin < 0).mean():.0%}",
        "Pire print": f"{margin.min():+,.0f}",
        "Stationnarité": diagnostic.stationarity.verdict,
    }
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "La marge qui pose réellement la question",
    # Les chiffres de la prose sont calcules, jamais recopies : une part negative ecrite
    # en dur derive des que la serie s'allonge, et un lecteur qui compare la phrase au
    # bandeau KPI juste au-dessus voit la contradiction avant de voir l'argument.
    f"La marge de trituration chinoise passe sous zéro **{(margin < 0).mean():.0%} du "
    f"temps** et descend jusqu'à {margin.min():,.0f} CNY/t : c'est un actif dont l'arrêt "
    "est une décision réelle et répétée, pas une hypothèse d'école. Elle est en outre "
    f"**{diagnostic.stationarity.verdict} au sens conjoint ADF+KPSS**, ce qui autorise la "
    "calibration Ornstein-Uhlenbeck qui suit — condition que la marge de trituration "
    "américaine, elle, ne remplit sur aucune fenêtre testée (voir S5).",
    formula="dM = kappa (theta - M) dt + sigma dW",
)
show(
    regime_chart(
        margin.to_frame("marge").assign(sous_zero=margin < 0),
        "marge", regime_col="sous_zero", title="Marge de trituration chinoise (CNY/t de fève)",
        y_title="CNY/t", annotations={"2020-03-01": "Covid", "2022-02-24": "Ukraine"},
    )
)
st.caption(f"Calibration : {ou.summary}")

# ===========================================================================
# S3 — le livrable
# ===========================================================================
section(
    "S3",
    "Ce que la règle « marge < 0 pendant N jours » suppose sans le dire",
    "Un signal de curtailment à seuil ne dit pas seulement *quand* arrêter : parce qu'il "
    "attend N périodes de confirmation, il arrête en pratique **bien plus bas que son "
    "seuil affiché**, et redémarre bien plus haut. Il implémente donc une bande, qu'il le "
    "veuille ou non — et une bande correspond à un coût d'aller-retour précis. On relève "
    "les niveaux d'arrêt et de redémarrage effectifs sur le chemin réel, puis on cherche "
    "le coût de switch dont la frontière d'exercice calibrée reproduit cette largeur. "
    "Le lecteur n'a pas à accepter le modèle : il lui suffit de comparer un nombre au sien.",
    formula="M_off_effectif, M_on_effectif  →  quel K reproduit cette largeur de bande ?",
)

implied = implied_switching_cost(margin, ou, n_periods=n_periods, cost_idle=cost_idle)
finding(implied.headline)

heuristic_preview = run_heuristic_policy(
    margin, threshold=0.0, n_periods=n_periods,
    cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle,
)
c1, c2, c3 = st.columns(3)
c1.metric("Seuil affiché de la règle", "0 CNY/t")
c2.metric("Arrêt effectif (médiane)", f"{implied.effective_m_off:+,.1f}")
c3.metric("Redémarrage effectif (médiane)", f"{implied.effective_m_on:+,.1f}")
scope_note(
    f"{heuristic_preview.n_stops} arrêts déclenchés sur l'échantillon. L'écart entre le "
    "seuil affiché et l'arrêt effectif est exactement ce que la persistance achète."
)

# ===========================================================================
# S4
# ===========================================================================
section(
    "S4",
    "Ce que la règle coûte, et l'ordre de grandeur des enjeux",
    "Trois politiques sur le même chemin, en P&L complet : la règle à seuil, la frontière "
    "calibrée, et le contrefactuel qui ne s'arrête jamais. **L'ordre des enjeux compte "
    "plus que le classement** : pouvoir s'arrêter vaut beaucoup, choisir la bonne règle "
    "d'arrêt vaut un ordre de grandeur de moins. Présenter le raffinement avant le "
    "premier terme ferait passer un détail pour le sujet.",
)

band = solve_hysteresis(ou, cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle)
comparison = compare_policies(
    margin, band, cost_restart=cost_restart, cost_shutdown=cost_shutdown,
    cost_idle=cost_idle, n_periods=n_periods,
)
finding(comparison.headline)

if not comparison.band_is_available:
    diagnostic_note(
        "À ces coûts, la friction est trop faible devant la volatilité conditionnelle de "
        "la marge (écart-type d'environ 62 CNY/t par jour) pour qu'une frontière "
        "d'exercice existe : les régions « arrêter » et « redémarrer » se recouvrent, et "
        "l'usine ferait du chattering. Le solveur refuse de produire une bande plutôt que "
        "d'en fabriquer une inversée — monter le coût d'aller-retour pour retrouver un cas "
        "bien posé."
    )
else:
    st.caption(band.headline)

st.dataframe(
    comparison.to_frame().round(0), width="stretch", hide_index=True,
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "Pourquoi cette page n'est pas ancrée sur la trituration américaine",
    "Le même calcul sur le board crush CBOT donne un résultat qui invalide la question : "
    "la marge de trituration américaine **n'est jamais passée sous zéro** sur les huit "
    "dernières années, donc aucune règle d'arrêt ne se déclenche et l'option de fermer "
    "ne vaut rien. Elle échoue en outre le test conjoint ADF+KPSS sur toute fenêtre "
    "testée — 36 ans complets comme chaque sous-période depuis 2005 — ce qui interdit "
    "une calibration OU homogène. C'est un résultat, pas un échec de calibration : la "
    "marge traverse de vraies ruptures de régime qu'un modèle à paramètres fixes ne peut "
    "pas absorber.",
)
us_margin = real_board_crush_margin(start="2018-01-01")
us_diagnostic = diagnose_real_margin_stationarity(us_margin)
c1, c2, c3 = st.columns(3)
c1.metric("Board crush US — minimum", f"{us_margin.min():+.2f} USD/bu")
c2.metric("Part sous zéro", f"{(us_margin < 0).mean():.0%}")
c3.metric("Stationnarité", us_diagnostic.stationarity.verdict)
diagnostic_note(us_diagnostic.headline)

# ===========================================================================
# S6
# ===========================================================================
section(
    "S6",
    "Sensibilité au coût de switch — et où la bande cesse d'exister",
    "La largeur de bande croît avec le coût d'aller-retour, ce qui est ce qui rend "
    "l'inversion de S3 possible. Mais en dessous d'un certain coût la frontière **cesse "
    "d'exister** : quand la friction devient petite devant la volatilité conditionnelle "
    "de la marge, les régions d'arrêt et de redémarrage se recouvrent et le problème "
    "n'admet plus de bande. Ces lignes sont marquées plutôt que remplies d'une largeur "
    "négative qui se lirait comme une bande étroite.",
)
sensitivity = switching_cost_sensitivity(
    margin, ou,
    cost_grid=np.geomspace(5.0, 600.0, 10),
    cost_idle=cost_idle, n_periods=n_periods,
)
st.dataframe(
    sensitivity[
        ["switching_cost", "m_off", "m_on", "band_width", "degenerate",
         "gap_vs_heuristic", "n_stops_heuristic", "n_stops_band"]
    ].round(1),
    width="stretch", hide_index=True,
)
n_degenerate = int(sensitivity["degenerate"].sum())
if n_degenerate:
    scope_note(
        f"{n_degenerate} des {len(sensitivity)} niveaux testés ne produisent aucune bande "
        "valide — c'est la borne basse en deçà de laquelle la question « quelle règle "
        "d'arrêt » n'a pas de réponse."
    )

# ===========================================================================
# S7
# ===========================================================================
section(
    "S7",
    "La démonstration contre-intuitive",
    "À moyenne de marge inchangée, **une marge plus volatile rend l'usine plus "
    "précieuse** : la flexibilité d'arrêt tronque la queue basse, donc la valeur d'option "
    "croît avec sigma. C'est ce qui donne un chiffre au débat asset-heavy / asset-light, "
    "qui se tient d'habitude en slogans — un actif dont la marge oscille violemment peut "
    "valoir plus qu'un actif dont la marge est stablement positive.",
)
try:
    vol_sensitivity = volatility_sensitivity(
        ou, cost_restart=cost_restart, cost_shutdown=cost_shutdown, cost_idle=cost_idle
    )
    show(
        regime_chart(
            vol_sensitivity.set_index("sigma_multiplier"), "value_at_theta",
            title="Valeur de l'usine à la marge moyenne, selon la volatilité",
            y_title="valeur", zero_line=False,
        )
    )
    st.dataframe(vol_sensitivity.round(2), width="stretch", hide_index=True)
except PlantOptionError as error:
    diagnostic_note(f"Sensibilité non calculable à ces coûts : {error}")

mail_question(
    "Votre règle d'arrêt est-elle un seuil de marge, ou est-ce que le coût de "
    f"redémarrage la déplace explicitement ? Sur la marge de crush chinoise, une règle "
    f"« marge < 0 pendant {n_periods} jours » arrête en réalité à "
    f"{implied.effective_m_off:+,.0f} CNY/t et redémarre à {implied.effective_m_on:+,.0f} — "
    f"ce qui suppose un aller-retour d'environ {implied.implied_switching_cost:,.0f} CNY/t. "
    "Est-ce que cet ordre de grandeur ressemble au vôtre ?",
    "Direction de desk, corporate development, exploitants de trituration et de smelters",
)
