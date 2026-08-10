from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.freight_cf import (  # noqa: E402
    load_real_route_frame,
    market_implied_ballast_share,
)
from agri.core.voyage import ROUTES, VESSELS, VoyageParams, voyage_freight_usd_t  # noqa: E402
from agri.data.bloomberg_loader import DEFAULT_PATH, load  # noqa: E402
from page_template import (  # noqa: E402
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

st.set_page_config(page_title="T1-1 — Le fret dans le C&F", layout="wide")

if not DEFAULT_PATH.exists():
    st.error(f"Fichier Bloomberg introuvable : {DEFAULT_PATH}")
    st.stop()

st.sidebar.markdown("### Hypothèses de voyage")
speed_laden = st.sidebar.slider("Vitesse en charge (nœuds)", 10.0, 15.0, 12.5, 0.5)
port_days = st.sidebar.slider("Jours de port", 2.0, 12.0, 6.0, 0.5)
mgo_premium = st.sidebar.slider("Prime MGO sur VLSFO", 1.0, 1.8, 1.35, 0.05)

params = VoyageParams(speed_laden_kn=speed_laden, port_days=port_days)
spread = load_real_route_frame(params=params, mgo_premium=mgo_premium)
tce_2021 = load("p8_route_tce_2021")
boom_peak = float(tce_2021.max())

page_header(
    code="T1-1",
    title="Le fret dans le calcul C&F",
    subtitle=(
        "Le même taux de route publié dégage deux TCE séparés de 36 000 $/jour selon "
        "qu'on facture ou non le repositionnement à vide — et la donnée tranche"
    ),
    scope=Scope(
        unit_trap=(
            "Un taux de fret se cote soit en **USD/jour** (timecharter equivalent), soit "
            "en **USD/tonne** sur une route. Le passage de l'un à l'autre n'est pas une "
            "conversion mais une **estimation de voyage** — le facteur dépend de la "
            "distance, de la vitesse, des soutes et surtout de la part du repositionnement "
            "à vide qu'on impute au voyage. Ce facteur n'est pas une constante physique : "
            "c'est le désaccord lui-même."
        ),
        conversion=(
            "freight_usd_t = [ TCE x (D_laden + ballast_share x D_ballast + D_port)\n"
            "                  + soutes_mer + soutes_port + frais_port + canal ]\n"
            "                / ( cargo_t x (1 - commission) )"
        ),
        proxies=[
            "MGO reconstruit depuis le VLSFO Singapour (prime paramétrée) — l'export ne "
            "contient pas de série MGO séparée ; le poste pèse quelques pour cent du voyage",
            "distances et frais de port : ordres de grandeur codés, à revérifier avant "
            "tout usage engageant",
        ],
        out_of_scope=[
            "les jambes de prix de l'arb (FOB Santos, CIF Chine) : absentes de l'export, "
            "donc la page porte sur le terme de fret seul et pas sur l'arb complet",
            "démurrage et laytime — leur omission **sous-estime** le coût complet, donc "
            "le biais va contre la thèse",
        ],
        frequency_note=(
            "Taux de route P8 et VLSFO Singapour quotidiens, intersection stricte : aucun "
            "report en avant, le prix des soutes à la date de fixture étant précisément "
            "l'objet du désaccord."
        ),
        data_warnings=[
            "La cellule P8 de l'export contient DEUX régimes d'unité — voir S2. Seul le "
            "segment en USD/tonne alimente les calculs ; le segment en USD/jour sert de "
            "banc d'essai en S4.",
            "Un print de fret nul au 30/04/2022 est écarté : physiquement impossible.",
        ],
    ),
)

kpi_banner(
    {
        "Taux publié (USD/t)": f"{spread.route_rate_usd_t.iloc[-1]:,.1f}",
        "TCE si ballast = 0": f"{spread.tce_no_ballast.iloc[-1]:,.0f} $/j",
        "TCE si ballast = 1": f"{spread.tce_full_ballast.iloc[-1]:,.0f} $/j",
        "Écart": f"{spread.spread.iloc[-1]:,.0f} $/j",
        "Pic TCE réel 2021": f"{boom_peak:,.0f} $/j",
    }
)

# ===========================================================================
section(
    "S2",
    "Le défaut de donnée — et c'est le sujet de la page",
    "La cellule P8 de l'export contient **deux régimes d'unité dans la même série**. De "
    "juillet à octobre 2021 elle cote entre 24 500 et 38 000 : ce sont des **USD/jour**, "
    "un TCE Panamax au pic du boom vraquier. À partir du 18 novembre 2021 elle cote entre "
    "36 et 85 : ce sont des **USD/tonne**, un taux de voyage. Les deux segments n'ont "
    "aucune date commune et sont séparés par 19 jours de trou, ce qui interdit de "
    "calibrer le facteur de conversion sur la jonction — le marché a bougé entre les deux. "
    "Le fait que la source elle-même mélange USD/jour et USD/tonne n'est pas une "
    "coïncidence commode : c'est la confusion que cette page mesure, matérialisée dans la "
    "donnée avant même qu'on ait commencé à calculer.",
)
diagnostic_note(
    f"Segment TCE : {tce_2021.index.min():%d/%m/%Y} → {tce_2021.index.max():%d/%m/%Y}, "
    f"{len(tce_2021)} prints, {tce_2021.min():,.0f}–{tce_2021.max():,.0f} USD/jour. "
    f"Segment taux de voyage : {spread.route_rate_usd_t.index.min():%d/%m/%Y} → "
    f"{spread.route_rate_usd_t.index.max():%d/%m/%Y}, {len(spread.route_rate_usd_t)} prints, "
    f"{spread.route_rate_usd_t.min():,.1f}–{spread.route_rate_usd_t.max():,.1f} USD/t. "
    "Chaque segment est cohérent dans son unité ; c'est leur juxtaposition qui ne l'est pas."
)
show(
    regime_chart(
        spread.route_rate_usd_t.to_frame("taux publié"), "taux publié",
        title="P8 Santos → Qingdao, segment USD/tonne uniquement",
        y_title="USD/t", zero_line=False,
    )
)

# ===========================================================================
section(
    "S3",
    "Le même print, deux lectures",
    "Un desk trading lit « 55 $/t » et conclut que le marché paie 55 $/t. Un département "
    "fret lit le même print et demande : sur combien de jours ? Si le voyage inclut le "
    "repositionnement à vide, le même revenu doit couvrir presque deux fois plus de jours, "
    "donc le TCE qu'il dégage est presque deux fois plus faible. Ce n'est pas une "
    "divergence d'opinion sur le marché — les deux lisent le **même** chiffre — c'est une "
    "divergence sur ce que ce chiffre rémunère.",
    formula="TCE = ( freight_usd_t x cargo_payante − coûts de voyage ) / jours_de_cycle",
)
finding(spread.headline)
show(
    regime_chart(
        pd.DataFrame(
            {"TCE ballast = 0": spread.tce_no_ballast, "TCE ballast = 1": spread.tce_full_ballast}
        ).assign(dummy=False),
        "TCE ballast = 0", title="", y_title="", zero_line=False,
    )
    .add_scatter(
        x=spread.tce_full_ballast.index, y=spread.tce_full_ballast.values,
        name="TCE ballast = 1", mode="lines",
    )
    .update_layout(
        title="TCE impliqué par le taux publié, selon la convention de ballast",
        yaxis_title="USD/jour", showlegend=True,
    )
    .add_hline(
        y=boom_peak, line_dash="dash", line_color="crimson",
        annotation_text=f"pic TCE réel 2021 : {boom_peak:,.0f} $/j",
        annotation_position="top left",
    )
)

# ===========================================================================
section(
    "S4",
    "Ce que la donnée tranche",
    "Le désaccord paraît indécidable en théorie. Il ne l'est pas ici, parce que le segment "
    "de 2021 — celui qu'il a fallu isoler comme défaut de données en S2 — donne le niveau "
    "de TCE **réellement coté sur cette route**, au pic du boom vraquier. Il fournit donc "
    "un plafond de plausibilité, et il suffit à départager les deux lectures.",
)
share_above_no_ballast = float((spread.tce_no_ballast > boom_peak).mean())
share_above_full = float((spread.tce_full_ballast > boom_peak).mean())

c1, c2, c3 = st.columns(3)
c1.metric("Pic TCE réel (jul-oct 2021)", f"{boom_peak:,.0f} $/j")
c2.metric("Lecture sans ballast au-dessus du pic", f"{share_above_no_ballast:.0%} du temps")
c3.metric("Lecture avec ballast au-dessus du pic", f"{share_above_full:.0%} du temps")

finding(
    f"Lire le taux publié **sans facturer le ballast** implique un TCE supérieur au pic "
    f"du boom vraquier {share_above_no_ballast:.0%} du temps sur cinq ans — ce qui "
    "reviendrait à dire que le marché a passé cinq ans au-dessus de son propre sommet. "
    f"La même inversion **en facturant le ballast** ne dépasse ce pic que "
    f"{share_above_full:.0%} du temps. La lecture du desk trading n'est pas discutable : "
    "elle est arithmétiquement intenable."
)
scope_note(
    "Ce n'est pas un argument d'autorité en faveur du département fret : c'est une borne "
    "de plausibilité tirée d'un prix réellement coté sur la même route, et elle ne dit "
    "rien de la part exacte de ballast — seulement qu'elle n'est pas nulle."
)

# ===========================================================================
section(
    "S5",
    "La part de ballast que le marché price",
    "Une fois admis que le ballast est facturé, reste à savoir combien. En posant un TCE "
    "de référence — celui auquel les armateurs affrètent réellement — le taux de route "
    "publié révèle la part de repositionnement que le marché a déjà intégrée. La question "
    "au desk cesse alors d'être « qui a raison » et devient « le marché price X %, est-ce "
    "ce que vous facturez en interne ? ».",
    formula="chercher ballast_share tel que  modèle(TCE_référence, ballast_share) = taux_publié",
)
reference_tce = st.slider(
    "TCE de référence (USD/jour)", 8_000, 40_000, 18_000, 500,
    help="Le niveau auquel un armateur Panamax affrète réellement, à comparer au pic 2021 de "
         f"{boom_peak:,.0f}",
)
latest_rate = float(spread.route_rate_usd_t.iloc[-1])
latest_vlsfo = float(load("vlsfo_singapore").reindex(spread.route_rate_usd_t.index).ffill().iloc[-1])

implied = market_implied_ballast_share(
    latest_rate, float(reference_tce), latest_vlsfo, latest_vlsfo * mgo_premium,
    vessel=VESSELS["panamax"], route=ROUTES["santos_qingdao"], params=params,
)
finding(implied.headline)

grid = pd.DataFrame(
    {
        "ballast_share": [i / 20 for i in range(21)],
    }
)
grid["fret modélisé"] = [
    voyage_freight_usd_t(
        float(reference_tce), latest_vlsfo, latest_vlsfo * mgo_premium,
        vessel=VESSELS["panamax"], route=ROUTES["santos_qingdao"],
        params=params.with_ballast(share),
    ).freight_usd_t
    for share in grid["ballast_share"]
]
show(
    regime_chart(
        grid.set_index("ballast_share"), "fret modélisé",
        title=f"Fret modélisé selon la part de ballast, à TCE {reference_tce:,.0f} $/jour",
        y_title="USD/t", zero_line=False,
        reference_lines={f"taux publié {latest_rate:,.1f} $/t": latest_rate},
    )
)

mail_question(
    f"Sur Santos → Qingdao, le taux publié de {latest_rate:,.1f} $/t dégage un TCE de "
    f"{spread.tce_no_ballast.iloc[-1]:,.0f} $/jour si on ne facture aucun ballast, contre "
    f"{spread.tce_full_ballast.iloc[-1]:,.0f} si on le facture entièrement — et la "
    f"première lecture placerait le marché au-dessus de son pic de 2021 "
    f"{share_above_no_ballast:.0%} du temps. Quelle part de repositionnement votre taux "
    "interne facture-t-il réellement, et est-elle négociée avec le desk trading ou "
    "imposée par le département fret ?",
    "Desks fret (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra) et traders grains/oléagineux",
)
