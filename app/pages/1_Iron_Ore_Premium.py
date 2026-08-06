"""Projet A — S1 à S6. `streamlit run app/Home.py` puis choisir cette page."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from freight.chains.ironore import (
    DEFAULT_MOISTURE_AUSTRALIA,
    DEFAULT_MOISTURE_BRAZIL,
    carry_cost_of_extra_voyage_days,
    decompose_premium,
    explained_variance,
    freight_hedge_effect,
    negative_residual_episodes,
)
from freight.ingest.fixture import SYNTHETIC_TICKERS, synthetic_ironore
from freight.ingest.loader import load_raw_directory
from freight.ingest.series import MissingSeries, coverage_report, to_series

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Mapping rôle -> ticker réel. À remplir quand les séries arrivent, en cohérence avec
# data_dictionary.csv. Tant que c'est vide, la page tourne en mode synthétique.
REAL_TICKERS: dict[str, str] = {
    # "p62": "...",
    # "p65": "...",
    # "c3":  "...",
    # "c5":  "...",
}

st.set_page_config(page_title="Iron ore 65-62 premium", layout="wide")


# --------------------------------------------------------------------------- données
@st.cache_data(show_spinner=False)
def load_data() -> tuple[dict[str, pd.Series], bool, list[str]]:
    """Renvoie ({rôle: série}, synthetic, avertissements)."""
    warnings: list[str] = []
    if REAL_TICKERS:
        try:
            raw = load_raw_directory(RAW_DIR)
            series = {role: to_series(raw, tk) for role, tk in REAL_TICKERS.items()}
            return series, False, warnings
        except (MissingSeries, ValueError) as exc:
            warnings.append(f"chargement des données réelles impossible : {exc}")

    raw = synthetic_ironore()
    series = {role: to_series(raw, tk) for role, tk in SYNTHETIC_TICKERS.items()}
    return series, True, warnings


series, synthetic, warnings = load_data()

st.title("Le premium 65-62 % Fe est en partie un spread de fret Capesize")
st.caption(
    "Décomposition du premium haute teneur en une part fret C3 − C5 corrigée de "
    "l'humidité et un résidu qualité / valeur-en-usage / tension / base FFA."
)

if synthetic:
    st.error(
        "**DONNÉES SYNTHÉTIQUES — NE RIEN INTERPRÉTER.** Cette page tourne sur un jeu "
        "de test généré aléatoirement, dont la structure a été imposée à la main. Elle "
        "prouve que le pipeline et les six sections fonctionnent, elle ne prouve rien "
        "sur le marché. Aucun chiffre de cette page ne doit sortir d'ici. Voir "
        "`FICHE_DONNEES.md` pour les quatre séries à fournir.",
        icon="🚫",
    )
for w in warnings:
    st.warning(w)

with st.sidebar:
    st.header("Hypothèses")
    moisture_br = st.slider(
        "Humidité fines brésiliennes (A-H2)",
        0.06, 0.11, DEFAULT_MOISTURE_BRAZIL, 0.005, format="%.3f",
    )
    moisture_au = st.slider(
        "Humidité fines australiennes (A-H2)",
        0.06, 0.11, DEFAULT_MOISTURE_AUSTRALIA, 0.005, format="%.3f",
    )
    st.caption(
        "Le fret est payé sur le poids embarqué (humide), les indices minerai sont "
        "cotés en tonne sèche. Corriger l'humidité augmente la part fret du premium."
    )
    st.divider()
    st.header("Sensibilités")
    extra_days = st.number_input("Jours de voyage supplémentaires Brésil", 0, 60, 25)
    annual_rate = st.slider("Taux de portage annuel", 0.0, 0.15, 0.06, 0.005)

decomp = decompose_premium(
    p65=series["p65"], p62=series["p62"], c3=series["c3"], c5=series["c5"],
    moisture_brazil=moisture_br, moisture_australia=moisture_au,
)

# ------------------------------------------------------------------- couverture data
with st.expander("Couverture des séries — à lire avant les graphiques", expanded=False):
    st.dataframe(coverage_report(series), use_container_width=True, hide_index=True)
    st.markdown(
        f"""
**Alignement.** La décomposition est calculée sur l'intersection des quatre calendriers :
**{len(decomp)} dates retenues**. Aucun trou n'a été comblé. Un jour où le fret ne cote
pas est un jour sans décomposition, pas un jour recopié depuis la veille.

**Base FFA (A-H3).** Si les séries de fret proviennent de FFA front-month plutôt que des
évaluations spot Baltic, il existe une base entre les deux. Elle n'est pas corrigée ici
et elle contamine le résidu.
"""
    )

# ------------------------------------------------------------------------------- S1
st.header("S1 — État actuel")
last = decomp.iloc[-1]
c1, c2, c3_, c4, c5_, c6 = st.columns(6)
c1.metric("P62 CFR", f"${last['p62']:,.1f}/dmt")
c2.metric("P65 CFR", f"${last['p65']:,.1f}/dmt")
c3_.metric("Premium 65-62", f"${last['premium_observed']:,.2f}/dmt")
c4.metric("Fair value fret", f"${last['freight_fair_value']:,.2f}/dmt")
c5_.metric("Résidu", f"${last['residual']:,.2f}/dmt")
share = last["freight_share"]
c6.metric("Part fret", "n/a" if pd.isna(share) else f"{100 * share:,.1f}%")

mean_share = decomp["freight_share"].mean()
naive_share = (decomp["premium_naive_freight"] / decomp["premium_observed"]).replace(
    [np.inf, -np.inf], np.nan
).mean()
st.markdown(
    f"""
Sur l'échantillon : part fret moyenne du premium **{100 * mean_share:,.1f} %** avec la
correction d'humidité, contre **{100 * naive_share:,.1f} %** sans. L'écart de
**{100 * (mean_share - naive_share):,.1f} point(s)** est exactement ce que coûte le fait
de confondre tonne humide et tonne sèche.
"""
)

# ------------------------------------------------------------------------------- S2
st.header("S2 — Décomposition du premium")
st.markdown(
    """
```
premium_observé = P65_CFR − P62_CFR
fair_value_fret = C3/(1−h_BR) − C5/(1−h_AU)
résidu          = premium_observé − fair_value_fret
```
Le résidu contient la qualité, la valeur-en-usage, la tension physique **et** la base
FFA. On ne le baptise pas « tension » : aucune donnée publique ne permet de séparer ces
quatre termes, et poser une valeur-en-usage inventée transformerait le résultat en
artefact.
"""
)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=decomp.index, y=decomp["freight_fair_value"], name="part fret (C3−C5, dmt)",
        mode="lines", stackgroup="one",
    )
)
fig.add_trace(
    go.Scatter(
        x=decomp.index, y=decomp["residual"].clip(lower=0),
        name="résidu (qualité + VIU + tension + base)", mode="lines", stackgroup="one",
    )
)
fig.add_trace(
    go.Scatter(
        x=decomp.index, y=decomp["premium_observed"], name="premium observé",
        mode="lines", line=dict(color="black", width=2),
    )
)
fig.update_layout(
    height=430, yaxis_title="USD/dmt", xaxis_title="date",
    legend=dict(orientation="h", y=-0.2),
    title="Premium 65-62 décomposé (aires empilées ; le résidu négatif sort de la pile)",
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Caveat : l'empilement ne représente les périodes de résidu négatif que par "
    "l'écart entre la ligne noire et la pile. Ces périodes sont l'objet de S3."
)

# ------------------------------------------------------------------------------- S3
st.header("S3 — Quand la qualité ne se paie plus")
episodes = negative_residual_episodes(decomp, min_days=5)
st.markdown(
    """
Un résidu négatif signifie que le premium haute teneur est **inférieur au seul surcoût de
distance** : le marché ne paie pas la qualité, ou la structure d'offre force le flux
malgré tout. C'est l'anomalie qui porte la conversation avec un desk.
"""
)
if episodes.empty:
    st.info("Aucun épisode de résidu négatif de 5 jours ou plus sur l'échantillon.")
else:
    st.dataframe(episodes, use_container_width=True, hide_index=True)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=decomp.index, y=decomp["residual"], name="résidu", mode="lines"))
fig3.add_hline(y=0, line_dash="dash")
fig3.update_layout(height=320, yaxis_title="USD/dmt", xaxis_title="date",
                   title="Résidu — passages sous zéro")
st.plotly_chart(fig3, use_container_width=True)

ev_levels = explained_variance(decomp["premium_observed"], decomp["freight_fair_value"])
ev_changes = explained_variance(
    decomp["premium_observed"], decomp["freight_fair_value"], on_changes=True
)
a, b = st.columns(2)
a.markdown(f"**En niveau** — {ev_levels.summary}")
b.markdown(f"**En variation** — {ev_changes.summary}")
st.caption(
    "Le R² en niveau est presque toujours le plus flatteur des deux et ne veut pas dire "
    "grand-chose sur des séries non stationnaires. Celui en variation est la lecture "
    "honnête. Les deux sont affichés parce que l'écart entre eux est une information."
)

# ------------------------------------------------------------------------------- S4
st.header("S4 — La boucle de rétroaction, testée sur des flux réels")
st.markdown(
    """
Le mécanisme à tester : le Brésil compétitif tire la demande en **tonne-mille** Capesize
(≈ 11 000 nm contre ≈ 1 600 nm), donc C3 monte, donc l'avantage brésilien s'auto-annule.
Si la boucle existe, la part brésilienne des importations chinoises doit précéder un
élargissement de C3 − C5.

**Série requise :** importations chinoises de minerai de fer par origine, douanes (GACC),
mensuel, gratuit. Elle n'est pas encore branchée.

**Limite déclarée à l'avance :** la mensualité des douanes détruit la granularité. Si le
test ne conclut pas, ce n'est pas un échec du mécanisme, c'est une limite de résolution —
et il vaut mieux l'annoncer maintenant que la découvrir après.
"""
)
st.info("Section en attente de la série de flux GACC.", icon="⏳")

# ------------------------------------------------------------------------------- S5
st.header("S5 — Couvrir la jambe fret")
hedge_opt = freight_hedge_effect(decomp["premium_observed"], decomp["freight_fair_value"])
hedge_unit = freight_hedge_effect(
    decomp["premium_observed"], decomp["freight_fair_value"], beta=1.0
)
h1, h2, h3 = st.columns(3)
h1.metric("Vol quotidienne non couverte", f"${hedge_opt.vol_unhedged:,.3f}/dmt")
h2.metric(
    f"Couverture optimale (β = {hedge_opt.beta:,.2f})",
    f"${hedge_opt.vol_hedged:,.3f}/dmt",
    f"{-hedge_opt.vol_reduction_pct:,.1f}%",
)
h3.metric(
    "Couverture unitaire (β = 1)",
    f"${hedge_unit.vol_hedged:,.3f}/dmt",
    f"{-hedge_unit.vol_reduction_pct:,.1f}%",
)
st.markdown(
    """
Lecture de desk : si couvrir la part fret par des FFA C3/C5 retire une fraction
significative de la volatilité du trade de spread 65-62, alors ce trade est en partie un
trade de fret, et il devrait être géré comme tel.

**Caveat, et il est lourd :** les FFA C3/C5 sont des contrats mensuels sur moyenne de
route. Couvrir une exposition quotidienne avec ça laisse une base non triviale, non
modélisée ici. Le chiffre ci-dessus est une borne supérieure de ce qu'une couverture
réelle obtiendrait, pas un résultat exécutable.
"""
)

# ------------------------------------------------------------------------------- S6
st.header("S6 — Sensibilités")
st.subheader("Humidité")
grid_br = np.arange(0.07, 0.105, 0.005)
grid_au = np.arange(0.07, 0.105, 0.005)
rows = []
for hb in grid_br:
    row = {"h_BR": f"{hb:.1%}"}
    for ha in grid_au:
        d = decompose_premium(
            p65=series["p65"], p62=series["p62"], c3=series["c3"], c5=series["c5"],
            moisture_brazil=float(hb), moisture_australia=float(ha),
        )
        row[f"h_AU {ha:.1%}"] = round(100 * d["freight_share"].mean(), 1)
    rows.append(row)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("Part fret moyenne du premium (%), selon les deux hypothèses d'humidité.")

st.subheader("Coût de portage des jours de voyage supplémentaires (A-H5)")
carry = carry_cost_of_extra_voyage_days(
    decomp["p62"].mean(), float(extra_days), float(annual_rate)
)
st.markdown(
    f"""
Sur une cargaison valorisée à la moyenne P62 de l'échantillon
(**${decomp['p62'].mean():,.1f}/dmt**), **{extra_days} jours** de mer supplémentaires à
**{annual_rate:.1%}** annuel coûtent **${carry:,.2f}/dmt**.

À comparer au résidu moyen de **${decomp['residual'].mean():,.2f}/dmt**. Petit devant le
différentiel de fret, pas nécessairement petit devant le résidu — ce qui est
précisément la raison de l'afficher plutôt que de le supposer négligeable.
"""
)

st.divider()
st.markdown(
    """
#### Ce que cette page ne fait pas

- **Aucun modèle de valeur-en-usage.** Coke rate, productivité haut-fourneau, pénalités
  silice/alumine : ce sont des données de procédé que je n'ai pas. Le résidu les absorbe
  et le dit.
- **Aucune correction de la base FFA-vs-index.**
- **Aucune décomposition de l'origine réelle des indices.** Le 62 % contient aussi du
  brésilien et de l'indien (A-H1), ce qui **sous-estime** la part fret. Le biais va dans
  le sens conservateur, ce qui est le bon sens pour une thèse qui affirme que la part
  fret est grande.
"""
)
