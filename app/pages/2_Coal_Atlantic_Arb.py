"""Projet B — S1 à S6."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from freight.chains.coal import (
    BENCHMARK_CV_KCAL_PER_KG,
    DEFAULT_EMISSION_FACTOR,
    EXTRA_EU_SCOPE_FACTOR,
    ets_cost_per_cargo_tonne,
    freight_binding_test,
    ols,
    phase_in_factor,
    phase_in_series,
    reconstruct_ara_arb,
    regime_stats,
    to_energy_basis,
    voyage_emissions_t_co2,
)
from freight.ingest.fixture_coal import BREAKPOINT, SYNTHETIC_TICKERS, synthetic_coal
from freight.ingest.series import to_series

REAL_TICKERS: dict[str, str] = {}  # à remplir quand les séries arrivent

st.set_page_config(page_title="Coal Atlantic arb", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[dict[str, pd.Series], bool]:
    raw = synthetic_coal()
    return {r: to_series(raw, t) for r, t in SYNTHETIC_TICKERS.items()}, True


series, synthetic = load_data()

st.title("L'arb charbon Atlantique a perdu sa contrainte contraignante en 2022")
st.caption(
    "API2 − API4 − fret C4 − financement − ETS. Et le terme manquant : le netback vers "
    "la destination alternative."
)

if synthetic:
    st.error(
        "**DONNÉES SYNTHÉTIQUES — LECTURE ÉCONOMIQUE INTERDITE.** Pire que sur la page A : "
        "**la rupture de 2022 est imposée à la main dans le générateur.** Le coefficient de "
        "fret s'effondrera après 2022 parce que c'est écrit dans le code du générateur, pas "
        "parce que le marché l'a fait. Cette page prouve que les six sections et le test "
        "d'attribution tournent. Rien d'autre.",
        icon="🚫",
    )

with st.sidebar:
    st.header("Voyage (B-H3, B-H4)")
    voyage_days = st.number_input("Jours de voyage RB → ARA", 5, 60, 20)
    annual_rate = st.slider("Taux de financement annuel", 0.0, 0.15, 0.06, 0.005)
    cargo_t = st.number_input("Cargaison (t)", 50_000, 210_000, 150_000, step=10_000)
    st.divider()
    st.header("ETS maritime (B-H5, B-H6)")
    bunker_t = st.number_input("Combustible consommé sur le voyage (t)", 100, 2_000, 480)
    emission_factor = st.slider(
        "Facteur d'émission (tCO2/t fuel)", 2.9, 3.3, DEFAULT_EMISSION_FACTOR, 0.001
    )
    scope = st.slider("Portée voyage extra-UE", 0.0, 1.0, EXTRA_EU_SCOPE_FACTOR, 0.05)
    st.caption(
        "Portée × montée en charge = couverture effective : 20 % en 2024, 35 % en 2025, "
        "50 % à partir de 2026."
    )
    st.divider()
    st.header("Pouvoir calorifique (B-H2)")
    cv_actual = st.slider("CV réellement exporté (kcal/kg)", 5_300, 6_100, 5_750, 50)
    st.divider()
    breakpoint_str = st.text_input("Date de rupture testée", str(BREAKPOINT.date()))

emissions = voyage_emissions_t_co2(float(bunker_t), float(emission_factor))
phase_in = phase_in_series(series["api2"].index)
ets = ets_cost_per_cargo_tonne(
    eua_price_eur=series["eua"],
    eurusd=series["eurusd"],
    emissions_t_co2=emissions,
    cargo_t=float(cargo_t),
    phase_in=phase_in,
    scope_factor=float(scope),
)

arb = reconstruct_ara_arb(
    api2=series["api2"], api4=series["api4"], freight=series["freight"],
    voyage_days=float(voyage_days), annual_rate=float(annual_rate),
    ets_cost=ets.reindex(series["api2"].index),
)

# ------------------------------------------------------------------------------- S1
st.header("S1 — État actuel")
last = arb.iloc[-1]
cols = st.columns(7)
cols[0].metric("API2 CIF ARA", f"${last['api2']:,.1f}/t")
cols[1].metric("API4 FOB RB", f"${last['api4']:,.1f}/t")
cols[2].metric("Spread", f"${last['spread']:,.1f}/t")
cols[3].metric("Fret C4", f"${last['freight']:,.2f}/t")
cols[4].metric("Financement", f"${last['financing']:,.2f}/t")
cols[5].metric("ETS", f"${last['ets']:,.2f}/t")
cols[6].metric(
    "Arb", f"${last['arb']:,.2f}/t", "OUVERT" if last["is_open"] else "FERMÉ"
)

# ------------------------------------------------------------------------------- S2
st.header("S2 — L'arb, terme par terme")
st.markdown(
    """
```
arb_ARA = API2 − API4 − fret(C4) − financement − ETS
```
Si le fret est la contrainte contraignante, cet arb doit osciller autour de zéro sans
persistance : dès qu'il s'ouvre, des tonnes partent et il se referme.
"""
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=arb.index, y=arb["spread"], name="spread API2−API4", mode="lines"))
fig.add_trace(
    go.Scatter(
        x=arb.index, y=arb["freight"] + arb["financing"] + arb["ets"],
        name="coûts (fret + financement + ETS)", mode="lines",
    )
)
fig.add_trace(go.Scatter(x=arb.index, y=arb["arb"], name="arb", mode="lines",
                         line=dict(color="black", width=2)))
fig.add_hline(y=0, line_dash="dash")
fig.add_vline(x=pd.Timestamp(breakpoint_str), line_dash="dot")
fig.update_layout(height=430, yaxis_title="USD/t", xaxis_title="date",
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------- S3
st.header("S3 — Le test de rupture, avec le contrôle qui décide de tout")
st.markdown(
    """
**Le piège.** 2022 est l'année où le charbon sud-africain se réoriente vers l'Inde, **et**
l'année du choc gazier européen. Attribuer le décrochage à l'Inde sans contrôler par le
TTF, c'est se tromper de mécanisme — et la première personne compétente qui lit l'email le
verra. Les deux régressions ci-dessous sont donc affichées côte à côte, sans et avec le
contrôle.
"""
)
stats = regime_stats(arb, breakpoint_str)
st.dataframe(
    pd.DataFrame(
        [
            {
                "régime": s.label, "n": s.n_obs,
                "arb moyen ($/t)": round(s.arb_mean, 2),
                "écart-type": round(s.arb_std, 2),
                "part de jours ouverts": f"{100 * s.share_open:.0f}%",
                "plus longue série ouverte (jours)": s.longest_open_run,
            }
            for s in stats
        ]
    ),
    use_container_width=True, hide_index=True,
)
st.caption(
    "La persistance compte autant que le niveau : un arb contraint se referme vite, un "
    "arb non contraint reste ouvert des mois."
)

naive = freight_binding_test(arb["spread"], arb["freight"], breakpoint_str)
controlled = freight_binding_test(
    arb["spread"], arb["freight"], breakpoint_str,
    controls={"ttf": series["ttf"].reindex(arb.index)},
)
left, right = st.columns(2)
with left:
    st.subheader("Sans contrôle")
    for label, res in naive.items():
        st.markdown(f"**{label}** — {res.summary()}")
with right:
    st.subheader("Avec contrôle TTF")
    for label, res in controlled.items():
        st.markdown(f"**{label}** — {res.summary()}")
st.caption(
    "Écarts-types classiques, non robustes à l'autocorrélation : sur des prix quotidiens "
    "les t de Student sont optimistes. Le signe et l'ordre de grandeur des coefficients "
    "sont lisibles, pas leur significativité au dernier décimal."
)

# ------------------------------------------------------------------------------- S4
st.header("S4 — Le terme manquant, mesuré en flux")
st.markdown(
    """
Le prix CFR Inde est sous licence, donc on ne peut pas prouver l'égalité de prix. On
montre la réorientation physique : **quand l'arb ARA se ferme, la part indienne des
exports de Richards Bay monte.**

C'est un résultat plus faible qu'une égalité d'arbitrage, et il faut le présenter comme
tel plutôt que d'inventer une série de prix.

**Séries requises :** importations européennes de charbon sud-africain (Eurostat, mensuel,
gratuit) et importations indiennes par origine.
"""
)
st.info("Section en attente des séries de flux Eurostat et indiennes.", icon="⏳")

# ------------------------------------------------------------------------------- S5
st.header("S5 — La couche ETS, que personne n'intègre")
st.markdown(
    f"""
Un voyage Richards Bay → Rotterdam a une extrémité hors UE : la couverture est de
**{scope:.0%}** des émissions du voyage, multipliée par la montée en charge
(**{phase_in_factor(2024):.0%}** en 2024, **{phase_in_factor(2025):.0%}** en 2025,
**{phase_in_factor(2026):.0%}** à partir de 2026). Soit une couverture effective de
{scope * phase_in_factor(2024):.0%}, puis {scope * phase_in_factor(2025):.0%},
puis {scope * phase_in_factor(2026):.0%}.

```
coût_ETS = émissions × portée × montée_en_charge × prix_EUA × EURUSD / cargaison
```

Avec **{bunker_t} t** de combustible et un facteur de **{emission_factor:.3f} tCO2/t**, le
voyage émet **{emissions:,.0f} tCO2**. Le quota est coté en EUR et l'arb en USD : la
conversion de change est un terme du calcul, pas un détail.

**Conséquence économique :** à distance égale, le fret vers l'Europe devient
structurellement plus cher que le fret vers l'Inde. C'est un terme récent, chiffrable, et
absent des modèles d'arb charbon publics.
"""
)
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=arb.index, y=arb["ets"], name="coût ETS ($/t cargaison)",
                          mode="lines", fill="tozeroy"))
fig5.update_layout(height=300, yaxis_title="USD/t", xaxis_title="date",
                   title="Coût ETS par tonne — les marches sont la montée en charge")
st.plotly_chart(fig5, use_container_width=True)

recent = arb.loc[arb.index >= "2024-01-01"]
if not recent.empty:
    st.markdown(
        f"Depuis 2024, l'ETS retire en moyenne **${recent['ets'].mean():,.2f}/t** à l'arb, "
        f"pour un arb moyen de **${recent['arb'].mean():,.2f}/t** sur la même période — "
        f"soit **{100 * recent['ets'].mean() / max(abs(recent['arb'].mean()), 1e-9):,.1f} %** "
        "de sa valeur absolue."
    )

# ------------------------------------------------------------------------------- S6
st.header("S6 — Sensibilités")
st.subheader("Dérive du pouvoir calorifique")
st.markdown(
    f"""
API2 et API4 sont **tous deux** des références 6 000 kcal/kg NAR : l'arb de référence est
donc neutre en CV **par construction**, et il reste juste. Le problème est ailleurs — il a
cessé de décrire la cargaison physique, dont le CV réel a dérivé.

Le fret se paie **à la tonne**, le charbon se vend **au kcal**. À
**{cv_actual:,} kcal/kg**, une tonne livre {cv_actual / BENCHMARK_CV_KCAL_PER_KG:.1%} de
l'énergie d'une tonne de référence, donc le fret par tonne-équivalent-6 000 vaut
**{BENCHMARK_CV_KCAL_PER_KG / cv_actual:.4f}×** le fret affiché.
"""
)
cv_grid = list(range(5300, 6101, 100))
rows = []
for cv in cv_grid:
    freight_energy = to_energy_basis(arb["freight"], float(cv))
    rows.append(
        {
            "CV (kcal/kg)": cv,
            "facteur": round(BENCHMARK_CV_KCAL_PER_KG / cv, 4),
            "fret moyen affiché ($/t)": round(arb["freight"].mean(), 2),
            "fret moyen par t-éq-6000 ($)": round(freight_energy.mean(), 2),
            "surcoût ($/t)": round(freight_energy.mean() - arb["freight"].mean(), 2),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Sensibilité du seuil au fret et au financement")
grid = []
for days in (10, 15, 20, 25, 30):
    row = {"jours de voyage": days}
    for rate in (0.03, 0.06, 0.09, 0.12):
        a = reconstruct_ara_arb(
            api2=series["api2"], api4=series["api4"], freight=series["freight"],
            voyage_days=float(days), annual_rate=rate,
            ets_cost=ets.reindex(series["api2"].index),
        )
        row[f"{rate:.0%}"] = round(100 * a["is_open"].mean(), 1)
    grid.append(row)
st.dataframe(pd.DataFrame(grid), use_container_width=True, hide_index=True)
st.caption("Part de jours où l'arb est ouvert (%), selon la durée de voyage et le taux.")

st.divider()
st.markdown(
    """
#### Ce que cette page ne fait pas

- **Aucun prix indien.** Le netback alternatif est mesuré en flux, pas en prix. C'est la
  limite centrale du projet et elle est déclarée.
- **Aucune correction du mix de navires.** B-H3 suppose que C4 (Capesize) est le fret
  pertinent. Une partie du flux voyage en Panamax ou Supramax, à coût différent.
- **Aucun écart-type robuste à l'autocorrélation.** Les t affichés sont optimistes, et on
  le dit plutôt que d'appliquer une correction dont on ne montrerait pas l'hypothèse.
- **Aucune modélisation du soutage réel.** Le combustible consommé est un paramètre, pas
  une donnée de voyage.
"""
)
