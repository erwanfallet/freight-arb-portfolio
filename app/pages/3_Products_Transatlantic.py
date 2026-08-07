"""Projet C — S1 à S6."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from freight.chains.products import (
    DEFAULT_BBL_PER_TONNE,
    decompose_freight_change,
    density_sensitivity,
    flat_rate_step_series,
    freight_model_vs_quoted,
    freight_usd_per_tonne,
    implied_freight_from_tce,
    january_reset_effect,
    monthly_profile,
    open_days_comparison,
    reconstruct_arb,
)
from freight.ingest.fixture_products import (
    ROUTE,
    SYNTHETIC_FLAT_RATES,
    SYNTHETIC_TICKERS,
    synthetic_products,
)
from freight.ingest.series import to_series
from freight.voyage.config import VoyageParams
from freight.voyage.consumption import leg_bunker_consumption_t, sea_days

st.set_page_config(page_title="Transatlantic distillate arb", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[dict[str, pd.Series], bool]:
    raw = synthetic_products()
    return {r: to_series(raw, t) for r, t in SYNTHETIC_TICKERS.items()}, True


series, synthetic = load_data()

st.title("Arb distillat transatlantique — et le fret qui n'est pas un coût")
st.caption(
    "USGC ULSD → gasoil ARA. Deux termes de l'équation ne sont pas ce qu'ils paraissent : "
    "la conversion volume/masse et les points Worldscale."
)

if synthetic:
    st.error(
        "**DONNÉES SYNTHÉTIQUES — LECTURE ÉCONOMIQUE INTERDITE.** Les marches de flat rate "
        "au 1er janvier sont **écrites à la main** dans le générateur. Le saut de coût que "
        "vous allez voir au 1er janvier est là parce que je l'ai mis, pas parce que la "
        "Worldscale Association l'a fait cette année-là.",
        icon="🚫",
    )

with st.sidebar:
    st.header("Densité (C-H1)")
    bbl_per_tonne = st.slider(
        "Barils par tonne", 7.20, 7.70, DEFAULT_BBL_PER_TONNE, 0.01,
        help="Le terme le plus sensible du projet. Il ne peut pas être un nombre en dur.",
    )
    st.header("Pont de spécification (C-H2)")
    spec_bridge = st.slider("USGC 15 ppm → ICE Gasoil 10 ppm ($/t)", 0.0, 20.0, 5.0, 0.5)
    st.caption("Fourchette explicite, jamais un chiffre unique.")
    st.header("Voyage (C-H3, C-H4)")
    voyage_days = st.number_input("Jours de traversée", 8, 40, 16)
    annual_rate = st.slider("Taux de financement annuel", 0.0, 0.15, 0.06, 0.005)
    losses = st.slider("Pertes en transit + surestaries ($/t)", 0.0, 10.0, 1.0, 0.5)

flat_rates = flat_rate_step_series(series["ws"].index, ROUTE, SYNTHETIC_FLAT_RATES)
freight = freight_usd_per_tonne(series["ws"], flat_rates)

arb = reconstruct_arb(
    price_destination_usd_t=series["p_ara"],
    price_origin_usd_per_gallon=series["p_usgc"],
    freight_usd_t=freight,
    bbl_per_tonne=float(bbl_per_tonne),
    spec_bridge_usd_t=float(spec_bridge),
    voyage_days=float(voyage_days),
    annual_rate=float(annual_rate),
    losses_usd_t=float(losses),
)

# ------------------------------------------------------------------------------- S1
st.header("S1 — État actuel")
last = arb.iloc[-1]
cols = st.columns(7)
cols[0].metric("Gasoil ARA", f"${last['p_dest']:,.1f}/t")
cols[1].metric("ULSD USGC", f"${last['p_origin_gal']:,.3f}/gal")
cols[2].metric("→ converti", f"${last['p_origin_t']:,.1f}/t")
cols[3].metric("Spread nu", f"${last['spread_naif']:,.2f}/t")
cols[4].metric("Fret", f"${last['freight']:,.2f}/t")
cols[5].metric("Financement", f"${last['financing']:,.2f}/t")
cols[6].metric("Arb", f"${last['arb']:,.2f}/t", "OUVERT" if last["is_open"] else "FERMÉ")

# ------------------------------------------------------------------------------- S2
st.header("S2 — L'arb, terme par terme")
st.markdown(
    """
```
arb = P_ARA($/t) − P_USGC($/t) − fret($/t) − pont_spec − financement − pertes
      avec P_USGC($/t) = P_USGC($/gal) × 42 × bbl_par_tonne
```
"""
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=arb.index, y=arb["spread_naif"], name="spread nu (avant coûts)",
                         mode="lines"))
fig.add_trace(go.Scatter(x=arb.index, y=arb["arb"], name="arb réel", mode="lines",
                         line=dict(color="black", width=2)))
fig.add_hline(y=0, line_dash="dash")
fig.update_layout(height=420, yaxis_title="USD/t", xaxis_title="date",
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------- S3
st.header("S3 — Les points Worldscale ne sont pas un coût")
st.markdown(
    """
Le fret tanker se cote en points Worldscale. WS 100 correspond au *flat rate* de la route,
**recalculé chaque 1er janvier** à partir de l'environnement de coûts de l'année
précédente :

```
fret($/t) = WS/100 × flat_rate(route, année)

Δfret = [ ΔWS·FR_prev  +  WS_prev·ΔFR  +  ΔWS·ΔFR ] / 100
          └ marché ┘      └ réglage ┘     └ croisé ┘
```

Les trois termes somment **exactement** à Δfret — c'est une identité algébrique. Le terme
« réglage » est celui qu'un modèle en points WS ne voit pas : au 1er janvier, ΔWS peut être
nul et Δfret ne pas l'être.
"""
)
decomp = decompose_freight_change(series["ws"], flat_rates)
resets = january_reset_effect(decomp)
if not resets.empty:
    st.subheader("Réinitialisations de flat rate")
    st.dataframe(resets, use_container_width=True, hide_index=True)
    biggest = resets.iloc[resets["part_reglage"].abs().argmax()]
    st.markdown(
        f"""
La plus grosse marche de l'échantillon : le **{pd.Timestamp(biggest['date']).date()}**, le
coût de fret bouge de **${biggest['saut_total']:,.2f}/t**, dont
**${biggest['part_reglage']:,.2f}/t** attribuables au seul réglage du flat rate et
**${biggest['part_marche']:,.2f}/t** au marché.

À comparer à l'arb moyen de **${arb['arb'].mean():,.2f}/t** : un modèle qui suit les points
WS se trompe de seuil d'un montant du même ordre que le signal qu'il cherche.
"""
    )

fig3 = go.Figure()
fig3.add_trace(go.Bar(x=decomp.index, y=decomp["market_component"], name="part marché"))
fig3.add_trace(go.Bar(x=decomp.index, y=decomp["reset_component"], name="part réglage"))
fig3.update_layout(barmode="relative", height=320, yaxis_title="USD/t par jour",
                   xaxis_title="date", title="Variation quotidienne du fret, décomposée",
                   legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Le module `signals/worldscale.py` refuse par construction de convertir des points en "
    "$/t sans flat rate daté : il lève une exception plutôt que de se rabattre sur "
    "l'année précédente. C'est cette erreur-là que le projet démontre."
)

# ------------------------------------------------------------------------------- S4
st.header("S4 — Combien de jours « ouverts » n'existent pas")
comp = open_days_comparison(arb)
c1, c2, c3 = st.columns(3)
c1.metric("Jours ouverts sur le spread nu", f"{comp.days_looking_open:,}",
          f"{100 * comp.share_looking_open:.1f}%")
c2.metric("Jours réellement ouverts", f"{comp.days_really_open:,}",
          f"{100 * comp.share_really_open:.1f}%")
c3.metric("Illusion", f"{100 * comp.illusion_share:.1f}%" if comp.days_looking_open else "n/a")
st.markdown(
    """
**Validation de flux, la série qui rend ce projet vérifiable :** exportations américaines
de distillat par destination (EIA, mensuel, gratuit). Si l'arb est réel, les tonnes
partent — et contrairement au projet A, cette série est officielle, gratuite et directement
liée au trade.
"""
)
st.info("Série de flux EIA à brancher.", icon="⏳")

st.subheader("Saisonnalité")
profile = monthly_profile(arb["arb"])
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=profile.index, y=profile["mean"], name="arb moyen"))
fig4.update_layout(height=300, yaxis_title="USD/t", xaxis_title="mois",
                   title="Arb moyen par mois calendaire")
st.plotly_chart(fig4, use_container_width=True)
st.caption(
    "Un arb qui ne survit que quelques mois par an n'est pas le même objet qu'un arb "
    "permanent. Chauffage européen et campagnes de maintenance."
)

# ------------------------------------------------------------------------------- S5
st.header("S5 — Variante C-2 : le fret calculé plutôt qu'acheté")
st.markdown(
    """
Si les flat rates Worldscale sont indisponibles, on ne renonce pas au projet : on inverse
le moteur TCE.

```
TCE = (cargo · F · (1 − c) − coûts) / jours   =>   F = (TCE · jours + coûts) / (cargo · (1 − c))
```

À partir d'un TCE de marché MR, des distances, du prix des soutes et des jours de port, on
remonte au taux de fret que ce TCE implique. **« Je n'ai pas acheté le fret, je l'ai
reconstruit » est un argument de crédibilité plus fort qu'un abonnement.**
"""
)
mr_col1, mr_col2, mr_col3 = st.columns(3)
distance = mr_col1.number_input("Distance chargée (nm)", 2_000, 9_000, 5_000, step=250)
cargo_t = mr_col2.number_input("Cargaison MR (t)", 20_000, 60_000, 37_000, step=1_000)
port_days = mr_col3.number_input("Jours de port", 1, 12, 4)

mr_params = VoyageParams(
    cargo_t=float(cargo_t), laden_speed_kn=13.0, ballast_speed_kn=14.0,
    reference_speed_kn=13.0, reference_consumption_t_per_day=25.0,
    port_costs_usd=120_000.0, brokerage_commission=0.0375,
)
laden_days = sea_days(float(distance), mr_params.laden_speed_kn)
ballast_days = sea_days(float(distance), mr_params.ballast_speed_kn)
total_days = laden_days + ballast_days + float(port_days)
bunker_t = (
    leg_bunker_consumption_t(float(distance), mr_params.laden_speed_kn, mr_params)
    + leg_bunker_consumption_t(float(distance), mr_params.ballast_speed_kn, mr_params)
)
voyage_costs = bunker_t * series["bunker"] + mr_params.port_costs_usd
modelled_freight = pd.Series(
    [
        implied_freight_from_tce(
            target_tce_usd_per_day=float(tce), total_days=total_days,
            total_voyage_costs_usd=float(cost), cargo_t=float(cargo_t),
            commission=mr_params.brokerage_commission,
        )
        for tce, cost in zip(series["tce_mr"], voyage_costs)
    ],
    index=series["tce_mr"].index,
    name="modelled_freight",
)

stats = freight_model_vs_quoted(modelled_freight, freight)
s1, s2, s3 = st.columns(3)
s1.metric("Écart moyen", f"${stats['ecart_moyen_usd_t']:,.2f}/t")
s2.metric("Écart moyen relatif", f"{stats['ecart_moyen_pct']:,.1f}%")
s3.metric("Corrélation", f"{stats['correlation']:,.2f}")

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=freight.index, y=freight, name="fret coté (WS × flat rate)",
                          mode="lines"))
fig5.add_trace(go.Scatter(x=modelled_freight.index, y=modelled_freight,
                          name="fret reconstruit (TCE inversé)", mode="lines"))
fig5.update_layout(height=350, yaxis_title="USD/t", xaxis_title="date",
                   legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig5, use_container_width=True)
st.markdown(
    f"""
Voyage retenu : **{laden_days:.1f} j** chargés + **{ballast_days:.1f} j** sur ballast +
**{port_days} j** de port = **{total_days:.1f} j**, pour **{bunker_t:,.0f} t** de soutes.

Un écart systématiquement d'un côté n'est pas une erreur de modèle : c'est ce que la route
intègre au-delà du coût de voyage — attente, positionnement, pouvoir de négociation. C'est
le résultat de la variante, pas son échec.
"""
)

# ------------------------------------------------------------------------------- S6
st.header("S6 — Sensibilités")
st.subheader("Densité : le terme le plus incertain est aussi l'un des plus gros")
st.dataframe(
    density_sensitivity(float(arb["p_origin_gal"].iloc[-1])),
    use_container_width=True, hide_index=True,
)
st.markdown(
    f"""
À **${arb['p_origin_gal'].iloc[-1]:,.3f}/gal**, passer de 7,45 à 7,50 bbl/t déplace la jambe
US de plusieurs dollars la tonne — souvent davantage que l'arb tout entier, dont la moyenne
est de **${arb['arb'].mean():,.2f}/t** sur l'échantillon.

C'est le résultat central du projet : **le facteur de conversion que tout le monde traite
comme une constante pèse autant que le signal.**
"""
)

st.subheader("Pont de spécification et jours de voyage")
grid = []
for spec in (0.0, 2.5, 5.0, 7.5, 10.0):
    row = {"pont spec ($/t)": spec}
    for days in (12, 16, 20):
        a = reconstruct_arb(
            price_destination_usd_t=series["p_ara"],
            price_origin_usd_per_gallon=series["p_usgc"],
            freight_usd_t=freight,
            bbl_per_tonne=float(bbl_per_tonne), spec_bridge_usd_t=spec,
            voyage_days=float(days), annual_rate=float(annual_rate),
            losses_usd_t=float(losses),
        )
        row[f"{days} j"] = round(100 * a["is_open"].mean(), 1)
    grid.append(row)
st.dataframe(pd.DataFrame(grid), use_container_width=True, hide_index=True)
st.caption("Part de jours où l'arb est ouvert (%).")

st.divider()
st.markdown(
    """
#### Ce que cette page ne fait pas

- **Aucun pont de spécification estimé.** C-H2 est une fourchette réglée à la main, pas un
  modèle. Bricoler ce terme transformerait l'arb en bruit.
- **Aucune modélisation de la densité réelle du lot.** La densité est une hypothèse balayée,
  pas une donnée.
- **Aucun coût de mise à quai, d'inspection ou de mélange à l'arrivée.**
- **Le fret reconstruit suppose un ballast retour à vide sur la même distance.** Un MR
  trouve souvent un chargement retour, ce qui abaisse le fret d'équilibre.
"""
)
