from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.hedge_cost import (  # noqa: E402
    SHORT_HEDGE,
    HedgeParams,
    forced_exit_price,
    forced_exit_schedule,
    hedge_capacity,
    hedging_intensity,
    implied_margin_rate,
    load_real_hedge_frame,
    procyclicality,
)
from agri.data.bloomberg_loader import DEFAULT_PATH, load  # noqa: E402
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

st.set_page_config(page_title="T1-2 — Coût du hedge", layout="wide")

_LIVE = snapshot_banner()

COMMODITIES = {
    "cacao_ny": ("Cacao ICE New York", "cocoa_ny", "USD/t"),
    "cacao_londres": ("Cacao ICE Londres", "cocoa_london", "GBP/t"),
    "cafe_arabica": ("Café Arabica ICE", "coffee_arabica", "c/lb"),
    "cafe_robusta": ("Café Robusta ICE", "coffee_robusta", "USD/t"),
}

st.sidebar.markdown("### Book et bilan")
commodity = st.sidebar.selectbox(
    "Matière première", list(COMMODITIES), format_func=lambda k: COMMODITIES[k][0]
)
book_kt = st.sidebar.slider("Taille du book physique (kt)", 10, 500, 100, 10)
line_musd = st.sidebar.slider("Lignes de crédit (M USD)", 50, 1500, 250, 25)
book_t = book_kt * 1000.0
line_usd = line_musd * 1e6

label, series_key, unit = COMMODITIES[commodity]
params = HedgeParams(side=SHORT_HEDGE, book_size_t=book_t, credit_line_usd=line_usd)
simulation = load_real_hedge_frame(commodity, params=params)
# Le prix affiche est celui que la simulation a REELLEMENT utilise, pas un rechargement
# separe : les deux coincident, mais seul le premier survit au mode snapshot.
price = simulation["front"].rename(series_key)
margin_rate = implied_margin_rate(simulation, book_size_t=book_t)

page_header(
    code="T1-2",
    title="Le coût complet de la couverture — cacao et café",
    subtitle=(
        "Le prix auquel le bilan d'une maison la force à cesser de couvrir, donc à cesser "
        "d'acheter du physique — et pourquoi tant de maisons ont heurté ce mur en même temps"
    ),
    scope=Scope(
        unit_trap=(
            "La **variation margin n'est pas un coût, c'est un transfert** — ce qui coûte, "
            "c'est de la financer. Le coût cumulé de couverture exclut donc la VM elle-même "
            "et ne retient que son financement. Confondre les deux fait passer une perte "
            "mark-to-market récupérable pour une destruction de valeur."
        ),
        conversion=(
            "cash_t   = IM_t + max(0, pertes cumulées non compensées)\n"
            "coût_t   = cash_t x (taux + spread) / 360      <- base 360, convention monétaire\n"
            "P*       = (B/Q + P0) / (1 + taux_de_marge)    <- le prix de sortie forcée"
        ),
        proxies=[
            "barème de marge initiale ICE indisponible → proxy k·σ·F calibré sur la "
            f"volatilité réalisée, soit un taux implicite de {margin_rate:.1%} du prix",
        ],
        out_of_scope=[
            "coût de roll : l'export ne contient que le contrat front-month générique, "
            "aucune échéance différée — le terme est neutralisé, pas estimé",
            "appels de marge intrajournaliers et haircuts sur collatéral non-cash, qui "
            "tous deux **sous-estiment** le besoin de trésorerie réel",
        ],
        frequency_note=(
            "Prix ICE et SOFR quotidiens. L'échantillon démarre au 02/04/2018, début de "
            "la couverture SOFR — au-delà, le taux de financement réel n'existe pas."
        ),
        data_warnings=[
            "Le cacao Londres cote en GBP/t et l'arabica en cents/lb : les seuils calculés "
            "sont dans l'unité native du contrat sélectionné, pas convertis en USD/t.",
        ],
    ),
)

capacity = hedge_capacity(simulation, credit_line_usd=line_usd, book_size_t=book_t)
intensity = hedging_intensity(simulation, book_size_t=book_t)

kpi_banner(
    {
        f"Prix ({unit})": f"{simulation['front'].iloc[-1]:,.0f}",
        "Pic historique": f"{simulation['front'].max():,.0f}",
        "Trésorerie au pic": f"{simulation['cash_usd'].max()/1e6:,.0f} M USD",
        "Intensité au pic": f"{intensity.peak_ratio:.0%}",
        "Capacité actuelle": f"{capacity.capacity_t.iloc[-1]/1000:,.0f} kt",
    }
)

# ===========================================================================
section(
    "S2",
    "Le prix auquel le bilan force la sortie",
    "Une maison longue de physique et courte de futures perd de la trésorerie quand le "
    "prix **monte** — exactement quand son stock prend de la valeur. Le seuil se calcule "
    "en forme fermée plutôt qu'au solveur, parce que la forme fermée montre de quoi il "
    "dépend : linéairement de la ligne rapportée à la taille du book, du prix d'entrée, "
    "et à peine du taux de marge. Au-delà de ce prix, la maison ne peut plus financer sa "
    "couverture ; elle ne peut donc plus couvrir de physique supplémentaire, et comme "
    "personne n'achète du physique non couvert à ce niveau de volatilité, **elle cesse "
    "d'acheter**. C'est le point où une crise de trésorerie de trading devient un arrêt "
    "des achats au bord-champ.",
    formula="P* = (B/Q + P0) / (1 + taux_de_marge)",
)

inception_choices = ["2018-04-02", "2022-01-03", "2023-01-03", "2023-09-01", "2024-01-02"]
schedule = forced_exit_schedule(
    price, inception_choices, book_size_t=book_t, credit_line_usd=line_usd, im_rate=margin_rate
)

span_days: int | None = None
median_exit: float | None = None

if not schedule.empty:
    median_exit = float(schedule["sortie forcée"].median())

if not schedule.empty and schedule["franchi le"].notna().any():
    crossed = schedule["franchi le"].dropna()
    span_days = int((crossed.max() - crossed.min()).days)
    earliest = schedule.loc[schedule["ouverture"].idxmin()]
    latest = schedule.loc[schedule["ouverture"].idxmax()]
    gap_years = (latest["ouverture"] - earliest["ouverture"]).days / 365.25
    finding(
        f"Une maison couverte depuis {earliest['ouverture']:%b %Y} et une maison couverte "
        f"depuis {latest['ouverture']:%b %Y} — {gap_years:.0f} ans d'écart sur le point "
        f"d'entrée — sont forcées de sortir à **{span_days} jours d'intervalle**. Le "
        "mouvement a été assez violent pour écraser la dispersion des dates d'ouverture : "
        "c'est pourquoi tant de maisons ont heurté la contrainte de bilan en même temps "
        "plutôt que chacune à son tour."
    )
else:
    finding(
        "Sur cet échantillon et ce paramétrage, le marché n'a jamais atteint le prix de "
        "sortie forcée — la ligne n'a jamais été la contrainte contraignante."
    )

# `.round()` s'applique aux colonnes numeriques uniquement : l'appliquer au frame entier
# emet un avertissement sur les colonnes de dates et ne les touche pas.
display = schedule.copy()
display["marge de manœuvre"] = display["marge de manœuvre"].map(lambda v: f"{v:+.0%}")
numeric = display.select_dtypes("number").columns
display[numeric] = display[numeric].round(0)
st.dataframe(display, width="stretch", hide_index=True)
scope_note(
    "« Jours de protection » = temps entre l'ouverture de la couverture et le franchissement "
    "du seuil. L'ordre est respecté — se couvrir tôt reste meilleur — mais l'écart se "
    "compte en semaines, pas en années."
)

# ===========================================================================
section(
    "S3",
    "La trésorerie immobilisée, rapportée au stock qu'elle protège",
    "C'est la grandeur que les avocats de Montesanto Tavares ont chiffrée en novembre "
    "2025 : le coût de maintien des couvertures était passé de **74 % des créances "
    "clients en mai à 158 % en novembre**, et ils l'ont qualifiée d'insoutenable. La "
    "même mesure se reconstruit ici sans avoir besoin du bilan de qui que ce soit — au "
    "départ on ne poste que la marge initiale, puis la variation margin s'empile jusqu'à "
    "immobiliser presque autant de trésorerie que le stock ne vaut.",
    formula="intensité_t = trésorerie mobilisée_t / (Q x prix_t)",
)
finding(intensity.headline)
show(
    regime_chart(
        intensity.ratio.to_frame("intensité").assign(au_dessus=intensity.ratio > 0.5),
        "intensité", regime_col="au_dessus",
        title="Trésorerie immobilisée / valeur du book physique",
        y_title="ratio", zero_line=False, reference_lines={"100 %": 1.0},
    )
)

# ===========================================================================
section(
    "S4",
    "La procyclicité — le coût explose quand la couverture est le plus nécessaire",
    "La marge initiale suit le prix : elle monte quand le marché s'emballe, c'est-à-dire "
    "au moment précis où une maison longue de physique a le plus besoin d'être couverte. "
    "La corrélation se mesure sur séries **différenciées** — deux séries en niveau non "
    "stationnaires produiraient une corrélation flatteuse qui ne dit rien.",
)
stats = procyclicality(simulation)
c1, c2 = st.columns(2)
c1.metric("Corr Δ(marge initiale) / Δ(prix)", f"{stats['corr_delta_im_delta_price']:+.2f}")
c2.metric("Observations", f"{stats['n_obs']:,}")
show(
    regime_chart(
        capacity.capacity_t.to_frame("capacité") / 1000,
        "capacité",
        title="Book maximal couvrable avec les lignes disponibles (kt)",
        y_title="kt", zero_line=False, reference_lines={f"book {book_kt} kt": float(book_kt)},
    )
)
if capacity.is_binding:
    diagnostic_note(
        f"La ligne devient contraignante : la capacité tombe à "
        f"{capacity.min_capacity_t/1000:,.0f} kt contre un book de {book_kt} kt, avec un "
        f"recul de {capacity.contraction_over(4):.0%} sur les quatre mois précédant le pic "
        f"du {capacity.peak_cash_date:%d/%m/%Y}."
    )
else:
    scope_note("Sur ce paramétrage, la ligne reste au-dessus du book sur tout l'échantillon.")

# ===========================================================================
section(
    "S5",
    "Décomposition du coût de couverture",
    "Trois postes, séparés parce qu'ils ne se compensent pas de la même façon : le "
    "financement dépend du niveau de trésorerie immobilisée, le roll de la structure par "
    "terme, la liquidité de la fréquence des rolls. Ici seul le financement est réel — "
    "l'export ne contient pas d'échéance différée, donc le roll et la liquidité sont "
    "neutralisés plutôt qu'estimés sur une hypothèse de courbe inventée.",
)
components = pd.DataFrame(
    {
        "financement": simulation["financing_usd"].cumsum() / book_t,
        "roll": simulation["roll_usd"].cumsum() / book_t,
        "liquidité": simulation["liquidity_usd"].cumsum() / book_t,
    }
)
show(
    regime_chart(
        components.assign(total=components.sum(axis=1)), "total",
        title=f"Coût de couverture cumulé ({unit} de book)", y_title=unit, zero_line=True,
    )
)
neutralised = [
    name for name in ("roll", "liquidité") if components[name].abs().max() == 0
]
if neutralised:
    scope_note(
        f"Postes neutralisés faute de donnée réelle : {', '.join(neutralised)}. "
        "Leur omission **sous-estime** le coût total — le biais va contre la thèse, "
        "donc dans le bon sens."
    )

exit_text = f"autour de {median_exit:,.0f} {unit}" if median_exit is not None else "non atteignable sur l'échantillon"
span_text = (
    f" — et surtout des dates de franchissement resserrées sur {span_days} jours quelle "
    "que soit cette date"
    if span_days is not None
    else ""
)
mail_question(
    f"Sur le {label}, avec {line_musd} M USD de lignes et un book de {book_kt} kt, "
    f"je trouve un prix de sortie forcée {exit_text} selon la date d'ouverture de la "
    f"couverture{span_text}. Est-ce que votre limite se déclenche à ce niveau-là, et "
    "est-elle formalisée, ou est-ce qu'elle se découvre en route ?",
    "ofi/Olam, ECOM, Volcafe, Sucden Coffee, Touton, Barry Callebaut, Cargill Cocoa, Freepoint softs",
)
