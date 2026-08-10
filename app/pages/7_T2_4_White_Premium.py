from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.white_premium import (  # noqa: E402
    POL_PLAUSIBLE_HI,
    POL_PLAUSIBLE_LO,
    WhitePremiumError,
    identification_check,
    implied_pol_adjust,
    implied_refining_cost,
    load_real_richness_frame,
    pol_adjust_sensitivity,
    summarise_richness,
)
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (  # noqa: E402
    ALT_COLOR,
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

st.set_page_config(page_title="T2-4 — White premium", layout="wide")

if not DEFAULT_PATH.exists():
    st.error(f"Fichier Bloomberg introuvable : {DEFAULT_PATH}")
    st.stop()

# ===========================================================================
# En-tête et périmètre
# ===========================================================================
page_header(
    code="T2-4",
    title="Le white premium, ou ce qu'un prix peut dire et ce qu'il ne peut pas",
    subtitle=(
        "Le niveau de la rente de raffinage n'est pas identifiable à partir des prix — sa "
        "variation l'est entièrement, et elle a changé de régime d'environ 60 USD/t"
    ),
    scope=Scope(
        unit_trap=(
            "Le No.11 cote en **cents par livre, base 96° pol** ; le No.5 en **USD par "
            "tonne** de sucre raffiné. Les comparer demande la conversion c/lb → USD/t "
            "**et** un ajustement de polarisation, parce qu'il faut plus d'une tonne de brut "
            "à 96° pour faire une tonne de blanc. Ce second facteur n'est **publié par "
            "personne** : il dépend de la spécification contractuelle retenue et vaut "
            f"entre {POL_PLAUSIBLE_LO:.2f} et {POL_PLAUSIBLE_HI:.2f}. Toute la page consiste "
            "à établir ce qu'on peut conclure malgré cette ignorance."
        ),
        conversion=(
            "white_premium = No5_usd_t - No11_c_lb x 22,0462 x pol_adjust\n"
            "richness      = white_premium - coût_de_raffinage\n"
            "                (> 0 : le blanc paie plus que le coût de le produire)"
        ),
        proxies=[
            "coût énergie : Henry Hub réel x une intensité énergétique paramétrée "
            "(~8 mmBtu/t) — le prix est réel, l'intensité ne l'est pas",
            "main-d'œuvre, fret et perte de rendement : forfaits, aucune comptabilité "
            "analytique de raffinerie n'est publique en série temporelle",
        ],
        out_of_scope=[
            "coût du capital de l'actif de raffinage : la richness est une marge de "
            "contribution, jamais un profit — ne pas la comparer à un ROIC (W-H4)",
            "primes de qualité et contraintes de livraison sur le No.5, qui font partie du "
            "résidu que la page ne prétend pas décomposer",
        ],
        frequency_note="No.11, No.5 et Henry Hub sont quotidiens ; la richness l'est donc aussi.",
        data_warnings=[
            "L'ajustement de polarisation est le paramètre qui décide du signe. La page ne "
            "le fixe pas : elle mesure d'abord ce que son incertitude coûte (S3), et ne "
            "conclut que sur ce qui y survit (S4).",
        ],
    ),
)

# ===========================================================================
# Paramètres
# ===========================================================================
st.sidebar.markdown("### Paramètres")
pol_adjust = st.sidebar.slider(
    "Ajustement de polarisation", 1.00, 1.20, 1.07, 0.005,
    help="Inobservable. Plage plausible 1,06–1,08 selon la spécification contractuelle.",
)
energy_intensity = st.sidebar.slider(
    "Intensité énergétique du raffinage (mmBtu/t)", 3.0, 15.0, 8.0, 0.5
)
window_start = st.sidebar.selectbox(
    "Fenêtre", ["2015-01-01", "2010-01-01", "1990-07-18"], index=0
)

common = dict(energy_intensity_mmbtu_t=energy_intensity)
frame = load_real_richness_frame(pol_adjust=pol_adjust, start=window_start, **common)
summary = summarise_richness(frame)
check = identification_check(start=window_start, pol_ref=pol_adjust, **common)
cost = implied_refining_cost(start=window_start, pol_adjust=pol_adjust, **common)

kpi_banner(
    {
        "White premium (médian)": f"{frame['white_premium'].median():.0f} USD/t",
        "Coût modélisé": f"{cost.modelled_usd_t:.0f} USD/t",
        "Richness médiane": f"{frame['richness'].median():+.1f} USD/t",
        "Amplitude du paramètre": f"{check.parameter_span_max:.1f} USD/t",
        "Amplitude du signal": f"{check.signal_span:.1f} USD/t",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "« Le white premium est la marge de raffinage » est une affirmation, pas une définition",
    "L'écart entre le sucre blanc No.5 et le sucre brut No.11 est couramment appelé la marge "
    "de raffinage, et traité comme si c'en était une par construction. Ce n'est pas le cas. "
    "C'est un écart entre deux contrats à terme qui diffèrent par **quatre choses à la "
    "fois** : le degré de raffinage, l'unité de cotation, la base de polarisation, et le "
    "lieu et la forme de livraison. Appeler cet écart « la marge de raffinage », c'est "
    "affirmer que le premier terme domine les trois autres.\n\n"
    "L'affirmation est peut-être vraie. Mais elle est testable, et le test bute d'emblée sur "
    "un obstacle qui n'est pas économique : pour seulement **écrire** l'écart, il faut "
    "connaître un facteur de conversion que personne ne publie.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "Le facteur que personne ne publie",
    "Le No.11 se livre à 96° de polarisation, le No.5 est du sucre raffiné. Il faut donc "
    "plus d'une tonne de brut pour faire une tonne de blanc, et le rapport dépend de la "
    "spécification contractuelle exacte — pas d'une loi physique. Les valeurs qui circulent "
    f"vont de {POL_PLAUSIBLE_LO:.2f} à {POL_PLAUSIBLE_HI:.2f}.\n\n"
    "L'écart entre ces deux bornes paraît anodin : deux points de pourcentage. Mais il "
    "multiplie le prix du brut converti en USD/t, c'est-à-dire une grandeur de plusieurs "
    "centaines de dollars. Sur l'échantillon réel, changer de borne déplace la richness "
    f"d'environ {check.parameter_span_max:.0f} USD/t — à comparer à une richness médiane de "
    f"{frame['richness'].median():+.1f} USD/t. **Le paramètre inobservable est du même ordre "
    "que la réponse cherchée.**",
    formula=f"pol_adjust ∈ [{POL_PLAUSIBLE_LO:.2f} ; {POL_PLAUSIBLE_HI:.2f}]  →  richness ± {check.parameter_span_max / 2:.1f} USD/t",
)
show(
    regime_chart(
        frame.assign(sous_zero=frame["richness"] <= 0),
        "richness",
        regime_col="sous_zero",
        regime_color=SHUT_COLOR,
        title=f"Richness du raffinage à pol_adjust = {pol_adjust:.3f}",
        y_title="USD/tonne",
        annotations={"2020-03-01": "Covid", "2023-01-01": "?"},
    )
)
st.caption(summary.headline)

# ===========================================================================
# S3 — le problème, posé franchement
# ===========================================================================
section(
    "S3",
    "Le niveau n'est pas identifiable — et l'inversion le montre proprement",
    "Plutôt que de choisir une valeur et de conclure, on retourne la question. Supposons le "
    "raffinage **concurrentiel** : une industrie mature ne dégage pas de rente médiane "
    "durable. Quel ajustement de polarisation le marché price-t-il alors ? C'est une "
    "équation à une inconnue, et sa solution se compare directement à la spécification qu'un "
    "raffineur connaît par cœur.",
    formula="pol* tel que médiane(richness) = 0",
)
try:
    implied = implied_pol_adjust(start=window_start, **common)
    finding(implied.headline)
    c1, c2, c3 = st.columns(3)
    c1.metric("pol* impliqué", f"{implied.pol_star:.4f}")
    c2.metric("Borne plausible haute", f"{POL_PLAUSIBLE_HI:.2f}")
    c3.metric(
        "Écart",
        f"{implied.pol_star - POL_PLAUSIBLE_HI:+.4f}",
        delta="dans la plage" if implied.within_plausible else "hors plage",
        delta_color="off",
    )
    scope_note(
        "Deux lectures restent ouvertes et **le prix ne permet pas de trancher entre elles** : "
        "soit la plage plausible est trop étroite et les raffineurs travaillent avec un "
        "ajustement plus élevé, soit le raffinage capte effectivement une rente. C'est "
        "exactement le genre de question qu'un insider règle en une phrase et qu'aucune "
        "quantité de données de prix ne réglera."
    )
except WhitePremiumError as error:
    diagnostic_note(f"Inversion impossible sur cette fenêtre : {error}")

# ===========================================================================
# S4 — LE RÉSULTAT
# ===========================================================================
section(
    "S4",
    "Mais la variation, elle, est entièrement identifiable",
    "Un paramètre inconnu ne rend pas une page inutile — il rend inutilisables les "
    "conclusions qui en dépendent, et seulement celles-là. L'ajustement de polarisation "
    "multiplie le prix du brut, donc il déplace la richness de toutes les années **dans le "
    "même sens et de façon comparable**. Les écarts entre années lui survivent.\n\n"
    "Le tableau ci-dessous donne la richness médiane par année aux deux bornes de la plage "
    "plausible et à la valeur retenue. Trois vérifications décident si on a le droit de lire "
    "les écarts : l'amplitude du paramètre comparée à celle du signal, la stabilité du "
    "classement des années, et le nombre d'années dont le **signe** dépend du paramètre. "
    "Celles-là, et seulement celles-là, restent non interprétables.",
)
finding(check.headline)

annual = check.annual.copy()
annual.columns = [
    f"pol {check.pol_lo:.2f}", f"pol {check.pol_ref:.3f}", f"pol {check.pol_hi:.2f}"
]
annual["amplitude du paramètre"] = (annual.iloc[:, 0] - annual.iloc[:, 2]).round(1)
st.dataframe(annual.round(1), width="stretch")

flipping = check.sign_flipping_years
if flipping:
    diagnostic_note(
        f"Année(s) non interprétable(s) : {', '.join(map(str, flipping))} — leur signe "
        "dépend du choix de pol_adjust, donc rien ne peut en être dit. Elles sont écartées "
        "de la lecture plutôt que résolues par un choix de paramètre commode."
    )
else:
    scope_note("Aucune année ne change de signe sur la plage plausible.")

# ===========================================================================
# S5 — ce que la variation révèle
# ===========================================================================
reference = check.annual["richness_ref"]
worst_year, best_year = int(reference.idxmin()), int(reference.idxmax())
section(
    "S5",
    "Ce que la variation révèle : un changement de régime, pas une oscillation",
    f"Une fois acquis qu'on peut lire les écarts, le graphe est net. La richness est "
    f"**persistamment négative de 2017 à 2021** — le raffinage détruit de la valeur au prix "
    f"affiché, avec un creux à {reference.min():+.0f} USD/t en {worst_year} — puis bascule "
    f"franchement positive à partir de 2023, jusqu'à {reference.max():+.0f} USD/t en "
    f"{best_year}, et elle y est restée.\n\n"
    f"L'amplitude de ce basculement, environ {check.signal_span:.0f} USD/t, est **{check.ratio:.1f} "
    "fois** ce que l'incertitude de paramètre peut produire. Ce n'est donc pas un artefact "
    "de convention : quelque chose a changé dans l'économie du raffinage entre 2021 et 2023, "
    "et c'est resté. La page mesure ce basculement ; elle ne prétend pas l'expliquer — "
    "restrictions à l'export chez les grands exportateurs, fermetures de capacité, "
    "réallocation des flux vers les raffineurs de destination sont toutes des hypothèses que "
    "la donnée de prix ne départage pas.",
)
show(
    regime_chart(
        check.annual.assign(positif=check.annual["richness_ref"] > 0),
        "richness_ref",
        regime_col="positif",
        regime_color=ALT_COLOR,
        title="Richness médiane par année — le basculement survit au choix de paramètre",
        y_title="USD/tonne",
    )
)

# ===========================================================================
# S6 — le nombre du mail
# ===========================================================================
section(
    "S6",
    "Le nombre à mettre dans un mail",
    "Toute la discussion précédente porte sur la **richness**, qui demande un modèle de "
    "coût. Le white premium lui-même n'en demande aucun : c'est le prix que le marché met "
    "sur la transformation d'une tonne de brut en une tonne de blanc, observé, sans "
    "hypothèse d'opex ni d'énergie. C'est donc lui qu'on présente à un raffineur, pas la "
    "richness — il connaît son propre coût et fera la soustraction lui-même.",
)
finding(cost.headline)

st.markdown("**Sensibilité au paramètre, en un tableau**")
st.dataframe(
    pol_adjust_sensitivity(frame["no5"], frame["no11"]).round(2),
    width="stretch", hide_index=True,
)
scope_note(
    "L'énergie est la seule composante de coût réellement observée (Henry Hub) ; "
    f"l'intensité de {energy_intensity:.1f} mmBtu/t qui la convertit en USD/t est, elle, un "
    "paramètre. Main-d'œuvre, fret et perte de rendement sont des forfaits — aucune source "
    "publique ne les donne en série temporelle."
)

mail_question(
    f"Sur ICE No.5 contre No.11 depuis {window_start[:4]}, je trouve que le marché paie "
    f"environ {cost.market_usd_t:.0f} USD/t pour l'acte de raffiner, et surtout que ce prix "
    f"a changé de régime d'environ {check.signal_span:.0f} USD/t entre {worst_year} et "
    f"{best_year} — un basculement {check.ratio:.0f} fois plus grand que l'incertitude sur "
    "l'ajustement de polarisation, donc pas un artefact de convention. Est-ce que votre coût "
    "de raffinage tout compris est de cet ordre ? Et qu'est-ce qui a changé de votre côté "
    "entre ces deux dates ?",
    "Raffineurs de destination (Al Khaleej, ASR, Tereos, Südzucker), desks sucre de Sucden, Czarnikow, ED&F Man",
)
