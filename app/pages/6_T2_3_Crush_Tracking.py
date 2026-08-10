from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.crush_tracking import (  # noqa: E402
    CBOT_MEAL_LB_BU,
    CBOT_OIL_LB_BU,
    hedge_ratio_identity_bias,
    load_real_board_frame,
    required_yield_precision,
    yield_exposure,
)
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

st.set_page_config(page_title="T2-3 — Board crush", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# En-tête et périmètre
# ===========================================================================
page_header(
    code="T2-3",
    title="Le board crush n'est pas un prix, c'est un rendement déguisé en prix",
    subtitle=(
        "À quelle précision de rendement le board crush contraint-il une usine sans le dire "
        "— et pourquoi cette exigence se resserre exactement quand la marge se resserre"
    ),
    scope=Scope(
        unit_trap=(
            "Le board crush empile **trois unités dans une seule formule** : la fève en "
            "USD/**boisseau**, le tourteau en USD/**short ton**, l'huile en **cents/lb**. "
            "Traiter la short ton comme une tonne métrique fausse la jambe tourteau de 10 %, "
            "soit environ la moitié du crush lui-même. Mais le vrai piège de cette page est "
            "ailleurs : les coefficients 0,022 et 0,11 ne sont **pas des conversions "
            "d'unité**, ce sont des rendements (44 lb de tourteau et 11 lb d'huile par "
            "boisseau) déguisés en constantes de formule."
        ),
        conversion=(
            f"board = ({CBOT_MEAL_LB_BU:.0f} lb / 2000 lb par short ton) x tourteau\n"
            f"        + {CBOT_OIL_LB_BU:.0f} lb x huile / 100  -  fève\n"
            "      = 0,022 x tourteau + 0,11 x huile - fève      [USD/boisseau]"
        ),
        proxies=[
            "opex de trituration : forfait paramétré, aucune source publique fiable — c'est "
            "le slider qui déplace le plus la conclusion",
        ],
        out_of_scope=[
            "prix cash locaux et basis tourteau intérieur : absents de l'export, donc aucun "
            "tracking error mesuré ici — la page est construite pour ne pas en avoir besoin",
            "décalage temporel entre l'achat de la fève et la vente des produits, qui "
            "ajouterait du tracking error : l'omettre biaise **contre** la thèse de la page",
        ],
        frequency_note="Les trois jambes sont quotidiennes ; le board crush l'est donc aussi.",
        data_warnings=[
            "Le board crush CBOT n'est jamais passé sous zéro sur l'échantillon — même "
            "signature qu'en T2-5. C'est une propriété du board, pas de l'économie d'une "
            "usine : le board ne porte aucun opex.",
        ],
    ),
)

# ===========================================================================
# Paramètres
# ===========================================================================
st.sidebar.markdown("### Paramètres")
opex = st.sidebar.slider(
    "Opex de trituration (USD/boisseau)", 0.20, 1.00, 0.55, 0.01,
    help="Le paramètre qui porte le signe : il fixe la marge nette, donc l'exigence de précision.",
)
meal_gap = st.sidebar.slider("Écart de rendement tourteau (lb/bu)", -4.0, 4.0, 1.0, 0.1)
oil_gap = st.sidebar.slider("Écart de rendement huile (lb/bu)", -2.0, 2.0, 0.0, 0.1)
window_start = st.sidebar.selectbox("Fenêtre", ["2015-01-01", "2020-01-01", "2024-01-01"], index=0)

frame = load_real_board_frame(window_start)
precision = required_yield_precision(frame, opex_usd_bu=opex)
exposure = yield_exposure(frame, meal_lb_gap=meal_gap, oil_lb_gap=oil_gap, opex_usd_bu=opex)
net_margin = frame["board"] - opex

kpi_banner(
    {
        "Board crush (dernier)": f"{frame['board'].iloc[-1]:+.2f} USD/bu",
        "Marge nette médiane": f"{net_margin.median():+.2f} USD/bu",
        "Précision exigée (médiane)": f"{precision.median_lb:.1f} lb/bu",
        "…dans le décile tendu": f"{precision.tight_decile_lb:.2f} lb/bu",
        "1 lb efface la marge": f"{precision.share_below(1.0):.0%} des séances",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Deux coefficients qui ressemblent à des conversions et n'en sont pas",
    "Le board crush s'écrit `0,022 x tourteau + 0,11 x huile - fève`, et ces deux "
    "coefficients passent pour des facteurs de conversion anodins. Ils n'en sont pas : "
    f"0,022 short ton par boisseau, c'est {CBOT_MEAL_LB_BU:.0f} livres de tourteau, et 0,11, "
    f"c'est {CBOT_OIL_LB_BU:.0f} livres d'huile. Ce sont des **rendements**. Le CBOT les a "
    "figés une fois pour toutes, parce qu'un contrat à terme a besoin d'une définition qui "
    "ne bouge pas.\n\n"
    "Une usine, elle, n'a pas de rendement fixe. Le tourteau qu'elle sort dépend de la "
    "teneur en protéines des fèves, qui dépend de l'origine, de la saison et du lot. Deux "
    "points de protéine déplacent le rendement de plusieurs livres au boisseau. Un "
    "triturateur qui se couvre au board ne se contente donc pas de couvrir : il **accepte "
    f"silencieusement {CBOT_MEAL_LB_BU:.0f}/{CBOT_OIL_LB_BU:.0f} comme son propre "
    "rendement**, et garde la différence en position nue.\n\n"
    "Cette position n'a été décidée par personne. Elle est le résidu d'une convention de "
    "contrat, et sa taille en dollars n'est pas fixée par l'usine mais par le prix du "
    "tourteau.",
)
show(
    regime_chart(
        frame.assign(marge_nette=net_margin, sous_zero=net_margin < 0),
        "marge_nette",
        regime_col="sous_zero",
        regime_color=SHUT_COLOR,
        title=f"Marge nette du board crush, opex à {opex:.2f} USD/bu",
        y_title="USD/boisseau",
        annotations={"2020-03-01": "Covid", "2022-02-24": "Ukraine"},
    )
)
scope_note(
    f"Le board brut n'est jamais passé sous zéro (minimum {frame['board'].min():+.2f} USD/bu) "
    f"— une fois l'opex retranché, la marge nette l'est {(net_margin < 0).mean():.0%} du temps. "
    "Toute la différence tient à un paramètre que le board ne porte pas."
)

# ===========================================================================
# S2 — le livrable
# ===========================================================================
section(
    "S2",
    "L'inversion : la précision de rendement que le board exige de vous",
    "Mesurer le tracking error demanderait des prix cash locaux, que l'export ne contient "
    "pas. On retourne donc la question, et elle devient plus utile : **quel écart de "
    "rendement suffit à consommer toute la marge nette ?** Le nombre qui sort est en livres "
    "par boisseau — l'unité dans laquelle un exploitant de trituration pense, et sur "
    "laquelle il peut répondre immédiatement.\n\n"
    "La réponse ne se lit pas sur sa moyenne. Le seuil est proportionnel à la marge nette et "
    "inversement proportionnel au prix du tourteau : il **s'effondre quand la marge se "
    "resserre**, c'est-à-dire précisément dans le régime où la couverture devait servir à "
    "quelque chose.",
    formula="seuil_lb = (board_crush - opex) / (prix_tourteau / 2000)",
)
finding(precision.headline)

c1, c2, c3 = st.columns(3)
c1.metric("Décile de marge le plus tendu", f"{precision.tight_decile_lb:.2f} lb/bu",
          delta=f"{precision.tight_decile_pct:.1%} des {CBOT_MEAL_LB_BU:.0f} lb du board",
          delta_color="off")
c2.metric("Médiane", f"{precision.median_lb:.1f} lb/bu")
c3.metric("Décile le plus large", f"{precision.wide_decile_lb:.1f} lb/bu")

show(
    regime_chart(
        precision.frame.assign(sous_2lb=precision.frame["breakeven_lb"] <= 2.0),
        "breakeven_lb",
        regime_col="sous_2lb",
        regime_color=SHUT_COLOR,
        title="Écart de rendement tourteau qui consomme toute la marge nette",
        y_title="lb par boisseau",
        reference_lines={"1 lb/bu": 1.0, "2 lb/bu": 2.0},
    )
)
scope_note(
    f"Zones ombrées : les séances où **2 livres suffisent** — {precision.share_below(2.0):.0%} "
    f"de l'échantillon. Une livre suffit {precision.share_below(1.0):.0%} du temps. Sur une "
    f"base de {CBOT_MEAL_LB_BU:.0f} lb, une livre est un écart de "
    f"{1.0 / CBOT_MEAL_LB_BU:.1%} — personne ne connaît son rendement tourteau à cette "
    "précision-là sur des fèves qu'il n'a pas encore triturées."
)

# ===========================================================================
# S3 — la position, vue comme une position
# ===========================================================================
section(
    "S3",
    "Ce que l'écart est réellement : une position, pas une imprécision",
    "Un écart de rendement ne se rattrape pas par une constante, parce que ce n'en est pas "
    "une. C'est un produit `quantité x prix` — donc une **position ouverte dans le tourteau "
    "et dans l'huile**, portée en permanence par une usine qui se croit couverte. Sa taille "
    "varie avec le prix des produits, sans que personne ne l'ait dimensionnée.",
    formula="position = (Δlb_tourteau / 2000) x prix_tourteau + Δlb_huile x prix_huile / 100",
)
finding(exposure.headline)
show(
    regime_chart(
        exposure.frame.assign(
            depasse=exposure.frame["position_usd_bu"].abs() > exposure.frame["net_margin"]
        ),
        "position_usd_bu",
        regime_col="depasse",
        regime_color=SHUT_COLOR,
        title=f"Position nue laissée par {meal_gap:+.1f} lb de tourteau et {oil_gap:+.1f} lb d'huile",
        y_title="USD/boisseau",
    )
)
scope_note(
    "Zones ombrées : les séances où cette position dépasse à elle seule la marge nette "
    "entière. Le ratio à la marge est calculé avec un plancher de 0,05 USD/bu au "
    "dénominateur — sans lui, les jours de marge quasi nulle produiraient des ratios "
    "arbitrairement grands qui écraseraient la lecture."
)

# ===========================================================================
# S4 — le contre-intuitif
# ===========================================================================
correlation = float(net_margin.corr(frame["meal"], method="spearman"))
section(
    "S4",
    "Le résultat qui contredit l'intuition — et qu'on garde tel quel",
    "L'intuition naturelle est que la position non couverte grossit quand la marge se "
    "resserre, ce qui composerait les deux problèmes. **C'est faux, et il faut le dire.** "
    f"La corrélation de rang entre la marge nette et le prix du tourteau vaut "
    f"{correlation:+.2f} : le tourteau est la principale recette du crush, donc un tourteau "
    "cher va avec une marge large. La position nue est donc la plus grosse quand la marge "
    "est la plus confortable — le désalignement s'amortit lui-même en partie.\n\n"
    "Ce qui reste vrai, et qui est le vrai résultat, est plus fin : dans le régime tendu la "
    "position est petite en dollars, mais la marge l'est encore plus. C'est le **rapport** "
    "qui se dégrade, pas la position. Une usine qui surveille son exposition en dollars "
    "absolus ne verra jamais le problème arriver.",
)
comparison = pd.DataFrame(
    {
        "régime": ["décile tendu", "médiane", "décile large"],
        "marge nette (USD/bu)": [
            net_margin.quantile(0.10), net_margin.median(), net_margin.quantile(0.90),
        ],
        "prix tourteau (USD/st)": [
            frame.loc[net_margin <= net_margin.quantile(0.10), "meal"].median(),
            frame["meal"].median(),
            frame.loc[net_margin >= net_margin.quantile(0.90), "meal"].median(),
        ],
        "précision exigée (lb/bu)": [
            precision.tight_decile_lb, precision.median_lb, precision.wide_decile_lb,
        ],
    }
)
st.dataframe(comparison.round(2), width="stretch", hide_index=True)

# ===========================================================================
# S5 — pourquoi pas une régression
# ===========================================================================
bias = hedge_ratio_identity_bias(
    frame, meal_lb_gap=meal_gap if meal_gap else 1.0, oil_lb_gap=oil_gap, opex_usd_bu=opex
)
section(
    "S5",
    "Pourquoi cette page ne régresse rien",
    "Le réflexe serait d'estimer un ratio de couverture en régressant la variation de la "
    "marge d'usine sur celle du board crush. C'est une mauvaise idée, et pas pour la raison "
    "qu'on croit. La marge d'usine s'écrit **exactement** `board + écart_rendement - opex` : "
    "régresser l'une sur l'autre, c'est régresser une grandeur sur une de ses propres "
    "composantes. Le coefficient obtenu vaut `1 + cov(Δécart, Δboard) / var(Δboard)`, et le "
    "second terme n'est pas nul puisque l'écart est lui-même fait de tourteau et d'huile.\n\n"
    "L'honnêteté oblige à dire que cette contamination est **petite** — de l'ordre du "
    "pourcent. Le danger n'est pas sa taille : c'est ce qu'un praticien ferait du "
    "coefficient. L'appliquer revient à couvrir son écart de rendement avec **davantage de "
    f"board crush**, alors que le board est un panier {CBOT_MEAL_LB_BU:.0f}/{CBOT_OIL_LB_BU:.0f} "
    "figé. On ne couvre pas un écart à 44/11 avec l'instrument dont l'hypothèse de rendement "
    "l'a créé. La bonne couverture, c'est des jambes tourteau et huile dimensionnées "
    "séparément — ce que la page calcule directement en S3, sans régression.",
    formula="beta = 1 + cov(Δécart_rendement, Δboard) / var(Δboard)",
)
diagnostic_note(bias.headline)
scope_note(
    "Ce paragraphe est hérité de T2-1 (basis contre flat price), sorti du portefeuille faute "
    "de séries cash dans l'export. Le résultat, lui, restait valable et s'applique ici sans "
    "modification."
)

# ===========================================================================
# S6 — sensibilité
# ===========================================================================
section(
    "S6",
    "Le paramètre qui porte le signe",
    "Toute la page dépend d'un nombre qu'aucune source publique ne donne proprement : "
    "l'opex de trituration. Il ne déplace pas la conclusion à la marge, il la commande — "
    "parce qu'il fixe la marge nette, dont le seuil de précision est directement "
    "proportionnel. Voici donc la grille complète plutôt qu'un chiffre choisi.",
)
grid = pd.DataFrame(
    [
        {
            "opex (USD/bu)": value,
            "marge nette médiane": float((frame["board"] - value).median()),
            "part sous zéro": float(((frame["board"] - value) < 0).mean()),
            "précision médiane (lb/bu)": required_yield_precision(frame, opex_usd_bu=value).median_lb,
            "décile tendu (lb/bu)": required_yield_precision(frame, opex_usd_bu=value).tight_decile_lb,
            "1 lb suffit (part)": required_yield_precision(frame, opex_usd_bu=value).share_below(1.0),
        }
        for value in np.arange(0.30, 0.91, 0.10)
    ]
)
st.dataframe(
    grid.style.format(
        {
            "opex (USD/bu)": "{:.2f}", "marge nette médiane": "{:+.2f}",
            "part sous zéro": "{:.0%}", "précision médiane (lb/bu)": "{:.2f}",
            "décile tendu (lb/bu)": "{:.2f}", "1 lb suffit (part)": "{:.0%}",
        }
    ),
    width="stretch", hide_index=True,
)

mail_question(
    f"Sur le board crush CBOT depuis {window_start[:4]}, avec un opex de {opex:.2f} USD/bu, je "
    f"trouve qu'un écart de {precision.tight_decile_lb:.1f} lb de tourteau par boisseau suffit "
    "à effacer toute la marge nette dans le décile de marge le plus tendu — soit "
    f"{precision.tight_decile_pct:.1%} des {CBOT_MEAL_LB_BU:.0f} lb que le contrat suppose. "
    "De combien votre rendement tourteau réel s'écarte-t-il des 44 lb du board sur une "
    "campagne, et est-ce que quelqu'un chez vous couvre cet écart séparément — ou est-ce "
    "qu'il reste dans le résultat ?",
    "Exploitants de trituration (ADM, Bunge, Cargill, LDC, CHS), risk managers oléagineux",
)
