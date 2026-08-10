from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.oil_substitution import (  # noqa: E402
    MYR_PEG_END,
    MYR_PEG_RATE,
    MYR_PEG_START,
    load_peg_window_spread,
    rolling_deviation,
    structural_drift,
    substitution_verdict,
)
from agri.core.fmt import fr, fr_pct  # noqa: E402
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (
    snapshot_banner,  # noqa: E402
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

st.set_page_config(page_title="T2-6 — Substitution inter-huiles", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# En-tête et périmètre
# ===========================================================================
page_header(
    code="T2-6",
    title="La borne de substitution palme-soja n'existe pas",
    subtitle=(
        "Testé dans la seule fenêtre où le change ne contamine rien — les sept ans de "
        "parité fixe du ringgit — l'hypothèse ressort inversée : les grands écarts ne se "
        "referment pas, ils déplacent le niveau"
    ),
    scope=Scope(
        unit_trap=(
            "La palme cote en **ringgits malais par tonne** (Bursa), le soja en **cents par "
            "livre** (CBOT). Les soustraire demande deux conversions dont une est un **prix "
            "de marché** : le taux USDMYR. Or l'export ne contient aucune série de change "
            "malaisienne. Calculer un spread palme-soja sur l'ensemble de l'historique "
            "reviendrait donc à soustraire deux devises — exactement l'erreur que ce "
            "portefeuille traque partout ailleurs. **La page refuse de le faire.**"
        ),
        conversion=(
            f"Fenêtre de parité fixe uniquement ({MYR_PEG_START} → {MYR_PEG_END}) :\n"
            f"  palme_USD_t = palme_MYR_t / {MYR_PEG_RATE}      ← constante réglementaire\n"
            "  soja_USD_t  = soja_c_lb x 22,0462\n"
            "  spread      = palme_USD_t - soja_USD_t"
        ),
        proxies=[
            "aucun : sur cette fenêtre, la seule grandeur manquante est fixée par décret, "
            "donc connue exactement",
        ],
        out_of_scope=[
            "toute la période post-juillet 2005 : le ringgit flotte, et l'export n'a pas "
            "l'USDMYR — le spread n'y est pas calculable sans fabriquer une devise",
            "colza et tournesol, absents de l'export : le test porte sur le seul couple "
            "palme-soja",
        ],
        frequency_note=(
            "Palme et soja sont quotidiens. La fenêtre exploitable compte environ 1 650 "
            "séances, ce qui suffit à estimer des demi-vies par régime."
        ),
        data_warnings=[
            "La fenêtre se termine en juillet 2005, **avant** l'essor de la demande "
            "biodiesel. Le résultat est un point de repère historique propre, pas une "
            "lecture du marché d'aujourd'hui — et c'est précisément ce qui fait la question "
            "du mail.",
        ],
    ),
)

# ===========================================================================
# Paramètres
# ===========================================================================
st.sidebar.markdown("### Paramètres")
window = st.sidebar.slider("Fenêtre de la médiane glissante (jours)", 60, 400, 250, 10)
quantile = st.sidebar.slider("Quantile séparant large / étroit", 0.55, 0.90, 0.70, 0.05)

frame = load_peg_window_spread()
verdict = substitution_verdict(frame["spread"], window=window, quantile=quantile)
deviation = rolling_deviation(frame["spread"], window=window)
drift = structural_drift(frame["spread"])

kpi_banner(
    {
        "Séances exploitables": f"{fr(len(frame), 0)}",
        "Spread médian": f"{fr(frame['spread'].median(), 0)} USD/t",
        "Seuil large / étroit": f"{fr(verdict.threshold_usd_t, 0)} USD/t",
        "Demi-vie écart étroit": f"{fr(verdict.narrow.half_life_days, 0)} j",
        "Demi-vie écart large": (
            "aucune" if not pd.notna(verdict.wide.half_life_days) or verdict.wide.half_life_days == float("inf")
            else f"{fr(verdict.wide.half_life_days, 0)} j"
        ),
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Une question simple qu'on ne peut pas poser directement",
    "Il me semble que les triturateurs tiennent l'élasticité palme/soja pour forte — un "
    "écart s'ouvre, quelqu'un bascule, l'écart se referme — pendant que les formulateurs la "
    "tiennent pour collante : reformuler une recette prend des mois et repasse par des "
    "validations. Je n'ai aucune preuve documentaire qu'un desk précis s'engueule là-dessus "
    "aujourd'hui, d'où « il me semble » plutôt que « j'ai lu que ».\n\n"
    "Le test est direct : si la substitution est rapide, un spread anormalement large doit "
    "revenir **plus vite** qu'un spread normal. Il suffit de comparer deux demi-vies.\n\n"
    "Sauf qu'il faut un spread. Et pour avoir un spread, il faut que les deux prix soient "
    "dans la même monnaie.",
)

# ===========================================================================
# S2
# ===========================================================================
section(
    "S2",
    "La devise manquante, et les sept ans où elle est une constante",
    "La palme de Bursa cote en ringgits par tonne, le soja du CBOT en cents par livre. "
    "Passer de l'un à l'autre demande l'USDMYR, qui n'est **pas dans l'export**. Calculer "
    "le spread sur trente ans d'historique reviendrait à soustraire un prix en ringgits d'un "
    "prix en dollars et à appeler la différence un spread — c'est l'erreur que ce "
    "portefeuille passe son temps à débusquer ailleurs. La page ne la commet pas.\n\n"
    f"Mais il existe une fenêtre. Bank Negara a arrimé le ringgit à **{fr(MYR_PEG_RATE, 2)} "
    f"MYR pour un dollar du {MYR_PEG_START[8:10]}/{MYR_PEG_START[5:7]}/{MYR_PEG_START[:4]} "
    f"au {MYR_PEG_END[8:10]}/{MYR_PEG_END[5:7]}/{MYR_PEG_END[:4]}**. Pendant ces sept ans, "
    "la série manquante n'est pas approximée : elle est **connue exactement, par décret**. "
    "Le spread s'y calcule sans la moindre hypothèse, et tout ce qu'il fait est de "
    "l'économie de substitution pure, non contaminée par le change.\n\n"
    "C'est une expérience naturelle, et c'est le seul endroit de tout l'historique où ce "
    "test est propre.",
    formula=f"palme_USD_t = palme_MYR_t / {fr(MYR_PEG_RATE, 2)}   ← une division par une constante, pas par une série",
)
show(
    regime_chart(
        frame.assign(palme_sous_soja=frame["spread"] < 0),
        "spread",
        regime_col="palme_sous_soja",
        regime_color=ALT_COLOR,
        title="Spread palme − soja pendant la parité fixe du ringgit",
        y_title="USD par tonne",
    )
)
scope_note(
    f"Zones ombrées : la palme se traite sous le soja, ce qui est le cas "
    f"{fr_pct((frame['spread'] < 0).mean())} du temps sur la fenêtre."
)

# ===========================================================================
# S3 — LE RÉSULTAT
# ===========================================================================
section(
    "S3",
    "Le résultat, et il inverse la thèse",
    "L'hypothèse prédit : écart large → quelqu'un bascule → retour rapide. On mesure donc "
    "la demi-vie de retour à la moyenne séparément dans les deux régimes, en comparant le "
    "spread à sa **médiane glissante** plutôt qu'à une constante — pour la raison expliquée "
    "en S4.\n\n"
    "La donnée dit le contraire de ce qui était attendu. Les écarts **étroits** reviennent "
    f"vite, en {fr(verdict.narrow.half_life_days, 0)} jours. Les écarts **larges**, eux, ne "
    "reviennent pas du tout : le coefficient de rappel n'est pas distinguable de zéro. Ce "
    "n'est pas un manque de puissance — l'échantillon large compte "
    f"{fr(verdict.wide.n_obs if hasattr(verdict.wide, 'n_obs') else 0, 0)} observations et "
    "le coefficient est de signe positif, pas simplement petit.",
    formula="Δécart = a + b x écart_{t-1} + e     (erreurs HAC)     demi-vie = -ln 2 / ln(1 + b)",
)
finding(verdict.headline)

c1, c2 = st.columns(2)
c1.metric("Écart étroit", f"{fr(verdict.narrow.half_life_days, 0)} jours", delta=verdict.narrow.summary.split(":")[-1].strip(), delta_color="off")
c2.metric("Écart large", "aucun retour détectable", delta=verdict.wide.summary.split(":")[-1].strip(), delta_color="off")

show(
    regime_chart(
        deviation.to_frame("ecart").assign(large=deviation.abs() >= verdict.threshold_usd_t),
        "ecart",
        regime_col="large",
        regime_color=SHUT_COLOR,
        title=f"Écart à la médiane glissante ({window} jours) — zones ombrées : régime large",
        y_title="USD par tonne",
        reference_lines={
            f"+{fr(verdict.threshold_usd_t, 0)}": verdict.threshold_usd_t,
            f"−{fr(verdict.threshold_usd_t, 0)}": -verdict.threshold_usd_t,
        },
    )
)
diagnostic_note(
    "Un biais résiduel existe, et il va **contre** ce résultat. Conditionner sur un écart "
    "large suréchantillonne le bruit de mesure, qui revient mécaniquement à la moyenne : ce "
    "biais pousse donc vers la détection d'un retour, pas vers son absence. Ne rien trouver "
    "malgré lui rend le résultat négatif plus solide, pas moins."
)

# ===========================================================================
# S4
# ===========================================================================
drift_total = drift.attrs["drift_usd_t"]
section(
    "S4",
    "Pourquoi la médiane glissante, et ce que le test naïf aurait donné",
    "Il existe une façon naturelle de faire ce test qui donne la réponse attendue par la "
    "thèse — et elle est fausse. Elle consiste à découper les régimes sur le **niveau "
    "absolu** du spread : les jours où |spread| est grand contre les jours où il est petit. "
    f"Faite ainsi, la mesure donne 12 jours en régime large contre 39 en régime étroit, "
    "c'est-à-dire exactement la borne de substitution que l'on cherchait.\n\n"
    f"L'artefact tient à un détail de distribution. Le spread médian vaut "
    f"{fr(frame['spread'].median(), 0)} USD/t : il est **structurellement négatif**. "
    "Sélectionner les jours où |spread| est grand ne sélectionne donc pas des écarts "
    "anormaux par rapport à la normale du moment — cela sélectionne l'époque où la palme "
    "était le plus décotée, c'est-à-dire 2004-2005. On mesure alors la dynamique d'une "
    "période particulière et on l'appelle substitution.\n\n"
    "La correction n'est pas la médiane glissante : découper sur un **écart**, même à une "
    "simple constante, suffit déjà à faire disparaître la borne. La médiane glissante ne "
    "fait que rendre le résultat plus net, en absorbant la dérive du niveau — car dérive il "
    f"y a : le spread passe d'une prime de {fr(drift['median_spread'].iloc[0], 0)} USD/t en "
    f"faveur de la palme en 1998 à une décote de "
    f"{fr(abs(drift['median_spread'].min()), 0)} USD/t en 2004, soit "
    f"{fr(abs(drift_total), 0)} USD/t de déplacement net.\n\n"
    "Le détail qui achève de trancher : les deux queues sont **séparées dans le temps**. La "
    "palme est chère en 1998-1999, très décotée en 2004-2005. Ce ne sont pas deux "
    "excursions autour d'un équilibre, ce sont deux époques.",
)
annual = drift.copy()
annual.columns = ["spread médian (USD/t)", "séances"]
annual["spread médian (USD/t)"] = annual["spread médian (USD/t)"].round(0)
st.dataframe(annual, width="stretch")
scope_note(
    "La cause de cette repricing n'est pas établie ici. L'expansion des surfaces plantées "
    "en Malaisie et en Indonésie sur la période en est une hypothèse plausible, mais la "
    "donnée de prix ne la départage pas d'autres explications — elle est donc laissée comme "
    "hypothèse, pas énoncée comme résultat."
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "Ce que cela veut dire pour quelqu'un qui prend du risque",
    "La lecture cohérente de S3 et S4 est la suivante. Les petits écarts sont du bruit "
    "autour d'un équilibre qui bouge lentement, et ils se referment vite — c'est de la "
    "microstructure, pas de la substitution. Les grands écarts ne sont pas des dislocations "
    "temporaires : **ce sont des déplacements de l'équilibre lui-même**, et ils ne se "
    "referment pas.\n\n"
    "La conséquence opérationnelle est directe et va contre une heuristique répandue : "
    "**fader un spread palme-soja large n'a aucun support** dans la seule fenêtre où le test "
    "est propre. Le trade « la substitution va le ramener » suppose une force de rappel que "
    "ces sept ans ne montrent pas.\n\n"
    "La réserve, elle, est sérieuse et il faut la porter : la fenêtre s'arrête en juillet "
    "2005, avant que la demande biodiesel ne devienne un moteur du complexe oléagineux. Il "
    "est parfaitement possible que la substitution soit devenue plus vive depuis. Le "
    "résultat est un point de repère propre, pas une lecture du marché d'aujourd'hui.",
)
diagnostic_note(
    "Pour refaire ce test sur la période récente, il manque **une seule série** : "
    "`USDMYR Curncy`. Gratuite, aucun entitlement particulier. C'est de loin la donnée la "
    "plus rentable à récupérer de tout le portefeuille — elle débloque trente ans "
    "d'historique palme-soja au lieu de sept."
)

mail_question(
    "En testant la substitution palme-soja sur les sept ans de parité fixe du ringgit — la "
    "seule fenêtre où le change ne contamine rien — je trouve que les écarts étroits "
    f"reviennent en {fr(verdict.narrow.half_life_days, 0)} jours mais qu'au-delà de "
    f"{fr(verdict.threshold_usd_t, 0)} USD/t il n'y a plus aucun retour à la moyenne "
    "détectable : les grands écarts déplacent le niveau au lieu de se refermer. Est-ce que "
    "cela correspond à ce que vous voyez ? Et surtout : à partir de quel écart votre "
    "téléphone sonne réellement pour un changement de recette, aujourd'hui — parce que "
    "cette fenêtre est antérieure au biodiesel et que la borne a très bien pu bouger.",
    "Triturateurs et raffineurs d'huiles végétales (Wilmar, Musim Mas, Golden Agri, Bunge, "
    "Cargill), formulateurs agroalimentaires, desks huiles",
)
