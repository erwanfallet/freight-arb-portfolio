from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.china_soy import (  # noqa: E402
    BUSHELS_PER_TONNE_SOYBEAN,
    DEFAULT_FREIGHT_USD_T,
    affordable_origination_budget,
    impossible_windows,
    load_real_crush_frame,
)
from agri.core.fmt import fr, fr_pct  # noqa: E402
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (  # noqa: E402
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
    waterfall_chart,
)

st.set_page_config(page_title="T3-4 — Chine soja", layout="wide")

if not DEFAULT_PATH.exists():
    st.error(f"Fichier Bloomberg introuvable : {DEFAULT_PATH}")
    st.stop()

# ===========================================================================
# En-tête et périmètre
# ===========================================================================
page_header(
    code="T3-4",
    title="Les fenêtres où aucune origine ne fonctionne",
    subtitle=(
        "Plutôt que de tester si les achats chinois ont une signature politique — ce qui "
        "demande des données d'enchères que personne ne publie —, on date les périodes où "
        "un achat commercial était arithmétiquement impossible"
    ),
    scope=Scope(
        unit_trap=(
            "**Trois pièges empilés dans un seul calcul.** Le CBOT cote en USD par "
            "**boisseau** (60 lb), la DCE en CNY par **tonne**. Les prix chinois sont "
            "**TTC** alors que la marge se calcule HT — et la TVA sur les oléagineux "
            "importés n'est pas celle des produits transformés, elle dépend du produit **et "
            "de la date**. Enfin les droits de douane s'appliquent à la valeur **CNF**, pas "
            "au FOB. Une seule de ces trois erreurs déplace la marge de plusieurs dizaines "
            "de CNY/t et inverse la conclusion."
        ),
        conversion=(
            "recette_HT    = (0,785 x tourteau_DCE + 0,185 x huile_DCE) / (1 + TVA)\n"
            "CNF_max_USD_t = [(recette_HT - transformation) / (1 + droit)] / USDCNY\n"
            "budget        = CNF_max_USD_t - CBOT_usd_bu x 36,7437"
        ),
        proxies=[
            "rendements de trituration chinois (0,785 tourteau / 0,185 huile) : barème "
            "standard, pas une mesure d'usine",
            "coût de transformation et taux de TVA : paramétrés, à revérifier par produit "
            "et par date",
        ],
        out_of_scope=[
            "les achats de réserve d'État eux-mêmes : Sinograin ne publie pas de série "
            "temporelle d'enchères, et l'export n'en contient aucune — c'est précisément "
            "pourquoi la page raisonne sur les prix seuls (voir S2)",
            "le basis FOB d'origine et le fret : ils **sortent** du calcul au lieu d'y "
            "entrer, ce qui est l'astuce de la page plutôt qu'une limite",
        ],
        frequency_note="CBOT, DCE et USDCNY sont quotidiens ; le budget l'est donc aussi.",
        data_warnings=[
            "La marge de crush chinoise sert aussi de support à T2-5. Les deux pages "
            "utilisent la même série mais n'en tirent pas la même grandeur : T2-5 la traite "
            "comme un processus à arrêter, celle-ci comme une contrainte budgétaire.",
        ],
    ),
)

# ===========================================================================
# Paramètres
# ===========================================================================
st.sidebar.markdown("### Paramètres")
freight_reference = st.sidebar.slider(
    "Fret de référence Chine (USD/t)", 20.0, 90.0, DEFAULT_FREIGHT_USD_T, 5.0,
    help="Sert de seuil de lecture, pas d'entrée du calcul : le budget ne le suppose pas.",
)
threshold = st.sidebar.slider("Seuil du calendrier (USD/t)", -20.0, 60.0, 0.0, 5.0)
window_start = st.sidebar.selectbox("Fenêtre", ["2018-01-01", "2022-01-01"], index=0)

budget = affordable_origination_budget(
    start=window_start, freight_reference_usd_t=freight_reference
)
windows = impossible_windows(budget, threshold_usd_t=threshold)
crush = load_real_crush_frame(start=window_start)

kpi_banner(
    {
        "Budget médian": f"{fr(budget.median_budget, 0)} USD/t",
        "Budget au dernier cours": f"{fr(budget.last_budget, 0)} USD/t",
        "Budget négatif": fr_pct(budget.share_impossible, 1),
        "Sous le fret seul": fr_pct(budget.share_below_freight),
        "Fenêtres datées": f"{len(windows)}",
    }
)

# ===========================================================================
# S1
# ===========================================================================
section(
    "S1",
    "Le test propre qu'on ne peut pas passer",
    "La question d'origine est bonne et le test qui y répond est binaire : si les achats de "
    "réserve d'État se concentrent dans les quintiles de marge de crush les plus **bas**, ce "
    "sont des achats que l'économie du crush interdit — signature politique. S'ils se "
    "concentrent dans les quintiles hauts, c'est de la rotation de stock opportuniste. Le "
    "signe d'un seul coefficient tranche.\n\n"
    "Il manque une chose : la série des achats. Sinograin ne publie pas ses enchères en "
    "série temporelle, et l'export n'en contient aucune. Le test ne peut pas être passé, et "
    "le simuler sur données fabriquées ne prouverait que la capacité du code à retrouver ce "
    "qu'on y a mis.\n\n"
    "On change donc d'angle sans changer de question. Au lieu de demander *où* les achats se "
    "situent dans la distribution des marges, on identifie les périodes où **aucun achat "
    "commercial n'était possible, quelle que soit l'origine**. Dans ces fenêtres, la "
    "signature n'a plus besoin d'être estimée : toute cargaison arrivée est non commerciale "
    "par construction.",
)

# ===========================================================================
# S2 — le mécanisme
# ===========================================================================
section(
    "S2",
    "Le budget d'origination, et pourquoi il ne suppose ni basis ni fret",
    "La marge de crush chinoise borne par le haut ce qu'un triturateur peut payer pour une "
    "tonne de fève **rendue à quai**. On retranche de cette borne le CBOT converti à la "
    "tonne, et il reste le budget dont un originateur dispose pour aller chercher la fève : "
    "le basis d'origine plus le fret.\n\n"
    "Le point important est ce que ce calcul **ne** contient pas. Le basis FOB et le fret "
    "sont les deux termes que l'export ne fournit pas, et qui obligeaient jusqu'ici à des "
    "forfaits. Ici ils sortent du calcul au lieu d'y entrer — le budget est ce qu'il reste "
    "*pour* eux, pas une quantité qui les suppose. La conclusion de la page ne dépend donc "
    "d'aucune hypothèse de fret.",
    formula=(
        "budget = CNF_max_USD_t - CBOT_usd_bu x 36,7437\n"
        "         ↑ aucune hypothèse de basis ni de fret n'entre ici"
    ),
)
finding(budget.headline)
scope_note(
    "À dire avant qu'un lecteur ne le remarque : le budget **est** la marge de crush, "
    "redénominée en USD/t et débarrassée des deux forfaits. L'identité est exacte — "
    "`marge = (1 + droit) × USDCNY × (budget − forfait)` — et vérifiée au flottant près par "
    "les tests. Le budget n'apporte donc aucune information nouvelle : il retire deux "
    "paramètres arbitraires. C'est exactement ce qui rend son passage à zéro interprétable "
    "là où celui de la marge ne l'est pas — le zéro de la marge dépend du forfait qu'on a "
    "choisi, celui du budget ne dépend de rien."
)
show(
    regime_chart(
        budget.frame,
        "budget_usd_t",
        regime_col="impossible",
        regime_color=SHUT_COLOR,
        title="Budget disponible pour le basis d'origine et le fret",
        y_title="USD par tonne",
        reference_lines={f"fret seul : {fr(freight_reference, 0)}": freight_reference},
        annotations={"2022-02-24": "Ukraine", "2023-06-01": "creux 2023"},
    )
)
scope_note(
    "Zones ombrées : budget négatif. La ligne pointillée est le fret de référence — sous "
    "elle, le fret consomme tout le budget et il faudrait acheter la fève **sous** le CBOT "
    "à l'origine, ce qu'aucun exportateur ne fait durablement."
)

# ===========================================================================
# S3 — LE LIVRABLE
# ===========================================================================
section(
    "S3",
    "Le calendrier — des dates, pas un coefficient",
    "C'est le livrable de la page, et il a une propriété qu'un coefficient n'a pas : il se "
    "confronte directement au carnet d'un originateur. La question devient « avez-vous "
    "chargé pendant ces fenêtres ? », et elle se répond par oui ou par non.\n\n"
    "La concentration temporelle est le fait saillant. Toutes les fenêtres de budget "
    "strictement négatif tombent en **2023**, la plus longue courant du 7 juin au 5 juillet. "
    "Ce n'est pas une dispersion de bruit autour de zéro : c'est un épisode, et il a une "
    "date de début et de fin.",
)
if len(windows):
    display = windows[
        [c for c in ("start", "end", "duration_days", "min_depth", "mean_depth") if c in windows.columns]
    ].copy()
    display.columns = [
        {"start": "début", "end": "fin", "duration_days": "durée (jours)",
         "min_depth": "budget le plus bas", "mean_depth": "budget moyen"}.get(c, c)
        for c in display.columns
    ]
    # Arrondir le frame entier avertit sur les colonnes de dates ; on cible les numériques.
    numeric = display.select_dtypes("number").columns
    display[numeric] = display[numeric].round(1)
    st.dataframe(display, width="stretch", hide_index=True)
    finding(
        f"{len(windows)} fenêtre(s) sous {fr(threshold, 0)} USD/t, pour un total de "
        f"{fr(float(windows['duration_days'].sum()), 0)} jours. La plus longue dure "
        f"{fr(float(windows['duration_days'].max()), 0)} jours."
    )
else:
    diagnostic_note(
        f"Aucune fenêtre sous {fr(threshold, 0)} USD/t sur cette période — le seuil est "
        "peut-être trop bas, ou la fenêtre trop courte."
    )

# ===========================================================================
# S4 — d'où vient le budget
# ===========================================================================
last = budget.frame.iloc[-1]
crush_last = crush.iloc[-1]
section(
    "S4",
    "D'où vient le chiffre, poste par poste",
    "Un budget qui sort d'une chaîne de trois conversions mérite d'être ouvert. Le waterfall "
    "ci-dessous part de la recette hors taxe du triturateur au dernier cours et descend "
    "jusqu'au budget disponible. Un praticien qui conteste le résultat peut pointer la ligne "
    "dont il conteste le niveau, plutôt que le total.",
)
show(
    waterfall_chart(
        {
            "recette HT (tourteau + huile)": float(crush_last["revenue_ex_vat"] / last["usdcny"]),
            "transformation": -120.0 / float(last["usdcny"]),
            "droit de douane": float(
                last["cnf_max_usd_t"] - (crush_last["revenue_ex_vat"] - 120.0) / last["usdcny"]
            ),
            "CBOT converti à la tonne": -float(last["cbot_usd_t"]),
        },
        total_label="budget basis + fret",
        title=f"Décomposition au {budget.frame.index[-1]:%d/%m/%Y}",
        y_title="USD par tonne",
    )
)
c1, c2, c3 = st.columns(3)
c1.metric("CNF maximal finançable", f"{fr(last['cnf_max_usd_t'], 0)} USD/t")
c2.metric("CBOT à la tonne", f"{fr(last['cbot_usd_t'], 0)} USD/t")
c3.metric("Budget restant", f"{fr(last['budget_usd_t'], 0)} USD/t")
scope_note(
    f"Conversion boisseau → tonne : {fr(BUSHELS_PER_TONNE_SOYBEAN, 4)} boisseaux par tonne "
    "de soja, dérivée du poids réglementaire de 60 lb par boisseau plutôt qu'écrite en dur."
)

# ===========================================================================
# S5
# ===========================================================================
section(
    "S5",
    "Ce que la page ne prétend pas montrer",
    "Le budget dit ce qu'un triturateur **peut** payer, pas ce qui a été payé. Il ne "
    "démontre donc pas qu'une cargaison politique est arrivée dans les fenêtres datées — il "
    "établit que si une cargaison y est arrivée, elle ne s'expliquait pas par l'économie du "
    "crush. C'est une charge de la preuve déplacée, pas une preuve.\n\n"
    "Deux réserves supplémentaires, qui vont toutes deux **contre** la thèse et qu'il faut "
    "donc énoncer. D'abord un triturateur intégré peut accepter une marge nulle sur le crush "
    "s'il gagne ailleurs dans la chaîne — le budget serait alors sous-estimé. Ensuite les "
    "cargaisons se fixent des semaines avant l'arrivée : une fenêtre impossible à l'arrivée "
    "pouvait être finançable à la fixation. Le calendrier doit donc être confronté à des "
    "**dates de fixation**, pas à des dates de déchargement.",
)
diagnostic_note(
    "Le décalage fixation/arrivée est la limite la plus sérieuse de cette page. Elle n'est "
    "pas corrigeable ici — il faudrait un calendrier de fixation que seul un originateur "
    "possède — et c'est exactement ce qui fait de la question du mail une vraie question."
)

mail_question(
    f"En calculant ce que la marge de crush chinoise laisse pour le basis d'origine et le "
    f"fret, je trouve {fr_pct(budget.share_impossible, 1)} de séances où ce budget est "
    f"négatif — c'est-à-dire où même une fève gratuite et un fret gratuit ne rendraient pas "
    f"le crush rentable — toutes concentrées en 2023, dont une fenêtre de "
    f"{fr(float(windows['duration_days'].max()), 0) if len(windows) else '—'} jours. "
    "Est-ce que vous avez fixé des cargaisons Chine pendant ces fenêtres-là ? Et si oui, "
    "est-ce que la marge de crush était vraiment la contrainte, ou est-ce qu'un autre "
    "maillon de la chaîne portait le résultat ?",
    "Origination oléagineux (COFCO, Sinograin, Bunge, LDC, Cargill), desks soja Chine",
)
