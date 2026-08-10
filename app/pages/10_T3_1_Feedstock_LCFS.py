from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR.parent / "src"))
sys.path.insert(0, str(_APP_DIR))

from agri.chains.feedstock_lcfs import (  # noqa: E402
    CENTS_PER_USD,
    LCFS_PROGRAM_HIGH_USD_T,
    LCFS_PROGRAM_LOW_USD_T,
    SOYOIL_DOMESTIC,
    SRE_WARNING,
    Feedstock,
    calibration_gap_45z,
    crush_from_soyoil_lb,
    discount_burden,
    import_penalty,
    lcfs_neutral_price,
    load_soyoil_usd_lb,
    penalty_bounds,
    structural_exit,
    winner_grid,
)
from agri.data.bloomberg_loader import DEFAULT_PATH  # noqa: E402
from page_template import (
    snapshot_banner,  # noqa: E402
    ALT_COLOR,
    Scope,
    diagnostic_note,
    finding,
    kpi_banner,
    mail_question,
    page_header,
    regime_chart,
    scope_note,
    section,
    sensitivity_heatmap,
    show,
)

st.set_page_config(page_title="T3-1 — Feedstock LCFS", layout="wide")

_LIVE = snapshot_banner()

# ===========================================================================
# En-tête et périmètre
# ===========================================================================
page_header(
    code="T3-1",
    title="Deux subventions qui se contredisent",
    subtitle=(
        "Combien de cents sous le soyoil l'UCO importé doit-il se vendre, uniquement pour "
        "compenser un crédit d'impôt qu'il n'a pas le droit de réclamer — et pourquoi le "
        "prix du crédit LCFS ne peut pas trancher le débat"
    ),
    scope=Scope(
        unit_trap=(
            "Trois unités s'empilent dans une seule marge, et **deux ne sont pas des unités "
            "de prix** : le carburant se vend au gallon, le feedstock s'achète à la livre "
            "via un rendement d'environ 7,6 lb/gal, et le crédit LCFS se cote en USD par "
            "**tonne de CO2e** — il ne devient des cents par gallon qu'après passage par une "
            "intensité carbone en gCO2e/MJ et un contenu énergétique de 134,47 MJ/gal. "
            "Second piège, plus sournois : le soyoil CBOT cote en **cents** par livre alors "
            "que tout le calcul est en **USD** par livre. Le facteur 100 est du même ordre "
            "que le résultat cherché, donc l'oublier ne casse rien de visible — ça produit "
            "un nombre plausible et faux."
        ),
        conversion=(
            "valeur_LCFS_usd_gal = LCFS_usd_t x (CI_std - CI_f) x EER x 134,47e-6\n"
            "crédit_45Z_usd_gal  = 1,00 x max(0, (50 - CI_f)/50)   si nord-américain, sinon 0\n"
            "décote_requise_usd_lb = [crédit_45Z(dom) - valeur_LCFS(écart CI)] / rendement"
        ),
        proxies=[
            "prix du crédit LCFS : **absent de l'export Bloomberg**, publié gratuitement par "
            "CARB — traité ici comme un axe, jamais comme une série (voir S3)",
            "prix de l'UCO importé : inaccessible sans abonnement Platts/PGA — la page est "
            "construite pour ne jamais en avoir besoin (voir S2)",
            "plancher de collecte UCO rendu USGC : le seul paramètre non observable, et "
            "délibérément celui qu'on demande à l'interlocuteur (S5)",
        ],
        out_of_scope=[
            "prix du diesel, prix du RIN D4, opex et ROI de l'usine : ils s'annulent entre "
            "les deux filières et n'entrent dans aucun résultat de la page — c'est démontré "
            "en S2, pas supposé",
            "contraintes logistiques de terminal et contrats d'approvisionnement existants",
        ],
        frequency_note=(
            "Le soyoil CBOT est quotidien. Le crédit LCFS est publié mensuellement par CARB, "
            "le 45Z est une constante réglementaire : les résultats sont donc affichés comme "
            "des fonctions de ces deux paramètres, et non comme des séries temporelles."
        ),
        data_warnings=[
            "L'écart de calibration du 45Z (0,46 modélisé contre ~0,49 publié) n'est pas "
            "résorbé — il est affiché en bas de page, parce que le corriger par un facteur "
            "inventé transformerait le résultat en artefact.",
            SRE_WARNING,
        ],
    ),
)

# ===========================================================================
# Paramètres
# ===========================================================================
st.sidebar.markdown("### Paramètres")
lcfs_price = st.sidebar.slider(
    "Prix du crédit LCFS (USD/t CO2e)", 0.0, 400.0, 75.0, 5.0,
    help="Publié mensuellement par CARB. Absent de l'export — c'est un axe, pas une donnée.",
)
ci_imported = st.sidebar.slider("CI de l'UCO importé (gCO2e/MJ)", 5.0, 30.0, 15.0, 1.0)
yield_lb_gal = st.sidebar.slider("Rendement (lb de feedstock par gallon)", 6.5, 9.0, 7.6, 0.1)
uco_floor = st.sidebar.slider(
    "Plancher de collecte UCO rendu USGC (c/lb)", 15.0, 60.0, 35.0, 1.0,
    help="Collecte + fret. En dessous, il n'y a pas d'offre à l'export.",
)
window_start = st.sidebar.selectbox("Fenêtre soyoil", ["2015", "2020", "2024"], index=0)

imported = Feedstock("UCO importé", ci_imported, north_american=False)
common = dict(imported=imported, yield_lb_gal=yield_lb_gal)

penalty = import_penalty(lcfs_price, **common)
bounds = penalty_bounds(**common)
neutral = lcfs_neutral_price(imported=imported)
soyoil = load_soyoil_usd_lb(window_start)
burden = discount_burden(soyoil, lcfs_usd_t=lcfs_price, **common)
exit_point = structural_exit(
    soyoil, uco_floor_usd_lb=uco_floor / CENTS_PER_USD, lcfs_usd_t=lcfs_price, **common
)

kpi_banner(
    {
        "45Z que touche le soyoil": f"{penalty.credit_45z_usd_gal:.2f} $/gal",
        "LCFS neutralisant": f"{neutral:.0f} $/t",
        "Décote requise": f"{penalty.discount_required_c_lb:.2f} c/lb",
        "Poids sur le soyoil": f"{burden.burden_last:.1%}",
        "Soyoil (dernier)": f"{soyoil.iloc[-1] * CENTS_PER_USD:.1f} c/lb",
    }
)

# ===========================================================================
# S1 — l'histoire
# ===========================================================================
section(
    "S1",
    "Des usines sitées pour l'import, puis une règle qui exclut l'import",
    "Environ trois quarts de la capacité américaine de renewable diesel a été construite "
    "sur ou près des côtes du Golfe et de Californie. Ce n'est pas un accident de "
    "géographie : ces usines ont été implantées pour tourner sur du feedstock **débarqué à "
    "quai** — UCO d'Asie, suif — et le choix du site encode ce pari sur l'origine de la "
    "matière première. Il a été fait avant que la règle fiscale n'existe, et il n'est pas "
    "réversible : on ne déplace pas une usine vers la Corn Belt.\n\n"
    "Puis deux administrations écrivent deux subventions qui ne poursuivent pas le même "
    "objectif. Le Congrès écrit 45Z pour récompenser le feedstock **nord-américain**, et "
    "exclut du crédit tout ce qui ne l'est pas. La Californie continue de payer la **faible "
    "intensité carbone** par son LCFS, sans regarder l'origine — et l'UCO importé est "
    "précisément moins carboné que le soyoil domestique. Une politique pénalise donc "
    "exactement ce que l'autre récompense, sur le même gallon, dans la même usine, le même "
    "jour. L'exploitant n'arbitre pas entre deux marchés : il arbitre entre deux "
    "administrations.",
)
scope_note(
    "Le débat public se tient en ces termes : les usines côtières soutiennent que la prime "
    "LCFS suffit à maintenir les imports, le complexe soja soutient qu'elle ne suffit pas "
    "et que le soyoil rafle la part. La page ne prend pas parti — elle montre que les deux "
    "camps argumentent sur une variable qui ne peut pas trancher."
)

# ===========================================================================
# S2 — pourquoi le calcul est difficile à contester
# ===========================================================================
section(
    "S2",
    "Ce qui s'annule — et pourquoi ce nombre est difficile à récuser",
    "Les deux filières produisent le **même gallon de renewable diesel**, dans la **même "
    "usine**, vendu au **même prix**, avec les **mêmes RIN**, les **mêmes opex** et le "
    "**même ROI**. Quand on écrit l'écart de valeur entre les deux, tous ces termes "
    "apparaissent des deux côtés et disparaissent. Il ne reste que ce qui les distingue "
    "réellement : l'écart d'intensité carbone, valorisé par le LCFS, et le crédit 45Z que "
    "l'une encaisse et l'autre pas.\n\n"
    "Ce n'est pas un raffinement de présentation, c'est ce qui rend le résultat solide. Un "
    "interlocuteur ne peut pas le récuser en contestant une prévision de diesel, un niveau "
    "de RIN ou une hypothèse d'opex — **aucune des trois n'y figure**. Il ne peut le "
    "contester que sur deux nombres, tous deux observables : l'écart de CI entre les deux "
    "filières et le prix du crédit LCFS. C'est aussi ce qui permet de se passer entièrement "
    "du prix de l'UCO, qui est justement la donnée qu'on n'a pas.",
    formula=(
        "avantage_importé - avantage_domestique\n"
        "  = [LCFS x (CI_dom - CI_imp) x EER x 134,47e-6 - crédit_45Z(dom)] / rendement\n"
        "\n"
        "  ↑ ni P_diesel, ni RIN, ni opex, ni ROI : ils se sont annulés"
    ),
)
finding(penalty.headline)

# ===========================================================================
# S3 — LE RÉSULTAT
# ===========================================================================
section(
    "S3",
    "Le LCFS ne peut pas trancher le débat — et voici de combien il s'en faut",
    "Le raisonnement des deux camps porte sur le prix du crédit LCFS : suffira-t-il, ne "
    "suffira-t-il pas. On peut répondre exactement, parce qu'il existe un prix du LCFS qui "
    f"annule précisément le 45Z à parité de prix feedstock — **{neutral:.0f} $/t CO2e**. "
    "Or le programme californien n'a jamais coté à ce niveau : son plus haut historique, "
    f"atteint en 2019-2020, tourne autour de {LCFS_PROGRAM_HIGH_USD_T:.0f} $/t, et il a "
    f"passé 2023-2024 sous {LCFS_PROGRAM_LOW_USD_T:.0f} $/t.\n\n"
    "La conséquence est plus forte qu'un simple « ça ne suffit pas ». Entre son creux et "
    "son sommet historiques — c'est-à-dire sur **toute l'amplitude que le programme a "
    f"jamais réalisée** — le LCFS ne déplace la décote requise que de "
    f"{bounds.span_c_lb:.2f} c/lb. Pendant ce temps, le soyoil lui-même a parcouru "
    f"{(soyoil.max() - soyoil.min()) * CENTS_PER_USD:.0f} c/lb sur la même période. Le "
    "levier réglementaire est donc d'un ordre de grandeur inférieur au levier du prix de "
    "la matière première : la question n'est pas mal répondue, elle est **mal posée**.",
    formula="décote_requise(LCFS) = [0,46 - LCFS x écart_CI x 134,47e-6] / rendement",
)
finding(bounds.headline)

lcfs_axis = np.arange(0.0, 401.0, 5.0)
curve = pd.DataFrame(
    {
        "decote_c_lb": [
            import_penalty(x, **common).discount_required_c_lb for x in lcfs_axis
        ],
        "plage_realisee": (lcfs_axis >= LCFS_PROGRAM_LOW_USD_T)
        & (lcfs_axis <= LCFS_PROGRAM_HIGH_USD_T),
    },
    index=pd.Index(lcfs_axis, name="LCFS (USD/t CO2e)"),
)
curve_fig = regime_chart(
    curve,
    "decote_c_lb",
    regime_col="plage_realisee",
    # Volontairement PAS la couleur verte du template : ici la bande ombrée ne veut pas dire
    # « le trade est ouvert », elle marque seulement la plage que le paramètre a réellement
    # parcourue. Réutiliser le vert coderait un jugement qui n'est pas celui du graphe.
    regime_color=ALT_COLOR,
    title="Décote que l'UCO importé doit tenir, selon le prix du crédit LCFS",
    y_title="c/lb sous le soyoil",
    reference_lines={"décote au prix retenu": penalty.discount_required_c_lb},
)
curve_fig.add_vline(
    x=neutral,
    line_dash="dot",
    line_color="crimson",
    annotation_text=f"{neutral:.0f} $/t — neutralité",
    annotation_position="top left",
)
show(curve_fig)
scope_note(
    f"Zone ombrée : la plage réellement parcourue par le crédit LCFS depuis la création du "
    f"programme ({LCFS_PROGRAM_LOW_USD_T:.0f}–{LCFS_PROGRAM_HIGH_USD_T:.0f} $/t). La courbe "
    f"ne coupe zéro qu'à {neutral:.0f} $/t, hors de cette plage — d'où le résultat. Les "
    "deux bornes sont documentées et modifiables : un lecteur disposant de la série CARB "
    "substitue les siennes sans rien changer au raisonnement."
)

# ===========================================================================
# S4 — le poids réel
# ===========================================================================
section(
    "S4",
    "La même décote ne pèse pas le même poids selon le prix de l'huile",
    "Le 45Z est écrit en **dollars par gallon**, donc la décote qu'il impose est un nombre "
    "de cents fixe. Mais un acheteur ne raisonne pas en cents absolus : il raisonne en "
    "pourcentage du prix qu'il paie. Or le soyoil a coté entre "
    f"{soyoil.min() * CENTS_PER_USD:.0f} et {soyoil.max() * CENTS_PER_USD:.0f} c/lb depuis "
    f"{window_start}. La même décote de {penalty.discount_required_c_lb:.2f} c/lb "
    f"représente donc {burden.burden_min:.1%} du prix quand l'huile est chère et "
    f"{burden.burden_max:.1%} quand elle est bon marché.\n\n"
    "Le poids de la pénalité est ainsi **contracyclique au prix de l'huile végétale** : "
    "elle mord le plus fort exactement quand les marges de trituration sont déjà minces et "
    "quand les acheteurs sont le plus sensibles au prix. C'est une propriété de la "
    "rédaction du texte — un crédit libellé en dollars par gallon plutôt qu'en pourcentage "
    "de la valeur — et non un effet de marché.",
)
finding(burden.headline)
show(
    regime_chart(
        burden.frame.assign(burden_pct=burden.frame["burden_share"] * 100.0),
        "burden_pct",
        title="Poids de la décote requise dans le prix du soyoil",
        y_title="% du prix du soyoil",
        zero_line=False,
        annotations={"2025-01-01": "45Z en vigueur"},
    )
)

# ===========================================================================
# S5 — la sortie structurelle
# ===========================================================================
section(
    "S5",
    "Le point où l'import ne se contracte plus, il s'arrête",
    "Jusqu'ici l'import est désavantagé, pas empêché : il lui suffit de consentir la "
    "décote. Mais l'UCO a un coût de collecte et un fret, donc un **prix plancher** sous "
    "lequel il n'y a simplement pas d'offre à l'export — personne ne collecte de l'huile "
    "usagée en Asie et ne l'expédie à perte. Dès que la décote exigée pousse l'UCO sous ce "
    "plancher, la filière ne se contracte pas progressivement : elle s'arrête, et aucun "
    "prix du LCFS dans sa plage réalisée ne la fait repartir.\n\n"
    "Ce plancher est le seul paramètre que la page ne peut pas observer, et c'est "
    "délibérément celui qu'on demande à l'interlocuteur : tout le reste du calcul est "
    "fermé. Ce qui suit est donc la traduction de *sa* réponse en un prix critique du "
    "soyoil, puis en une fréquence historique.",
    formula="soyoil* = plancher_collecte_UCO + décote_requise(LCFS)",
)
finding(exit_point.headline)

recent = structural_exit(
    load_soyoil_usd_lb("2024"),
    uco_floor_usd_lb=uco_floor / CENTS_PER_USD,
    lcfs_usd_t=lcfs_price,
    **common,
)
c1, c2, c3 = st.columns(3)
c1.metric("Prix critique du soyoil", f"{exit_point.soyoil_critical_usd_lb * CENTS_PER_USD:.1f} c/lb")
c2.metric(f"Sous le seuil depuis {window_start}", f"{exit_point.share_below:.0%}")
c3.metric("Sous le seuil depuis 2024", f"{recent.share_below:.0%}")

show(
    regime_chart(
        (soyoil * CENTS_PER_USD).to_frame("soyoil_c_lb").assign(
            import_impossible=soyoil < exit_point.soyoil_critical_usd_lb
        ),
        "soyoil_c_lb",
        regime_col="import_impossible",
        regime_color="rgba(248, 113, 113, 0.20)",
        title="Soyoil CBOT et prix critique en dessous duquel l'import n'est plus finançable",
        y_title="c/lb",
        zero_line=False,
        reference_lines={
            "prix critique": exit_point.soyoil_critical_usd_lb * CENTS_PER_USD
        },
        annotations={"2025-01-01": "45Z en vigueur"},
    )
)

if exit_point.share_below is not None and recent.share_below is not None:
    finding(
        f"C'est le renversement qui donne son sens à la page : sur {window_start}-2026, le "
        f"soyoil a passé {exit_point.share_below:.0%} du temps sous le prix critique ; "
        f"depuis 2024, {recent.share_below:.0%}. **Les imports fonctionnent aujourd'hui "
        "parce que l'huile est chère, pas parce que la politique est généreuse.** Un retour "
        "du soyoil vers ses niveaux de la fin des années 2010 les arrêterait, quel que soit "
        "le prix du crédit LCFS."
    )

# ===========================================================================
# S6 — la carte
# ===========================================================================
section(
    "S6",
    "Où bascule la frontière, en fonction des deux seuls paramètres qui restent",
    "Puisque tout le reste s'est annulé en S2, la réponse tient sur une carte à deux axes : "
    "l'intensité carbone attribuée à l'UCO importé, et le prix du crédit LCFS. Le trait "
    "noir est la frontière où les deux filières s'équivalent. Ce qu'il faut lire, ce n'est "
    "pas la position du point de marché — c'est la **pente** : il faut traverser presque "
    "toute la largeur de la carte en LCFS pour compenser quelques points de CI. C'est la "
    "même conclusion qu'en S3, vue autrement.",
)
grid = winner_grid(
    price_domestic_usd_lb=float(soyoil.iloc[-1]),
    price_imported_usd_lb=float(soyoil.iloc[-1]),
    yield_lb_gal=yield_lb_gal,
)
show(
    sensitivity_heatmap(
        grid,
        x_col="lcfs_usd_t",
        y_col="ci_imported",
        z_col="advantage_usd_lb",
        title="Avantage de l'UCO importé à parité de prix (bleu = l'importé gagne)",
        x_title="Prix du crédit LCFS (USD/t CO2e)",
        y_title="CI de l'UCO importé (gCO2e/MJ)",
        breakeven_note="Trait noir : frontière d'équivalence entre les deux filières",
    )
)
scope_note(
    "Carte tracée **à parité de prix feedstock** : elle isole l'effet des deux paramètres "
    "réglementaires en neutralisant le spread de prix. C'est précisément ce qui la rend "
    "lisible — et ce qui montre que le spread de prix, absent de cette carte, est le terme "
    "qui décide réellement."
)

# ===========================================================================
# S7 — la conséquence de l'autre côté du seuil
# ===========================================================================
section(
    "S7",
    "Si l'import s'arrête, la trituration domestique doit absorber le volume",
    "Le seuil de S5 n'est pas la fin de l'histoire : de l'autre côté, le volume ne "
    "disparaît pas, il doit venir du soyoil domestique. Le WASDE de juin 2026 projette la "
    "consommation d'huile de soja pour biocarburant en hausse de 14,55 à environ 17,8 "
    "milliards de livres. Cet incrément, converti au rendement de trituration de 11 livres "
    "d'huile par boisseau, donne le supplément de capacité de trituration que le complexe "
    "doit livrer — un nombre qu'un exploitant compare directement à son plan "
    "d'investissement.",
    formula="crush_requis_bu_jour = (Δhuile_lb / 11) / 365",
)
capacity = st.slider(
    "Capacité de trituration installée (M bu/jour)", 4.0, 10.0, 6.8, 0.1
)
increment_lb = st.slider(
    "Incrément de consommation d'huile (Md lb/an)", 0.5, 6.0, 3.25, 0.05,
    help="WASDE juin 2026 : 14,55 → ~17,8 Md lb.",
)
balance = crush_from_soyoil_lb(
    increment_lb * 1e9, installed_capacity_bu_day=capacity * 1e6
)
c1, c2 = st.columns(2)
c1.metric("Trituration requise par l'incrément", f"{balance.crush_required_bu_day:,.0f} bu/jour")
c2.metric(
    "En part de la capacité installée",
    f"{balance.crush_required_bu_day / balance.installed_capacity_bu_day:.1%}",
)
scope_note(
    "Lecture : l'incrément est un **supplément** à livrer, pas un besoin total — il se "
    "compare au plan d'expansion, pas à la capacité existante. La capacité installée est un "
    "slider parce qu'aucune source publique ne la donne sans ambiguïté sur le périmètre "
    "(annoncée, permitée, en service)."
)

# ===========================================================================
# Diagnostic
# ===========================================================================
calibration = calibration_gap_45z()
diagnostic_note(
    f"Contrôle de calibration du 45Z : la formule linéaire (50 − CI)/50 donne "
    f"{calibration['modelled_usd_gal']:.2f} $/gal sur un CI de "
    f"{SOYOIL_DOMESTIC.carbon_intensity:.0f}, contre {calibration['published_usd_gal']:.2f} "
    f"$/gal publié — un écart de {calibration['gap_usd_gal']:.3f} $/gal, soit "
    f"{calibration['gap_pct']:.0%}. Il tient à la définition exacte du CI retenue et **n'est "
    "pas résorbé** : le combler par un facteur d'ajustement inventé rendrait tous les "
    "chiffres de la page invérifiables. Reporté sur la décote requise, cet écart vaut "
    f"environ {calibration['gap_usd_gal'] / yield_lb_gal * CENTS_PER_USD:.2f} c/lb — à "
    "comparer aux "
    f"{bounds.span_c_lb:.2f} c/lb que le LCFS déplace sur toute son histoire."
)

mail_question(
    f"À {lcfs_price:.0f} $/t CO2e sur le crédit LCFS, je trouve que l'UCO importé doit se "
    f"vendre environ {penalty.discount_required_c_lb:.1f} c/lb sous le soyoil domestique "
    "juste pour compenser le 45Z qu'il ne peut pas réclamer — et que sur toute l'histoire "
    f"du programme LCFS, ce nombre n'a pu varier que de {bounds.span_c_lb:.1f} c/lb. Est-ce "
    "que la décote que vous voyez réellement sur l'UCO rendu USGC est de cet ordre ? Et "
    "en dessous de quel prix collecté cessez-vous simplement de charger ?",
    "Acheteurs de feedstock, exploitants de renewable diesel côtiers, origination huiles usagées",
)
