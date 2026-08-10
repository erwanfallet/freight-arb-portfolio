"""Structure de page, calquee sur les quatre pages metaux (Cu / Al / Zn / Li).

Ce qui fait tenir ces pages n'est pas la mise en forme, c'est l'ordre dans lequel
l'information arrive :

    1. Un sous-titre qui dit exactement ce que la page calcule, en une ligne.
    2. Une BOITE DE PERIMETRE **avant tout graphe** : le piege d'unite qui gouverne la
       page, ce qui est hors perimetre et pourquoi, ce qui est un proxy plutot qu'une
       vraie serie, la frequence reelle des donnees, et les avertissements de conversion.
       C'est ce qui empeche un lecteur de mal lire les chiffres qui suivent.
    3. Un bandeau de KPI : l'etat du monde aujourd'hui, en cinq nombres.
    4. Des sections numerotees S2..Sn. Chacune : une prose DENSE et causale (pas des
       puces), la formule en police code, un graphe a regimes ombres, et quand il y a
       lieu un nombre de tete.
    5. Les caveats la ou ils s'appliquent, specifiques, jamais en bloc generique a la fin.

Deux regles de fond, reprises telles quelles :

  - **Un print impossible est un diagnostic de donnee, pas un signal de marche.** Un fret
    negatif, une marge qui saute d'un facteur cent : on nomme la cause, on borne la
    fenetre, on montre ce que devient la serie une fois la fenetre exclue.
  - **Le parametre qui porte le signe est expose avec son breakeven.** Pas « voici une
    sensibilite » mais « il faut 376 $/t de credit acide pour equilibrer ».
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Couleurs de regime, communes a toutes les pages : vert = le trade est ouvert /
# l'economie fonctionne, saumon = ferme / sous l'eau.
OPEN_COLOR = "rgba(46, 160, 67, 0.18)"
SHUT_COLOR = "rgba(248, 113, 113, 0.20)"
ALT_COLOR = "rgba(147, 112, 219, 0.18)"


# ===========================================================================
# 1-2. En-tete et boite de perimetre
# ===========================================================================
@dataclass
class Scope:
    """La boite qui passe AVANT le premier graphe.

    Chaque champ existe parce qu'il manquait quelque part et que son absence rendait un
    chiffre mal lisible. `unit_trap` est le seul obligatoire : une page sans piege
    d'unite explicite est une page dont le lecteur ne peut pas verifier les unites.
    """

    unit_trap: str
    conversion: str = ""
    out_of_scope: list[str] = field(default_factory=list)
    proxies: list[str] = field(default_factory=list)
    frequency_note: str = ""
    data_warnings: list[str] = field(default_factory=list)


def page_header(*, code: str, title: str, subtitle: str, scope: Scope) -> None:
    """Titre, sous-titre d'une ligne, puis la boite de perimetre — dans cet ordre."""
    st.title(f"{code} — {title}")
    st.markdown(f"**{_escape_dollars(subtitle)}**")

    with st.container(border=True):
        st.markdown(f"**Piège d'unité** — {_escape_dollars(scope.unit_trap)}")
        if scope.conversion:
            st.code(scope.conversion, language="text")
        if scope.proxies:
            st.markdown("**Proxy, pas une vraie série** — " + " · ".join(_prose(p) for p in scope.proxies))
        if scope.out_of_scope:
            st.markdown("**Hors périmètre** — " + " · ".join(_prose(o) for o in scope.out_of_scope))
        if scope.frequency_note:
            st.markdown(f"**Fréquence** — {_prose(scope.frequency_note)}")
        if scope.data_warnings:
            st.warning(
                f"⚠ {len(scope.data_warnings)} avertissement(s) de données\n\n"
                + "\n\n".join(f"- {_prose(w)}" for w in scope.data_warnings)
            )


def kpi_banner(metrics: dict[str, str]) -> None:
    """L'etat du monde aujourd'hui, en cinq nombres maximum. Au-dela on ne lit plus."""
    columns = st.columns(min(len(metrics), 6))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, _localise_numbers(str(value)))


_INLINE_CODE = re.compile(r"`[^`]*`")
_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
THIN_NBSP = " "


def _localise_numbers(text: str) -> str:
    """Convertit les nombres de la convention anglo-saxonne vers la francaise.

    Python formate en `1,843.5`. Sur une page redigee en francais, un lecteur lit « 1,843 »
    comme mille huit cent quarante-trois **virgule** cinq : un facteur mille d'erreur sur un
    prix. Ce n'est pas une question de style, c'est une erreur de lecture — et sur un
    portefeuille dont tout l'argument tient a des unites correctement traitees, la laisser
    passer serait cocasse.

    Applique a la prose seulement, jamais aux blocs `st.code` : les formules et les
    identites restent en convention code, ou le point decimal est la norme. Les portions
    entre accents graves sont protegees pour la meme raison — `estimate_half_life(x, 0.7)`
    est du code, pas du texte.
    """
    protected: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    text = _INLINE_CODE.sub(_stash, text)
    text = _THOUSANDS_COMMA.sub(THIN_NBSP, text)
    text = _DECIMAL_POINT.sub(",", text)
    for index, original in enumerate(protected):
        text = text.replace(f"\x00{index}\x00", original)
    return text


def _prose(text: str) -> str:
    r"""Le traitement applique a tout texte de prose de la page.

    Deux corrections que Streamlit et Python imposent, reunies en un seul endroit pour
    qu'aucune page n'ait a y penser :

    1. **Les `$` sont echappes.** Streamlit interprete `$...$` comme une formule
       mathematique : une phrase contenant deux montants — « 55 $/t » et « 21 000 $/jour » —
       voit tout ce qui est entre les deux dollars avale et rendu en italique. Sur des pages
       ou chaque chiffre porte une unite monetaire, c'est systematique.
    2. **Les nombres sont localises**, cf. `_localise_numbers`.

    Le traitement est **idempotent** : l'appliquer deux fois donne le meme resultat. La
    version naive (`.replace("$", r"\$")`) ne l'etait pas — elle produisait `\\$` au second
    passage, donc un antislash visible dans la page. Ce n'est pas theorique : il suffit
    qu'une page compose deux helpers du template pour declencher le cas.
    """
    return _UNESCAPED_DOLLAR.sub(r"\\$", _localise_numbers(text))


# Conserve sous son ancien nom : plusieurs pages l'importent directement.
_escape_dollars = _prose


def section(number: str, title: str, prose: str, *, formula: str = "") -> None:
    """Une section numerotee : titre, prose dense et causale, formule.

    La prose est volontairement en paragraphe plein et non en puces — une chaine causale
    (« X pousse Y, donc Z, ce qui renforce X au lieu de le fermer ») ne se decoupe pas en
    puces sans perdre justement ce qui la rend interessante.
    """
    st.markdown(f"## {number} — {title}")
    st.markdown(_escape_dollars(prose))
    if formula:
        st.code(formula, language="text")


def finding(text: str) -> None:
    """Le nombre de tete d'une section : ce qu'on retient si on ne lit rien d'autre."""
    st.info(f"**{_escape_dollars(text)}**")


def diagnostic_note(text: str) -> None:
    """Un print impossible, sa cause, et ce que devient la serie une fois la cause ecartee."""
    st.warning(f"**Diagnostic de donnée** — {_escape_dollars(text)}")


def scope_note(text: str) -> None:
    """« Cette section est purement informative » et pourquoi — plutot qu'une omission."""
    st.caption(f"↳ {_escape_dollars(text)}")


# ===========================================================================
# Graphes
# ===========================================================================
def regime_chart(
    frame: pd.DataFrame,
    value_col: str,
    *,
    regime_col: str | None = None,
    regime_color: str = OPEN_COLOR,
    title: str = "",
    y_title: str = "",
    zero_line: bool = True,
    reference_lines: dict[str, float] | None = None,
    annotations: dict[str, str] | None = None,
) -> go.Figure:
    """Serie avec fond ombre sur un regime booleen, et annotations datees.

    `annotations` prend {date_iso: libelle} : c'est la ou vont les dates de regle
    (elimination du rebate TVA, palier de tarif Section 232, inception d'un indice).
    Une rupture de regle annotee sur le graphe evite de lire un saut reglementaire
    comme un mouvement de marche.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame[value_col], name=value_col, mode="lines")
    )

    if regime_col and regime_col in frame.columns:
        active = frame[regime_col].astype(bool)
        if active.any():
            block = (active != active.shift()).cumsum()
            for _, chunk in frame[active].groupby(block[active]):
                fig.add_vrect(
                    x0=chunk.index.min(), x1=chunk.index.max(),
                    fillcolor=regime_color, opacity=1.0, line_width=0, layer="below",
                )

    for label, level in (reference_lines or {}).items():
        fig.add_hline(
            y=level, line_dash="dot", line_color="gray",
            annotation_text=label, annotation_position="right",
        )
    if zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

    for date_iso, label in (annotations or {}).items():
        fig.add_vline(
            x=pd.Timestamp(date_iso), line_dash="dot", line_color="crimson",
            annotation_text=label, annotation_position="top",
        )

    fig.update_layout(
        title=title, yaxis_title=y_title, height=420,
        margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
    )
    return fig


def stacked_components_chart(
    components: pd.DataFrame,
    *,
    net_col: str,
    title: str = "",
    y_title: str = "",
    underwater_color: str = SHUT_COLOR,
) -> go.Figure:
    """Composantes empilees + ligne nette, ombree la ou le net passe sous zero.

    C'est la lecture qui montre *quelle* composante porte la marge a chaque instant,
    plutot qu'une marge nette dont on ne sait pas d'ou elle vient.
    """
    fig = go.Figure()
    for column in components.columns:
        if column == net_col:
            continue
        fig.add_trace(
            go.Bar(x=components.index, y=components[column], name=column)
        )
    fig.add_trace(
        go.Scatter(
            x=components.index, y=components[net_col], name=net_col,
            mode="lines", line=dict(color="black", width=2),
        )
    )

    underwater = components[net_col] < 0
    if underwater.any():
        block = (underwater != underwater.shift()).cumsum()
        for _, chunk in components[underwater].groupby(block[underwater]):
            fig.add_vrect(
                x0=chunk.index.min(), x1=chunk.index.max(),
                fillcolor=underwater_color, opacity=1.0, line_width=0, layer="below",
            )

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        barmode="relative", title=title, yaxis_title=y_title, height=440,
        margin=dict(t=50, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def waterfall_chart(
    components: dict[str, float],
    *,
    total_label: str = "net",
    title: str = "",
    y_title: str = "",
) -> go.Figure:
    """Waterfall sur une date choisie : de la recette au net, poste par poste.

    Le waterfall est ce qui rend une identite comptable *contestable* : un praticien peut
    pointer la ligne dont il conteste le niveau, au lieu de contester un resultat global.
    """
    labels = list(components) + [total_label]
    values = list(components.values()) + [sum(components.values())]
    measures = ["relative"] * len(components) + ["total"]

    fig = go.Figure(
        go.Waterfall(
            orientation="v", measure=measures, x=labels, y=values,
            text=[f"{v:,.1f}" for v in values], textposition="outside",
            connector=dict(line=dict(color="gray")),
        )
    )
    fig.update_layout(
        title=title, yaxis_title=y_title, height=420,
        margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
    )
    return fig


def sensitivity_heatmap(
    grid: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    title: str = "",
    x_title: str = "",
    y_title: str = "",
    breakeven_note: str = "",
) -> go.Figure:
    """Heatmap 2D d'une marge sur une grille de deux parametres, avec le contour zero.

    Le contour zero est le produit : il montre la frontiere ou le SIGNE bascule, donc
    quelle combinaison d'hypotheses suffit a retourner la conclusion.
    """
    pivot = grid.pivot(index=y_col, columns=x_col, values=z_col)
    fig = go.Figure(
        go.Heatmap(
            x=pivot.columns, y=pivot.index, z=pivot.values,
            colorscale="RdBu", zmid=0, colorbar=dict(title=z_col),
        )
    )
    fig.add_trace(
        go.Contour(
            x=pivot.columns, y=pivot.index, z=pivot.values,
            contours=dict(start=0, end=0, size=1, coloring="none",
                          showlabels=True, labelfont=dict(color="black")),
            line=dict(color="black", width=2), showscale=False, name="breakeven",
        )
    )
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title, height=460,
        margin=dict(t=50, b=20, l=10, r=10),
    )
    if breakeven_note:
        fig.add_annotation(
            text=breakeven_note, xref="paper", yref="paper", x=0.02, y=1.06,
            showarrow=False, font=dict(size=12),
        )
    return fig


def dual_axis_chart(
    frame: pd.DataFrame,
    left_col: str,
    right_col: str,
    *,
    title: str = "",
    left_title: str = "",
    right_title: str = "",
) -> go.Figure:
    """Deux series d'ordres de grandeur differents sur deux axes — un prix et un stock,
    une marge et un taux de change. A n'utiliser que quand les deux unites different
    reellement : deux axes sur la meme unite fabriquent une correlation visuelle fausse.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame[left_col], name=left_col, mode="lines"))
    fig.add_trace(
        go.Scatter(x=frame.index, y=frame[right_col], name=right_col, mode="lines", yaxis="y2")
    )
    fig.update_layout(
        title=title, height=420, margin=dict(t=50, b=20, l=10, r=10),
        yaxis=dict(title=left_title),
        yaxis2=dict(title=right_title, overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def ccf_panel(
    lags: np.ndarray,
    values: np.ndarray,
    band: float,
    *,
    title: str = "",
    lag_unit: str = "mois",
    n_eff: float | None = None,
) -> go.Figure:
    """Cross-correlation avec sa bande de significativite tracee.

    La bande est le produit, pas le pic : une correlation dans la bande doit se voir
    comme telle sur le graphe, pour qu'un pic non significatif ne se lise pas comme un
    resultat.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=values, name="CCF"))
    fig.add_hline(y=band, line_dash="dot", line_color="crimson")
    fig.add_hline(y=-band, line_dash="dot", line_color="crimson",
                  annotation_text=f"bande ±{band:.3f}", annotation_position="bottom right")
    fig.add_hline(y=0, line_color="gray")
    subtitle = f"  (n_eff = {n_eff:.0f})" if n_eff else ""
    fig.update_layout(
        title=title + subtitle, xaxis_title=f"lag ({lag_unit})", yaxis_title="corrélation",
        height=340, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
    )
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# Bas de page
# ===========================================================================
def mail_question(question: str, targets: str) -> None:
    """La question du mail : une seule, a laquelle seul un insider peut repondre."""
    st.markdown("---")
    st.markdown(f"### La question\n{_escape_dollars(question)}")
    st.caption(f"Cible : {_prose(targets)}")
