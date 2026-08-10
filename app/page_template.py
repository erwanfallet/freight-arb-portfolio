"""Page structure, modelled on the four metals pages (Cu / Al / Zn / Li).

What makes these pages hold together is not the formatting, it is the order in which
information arrives:

    1. A one-line subtitle that says exactly what the page computes.
    2. A SCOPE BOX **before any chart**: the unit trap that governs the page, what is
       out of scope and why, what is a proxy rather than a real series, the actual
       data frequency, and the conversion warnings. This is what stops a reader from
       misreading the numbers that follow.
    3. A KPI banner: the state of the world today, in five numbers.
    4. Numbered sections S2..Sn. Each: DENSE, causal prose (not bullets), the formula
       in a code font, a shaded-regime chart, and a headline number where relevant.
    5. Caveats where they apply, specific, never as a generic block at the end.

Two standing rules, carried through unchanged:

  - **An impossible print is a data diagnosis, not a market signal.** Negative
    freight, a margin that jumps by a factor of a hundred: name the cause, bound the
    window, show what the series becomes once the window is excluded.
  - **The parameter that carries the sign is exposed with its breakeven.** Not "here
    is a sensitivity" but "it takes 376 USD/t of acid credit to break even".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Regime colours, shared across every page: green = the trade is open / the economics
# work, salmon = shut / underwater.
OPEN_COLOR = "rgba(46, 160, 67, 0.18)"
SHUT_COLOR = "rgba(248, 113, 113, 0.20)"
ALT_COLOR = "rgba(147, 112, 219, 0.18)"


# ===========================================================================
# 1-2. Header and scope box
# ===========================================================================
@dataclass
class Scope:
    """The box that runs BEFORE the first chart.

    Every field exists because something was missing somewhere and its absence made a
    number hard to read correctly. `unit_trap` is the only required one: a page with no
    explicit unit trap is a page whose reader cannot verify the units.
    """

    unit_trap: str
    conversion: str = ""
    out_of_scope: list[str] = field(default_factory=list)
    proxies: list[str] = field(default_factory=list)
    frequency_note: str = ""
    data_warnings: list[str] = field(default_factory=list)


def page_header(*, code: str, title: str, subtitle: str, scope: Scope) -> None:
    """Title, one-line subtitle, then the scope box — in that order."""
    st.title(f"{code} — {title}")
    st.markdown(f"**{_escape_dollars(subtitle)}**")

    with st.container(border=True):
        st.markdown(f"**Unit trap** — {_escape_dollars(scope.unit_trap)}")
        if scope.conversion:
            st.code(scope.conversion, language="text")
        if scope.proxies:
            st.markdown("**Proxy, not a real series** — " + " · ".join(_prose(p) for p in scope.proxies))
        if scope.out_of_scope:
            st.markdown("**Out of scope** — " + " · ".join(_prose(o) for o in scope.out_of_scope))
        if scope.frequency_note:
            st.markdown(f"**Frequency** — {_prose(scope.frequency_note)}")
        if scope.data_warnings:
            st.warning(
                f"⚠ {len(scope.data_warnings)} data warning(s)\n\n"
                + "\n\n".join(f"- {_prose(w)}" for w in scope.data_warnings)
            )


def kpi_banner(metrics: dict[str, str]) -> None:
    """The state of the world today, in five numbers at most. Past that, nobody reads."""
    columns = st.columns(min(len(metrics), 6))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, str(value))


_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def _prose(text: str) -> str:
    r"""The one correction Streamlit forces on every piece of page prose: `$` escaping.

    Streamlit reads `$...$` as a math formula. A sentence with two amounts written with
    a literal dollar sign — "55 USD/t" and "21,000 USD/day" — would have everything
    between the two signs swallowed and rendered in math italics. On pages where every
    number carries a currency unit, that is systematic.

    The substitution is **idempotent**: applying it twice gives the same result as once.
    A naive `.replace("$", r"\$")` is not — it produces `\\$` on the second pass, a
    visible backslash on the page. Not theoretical: it only takes one page composing two
    template helpers to trigger the case.
    """
    return _UNESCAPED_DOLLAR.sub(r"\\$", text)


# Kept under its old name: several pages import it directly.
_escape_dollars = _prose


def snapshot_banner() -> bool:
    """Say plainly whether the page is live or reading committed results.

    Returns True when live data is present. On the deployed app there is no Bloomberg
    export — the repository is public and the data is licensed — so the pages read frames
    this codebase computed earlier. Parameters that feed the **data load** are frozen at
    their defaults; parameters applied **downstream** of it still recompute normally, and
    the distinction is stated rather than left for the reader to discover by moving a
    slider and seeing nothing happen.
    """
    from agri.data.snapshot import has_live_data

    if has_live_data():
        return True
    st.info(
        "**Published results.** This deployment reads frames computed from a licensed "
        "Bloomberg export that is deliberately not in the public repository. Sliders that "
        "feed the data load are pinned to the values these results were computed at; "
        "sliders applied downstream still work. Clone the repo, drop the export in place "
        "and every page becomes fully live."
    )
    return False


def section(number: str, title: str, prose: str, *, formula: str = "") -> None:
    """A numbered section: title, dense causal prose, formula.

    The prose is deliberately a full paragraph rather than bullets — a causal chain
    ("X pushes Y, so Z, which reinforces X instead of closing it") does not survive
    being cut into bullets without losing exactly what makes it worth reading.
    """
    st.markdown(f"## {number} — {title}")
    st.markdown(_escape_dollars(prose))
    if formula:
        st.code(formula, language="text")


def finding(text: str) -> None:
    """The headline number of a section: what a reader keeps if they read nothing else."""
    st.info(f"**{_escape_dollars(text)}**")


def diagnostic_note(text: str) -> None:
    """An impossible print, its cause, and what the series becomes once it is excluded."""
    st.warning(f"**Data diagnostic** — {_escape_dollars(text)}")


def scope_note(text: str) -> None:
    """"This section is purely informative" and why — rather than a silent omission."""
    st.caption(f"↳ {_escape_dollars(text)}")


# ===========================================================================
# Charts
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
    """A series with a shaded background on a boolean regime, plus dated annotations.

    `annotations` takes {iso_date: label}: this is where rule dates go (a VAT rebate
    removed, a Section 232 tariff step, an index's inception). A rule break annotated on
    the chart stops a regulatory jump from being read as a market move.
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
    """Stacked components plus a net line, shaded where the net goes below zero.

    This is the view that shows *which* component is carrying the margin at any given
    moment, rather than a net margin whose source is invisible.
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
    """A waterfall on one chosen date: from revenue to net, line item by line item.

    The waterfall is what makes an accounting identity *contestable*: a practitioner can
    point at the line whose level they dispute, instead of disputing an overall result.
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
    """A 2D heatmap of a margin over a grid of two parameters, with the zero contour.

    The zero contour is the product: it shows the boundary where the SIGN flips, i.e.
    which combination of assumptions is enough to reverse the conclusion.
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
    """Two series of different orders of magnitude on two axes — a price and a stock,
    a margin and an exchange rate. Use only when the two units genuinely differ: two axes
    on the same unit manufacture a false visual correlation.
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
    lag_unit: str = "months",
    n_eff: float | None = None,
) -> go.Figure:
    """Cross-correlation with its significance band drawn in.

    The band is the product, not the peak: a correlation inside the band has to read as
    such on the chart, so an insignificant peak does not get read as a result.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lags, y=values, name="CCF"))
    fig.add_hline(y=band, line_dash="dot", line_color="crimson")
    fig.add_hline(y=-band, line_dash="dot", line_color="crimson",
                  annotation_text=f"band ±{band:.3f}", annotation_position="bottom right")
    fig.add_hline(y=0, line_color="gray")
    subtitle = f"  (n_eff = {n_eff:.0f})" if n_eff else ""
    fig.update_layout(
        title=title + subtitle, xaxis_title=f"lag ({lag_unit})", yaxis_title="correlation",
        height=340, margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
    )
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# Page footer
# ===========================================================================
def mail_question(question: str, targets: str) -> None:
    """The question for the email: one only, the kind only an insider can answer."""
    st.markdown("---")
    st.markdown(f"### The question\n{_escape_dollars(question)}")
    st.caption(f"Target: {_prose(targets)}")
