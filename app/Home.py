"""Portfolio landing page. `streamlit run app/Home.py`

Nothing is hardcoded here: everything comes from `freight.portfolio`. Adding a project to
the portfolio means adding a `Project` to `src/freight/portfolio.py` — this page never has
to be edited by hand.
"""
from __future__ import annotations

import streamlit as st

from freight.portfolio import (
    DATA_REAL,
    DATA_SYNTHETIC,
    PROJECTS,
    STATUS_READY,
    by_tier,
    total_tests,
)

st.set_page_config(page_title="Physical arbitrage portfolio", layout="wide")

st.title("Physical arbitrage — when the quoted unit is not the economic unit")

st.markdown(
    """
Twelve projects across dry bulk, refined products, gas, softs and grains. One throughline,
and it is not the commodity: **every project turns on the quoted unit not being the unit
the economics run on.** Wet tonne against dry tonne, kcal against tonne, gallon against
tonne, cents per bushel against dollars per bushel, USD per day against USD per tonne,
ringgit against dollar.

That is not a presentational detail. In several of these projects the conversion factor
moves the answer by more than the phenomenon being measured — which means getting it wrong
does not produce an error, it produces a plausible number that is wrong.
"""
)

n_ready = sum(1 for p in PROJECTS if p.status == STATUS_READY)
n_real = sum(1 for p in PROJECTS if p.data_mode != DATA_SYNTHETIC)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Projects", len(PROJECTS))
c2.metric("Built and tested", n_ready)
c3.metric("On real market data", n_real)
c4.metric("Golden tests", total_tests())

st.divider()

_DATA_BADGE = {
    DATA_REAL: "🟩 real data",
    DATA_SYNTHETIC: "⬜ synthetic — no number here is a market result",
}

for tier, projects in by_tier().items():
    st.subheader(tier)
    for row_start in range(0, len(projects), 3):
        row = projects[row_start : row_start + 3]
        columns = st.columns(len(row))
        for column, project in zip(columns, row):
            with column:
                badge = _DATA_BADGE.get(project.data_mode, "🟨 hybrid — main legs real")
                st.markdown(f"**{project.code} — {project.title}**")
                st.caption(badge)
                st.markdown(f"*{project.thesis}*")
                with st.expander("Where the disagreement comes from"):
                    st.markdown(project.disagreement)
                    st.markdown(f"**The deliverable** — {project.pivot}")
                    if project.data_fallback:
                        st.caption(f"Data constraint: {project.data_fallback}")
                with st.expander("The question in the email"):
                    st.markdown(project.mail_question)
                    st.caption(f"Target: {project.targets}")
                if project.dashboard_page:
                    st.page_link(project.dashboard_page, label="Open the dashboard →")
                st.caption(f"{project.n_tests or 0} golden tests")
    st.divider()

st.markdown(
    """
### Method rules, portfolio-wide

- **A data gap is information**, not noise to smooth over. No forward-fill before the audit
  step — `gap_policy` accepts only `none`, and the refusal is in the code.
- **No series enters a calculation without a filled-in contract**: ticker, native unit,
  frequency, source, and a dated `verified` flag.
- **An impossible print is a data diagnosis, not a market signal.** A negative freight rate,
  a margin that jumps by a factor of a hundred: name the cause, bound the window, show what
  the series becomes once the window is excluded.
- **The parameter that carries the sign is exposed with its breakeven.** Not "here is a
  sensitivity" but "it takes 285 USD per tonne of CO2e to offset the credit".
- **A residual is called a residual.** It does not get renamed "physical tension" to make
  the conclusion easier to sell.
- **Synthetic mode is flagged**, never confused with a real result. No number produced on it
  should leave the repository.

Several projects end in a **negative result** — the substitution band that does not exist,
the rent level that is not identifiable from prices. Those are kept as they are. A portfolio
in which every thesis is confirmed is a portfolio that was not really tested.
"""
)
