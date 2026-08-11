# Method notebook

This file is not the showcase. The showcase is `README.md` and the dashboard. This is
where we keep the rules we hold ourselves to and the history of decisions, including the
bad ones.

## Rules we don't relitigate

1. **We don't pick the thesis before running the test.** Project A has a decisive test —
   how much of the 65-62 premium freight explains — and that test runs before the six
   sections get built. If the freight share is small and stable, there is no story, and
   we say so.
2. **We don't rename a residual.** A residual contains everything we didn't model. Naming
   it honestly is more credible than calling it "physical tension."
3. **We don't model what we have no data for.** No steel value-in-use model. The term is
   set to zero, the residual is said to absorb it, and the reader is asked what they think
   — that turns it into a question instead of a fault.
4. **We declare the resolution limit before running the test.** Customs data is monthly;
   if the S4 feedback loop doesn't reach a conclusion, that's a stated granularity limit,
   not a failure discovered after the fact.
5. **We never publish a number from synthetic mode.** Ever.
6. **We prefer a bias whose sign we know.** A-H1 underestimates the freight share; since
   the thesis claims that share is large, the bias works against us. That's the right
   direction, and it should be stated in the email.

## Order of attack

1. **Check M65F liquidity** before anything else. If the 65% Fe contract is too thinly
   traded, the computed premium is an artefact and the fallback is a 62-58 premium — less
   elegant but liquid.
2. **The S3 decisive test**: share of the premium's variance explained by C3 − C5, in
   level and in variation. If the gap between the two readings is large, variation wins.
3. Then the six sections, then GACC flow validation, then the email.

## Log

- **2026-08-07** — Moved from "three projects" to "portfolio": `src/freight/portfolio.py`
  becomes the single source of truth (one `Project` per chain, `status` ready/planned),
  and `app/Home.py` is rewritten to read it instead of hard-coded three-column content —
  it no longer needs to be touched when a chain is added. Two `STATUS_PLANNED` entries
  added (agriculture, LNG) as a reminder of the next scoping work, not as code.
  `docs/NOUVELLE_CHAINE.md` written: the seven files a new chain always touches, in order,
  with the reminder that the real subject of each project is a quoting-unit trap, not just
  "another arb." Verified under real conditions: app launched locally, the three
  dashboards and the two "to build" sections display, 90 tests still green.
- **2026-08-06 (3)** — Project C implemented: `chains/products.py` (volume/mass
  conversion, exact decomposition of the freight variation into market / reset / cross
  shares, open-days illusion, seasonal profile, inversion of the TCE engine for the C-2
  variant), 20 golden tests, 6-section dashboard, synthetic pipeline. `signals/worldscale.py`
  is reused as-is: it already refused to convert points into $/t without a dated flat
  rate, which makes it the best-aged module in the repo.
  The portfolio now has its signature: **all three projects turn on a quoting unit that
  is not the economic unit** — dry tonne vs wet tonne, kcal vs tonne, gallon vs tonne.
  That's not a coincidence, it's where the errors nobody corrects hide.
  `DEMANDE_DONNEES.md` written: priority-ordered list, files to produce, entitlements to
  test, a ready-to-send message.
- **2026-08-06 (2)** — Project B implemented: `chains/coal.py` (ARA arb, ETS layer with
  phase-in and FX conversion, CV energy basis, controlled MCO, break test and regime
  statistics), 22 golden tests, 6-section dashboard, synthetic pipeline. B's synthetic
  generator **imposes** the 2022 break: it's stated at the top of the module, and the page
  shows a starker warning than project A's.
  A method result obtained along the way: on the synthetic set, whose true post-break
  slope is 0.15, the regression without control gives 0.71 and with the TTF control 0.18.
  Without the control, the conclusion would be the exact opposite of the truth. That's the
  justification for the omitted-variable-bias test added afterward.
- **2026-08-06** — Project A defined and implemented (engine + 16 golden tests + 6-section
  dashboard + end-to-end synthetic pipeline). `data_dictionary.csv` written for the three
  chains with `exchange_code`, `bbg_ticker` and `verified`. No Bloomberg ticker written in:
  none has been seen yet.
