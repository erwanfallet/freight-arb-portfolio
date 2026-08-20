# Physical arbitrage — freight as the decisive term

A portfolio, not eighteen isolated projects: every chain added follows the same template
(`docs/NOUVELLE_CHAINE.md`), rebuilds its margin from raw series, and confronts it with
an **official, free physical-flow series**. The question is never just "is the arb
open," it's **"did the cargo actually leave."**

**Freight — dry bulk, tankers and bunkers, freight as the deciding term**

| | Project | Sector | The question it settles |
|---|---|---|---|
| **A** | Iron ore 65–62 premium | Dry bulk — ores | The high-grade premium does contain a Capesize freight spread — at about half the weight the raw decomposition suggests |
| **B** | Coal-to-gas switching ceiling | Power & gas | Distance from the switching level predicts TTF's 20-day forward return; a trailing-median placebo does not survive the horse race |
| **C** | Transatlantic distillate | Tankers — refined products | Volume is not mass, and Worldscale points are not a cost |
| **D** | Grain seasonality in freight | Dry bulk — agriculture | The Brazilian harvest, the largest cargo flow in the market, leaves no mark on 27 years of the Baltic Panamax Index |
| **E** | Freight incidence in the ore price | Dry bulk — ores | The CFR buyer does not absorb the freight — a null backed by a power benchmark, not a weak test |
| **F** | Bunker basis | Bunkers — fuel risk | The crude hedge ratio for VLSFO has fallen threefold since 2016, and it is drift rather than a break |
| **G** | The marginal ship | Dry bulk — chartering | How much less efficient a ship can be and still cover its fuel bill on P8, solved in closed form from the voyage engine |
| **H** | Index-vs-route basis | Freight derivatives | The BPI tracks P8 *better* in the tails, yet the unhedged residual still scales with the size of the move |
| **I** | CII and the ballast leg | Shipping regulation | The rating rewards slow-steaming an empty leg, with no cargo consequence at all |

**Agri, softs and biofuels — the same method applied to the crush, the refinery and the subsidy**

| | Project | Sector | The question it settles |
|---|---|---|---|
| **T1-1** | Freight inside the C&F | Grains & oilseeds | What the freight term is actually doing inside a C&F price nobody decomposes |
| **T1-2** | The full cost of hedging | Softs — cocoa & coffee | The hedge costs more than the spread: financing plus collateral, not just the basis |
| **T2-3** | Board crush | Oilseeds | The board crush is a yield in disguise, not a price |
| **T2-4** | White premium | Sugar | What a price can and cannot tell you about a refiner's rent |
| **T2-5** | The plant as an option | Processing | Hysteresis: restart and shutdown are not the same threshold, and the gap is the option |
| **T2-6** | Palm–soy substitution | Vegetable oils | The substitution bound everyone quotes does not exist in the data |
| **T3-1** | LCFS against 45Z | Biofuels feedstock | Two subsidies that point the feedstock choice in opposite directions |
| **T3-2** | The Brazilian cost floor | Sugar & ethanol | The "cost floor" is an exchange rate wearing a cost's clothes |
| **T3-4** | China soy origination | Oilseeds — origination | The windows in which no origin works |

All eighteen are `STATUS_READY`. This table is a human-readable summary — the source of truth is
**`src/freight/portfolio.py`**, which `app/Home.py` reads. Adding a project there is
enough to make it appear on the platform; see **`docs/NOUVELLE_CHAINE.md`** for the exact
template of the seven files a new chain touches.

**`DEMANDE_DONNEES.md`** is the document to have open in front of the terminal: the list
of series by project and priority, the exact files to produce, the entitlements to test,
and a ready-to-send message. `FICHE_DONNEES.md` is its longer, reasoned version.

---

## Project A — the thesis in five lines

The 62% Fe and 65% Fe iron ore indices are both quoted **CFR China**. The 65% grade is
mostly Brazilian (Tubarão → Qingdao, ~11,000 nm, route C3), the 62% grade mostly
Australian (Port Hedland → Qingdao, ~1,600 nm, route C5). Freight is therefore already
inside both prices, and the high-grade premium mechanically contains the C3 − C5
differential. Decompose it, and look at what's left once the distance is paid for.

```
observed_premium = P65_CFR − P62_CFR
fair_value_freight = C3/(1 − h_BR) − C5/(1 − h_AU)
residual            = observed_premium − fair_value_freight
```

The residual is called a residual. It contains quality, use-value, physical tension
**and** the FFA-vs-index basis. No public data lets these four terms be separated, so it
doesn't get rebranded "physical tension" to make the conclusion easier to sell.

### The unit trap, which is the technical core

Iron ore indices are quoted in USD per **dry** metric tonne (dmt). Freight is paid on the
**loaded** weight, i.e. wet (wmt, bill-of-lading weight):

```
freight_per_dmt = freight_per_wmt / (1 − moisture)
```

Brazilian fines ~9% moisture, Pilbara Blend ~8%. On the golden test's worked example —
P65 120, P62 100, C3 20, C5 10 — the freight share of the premium moves from **50.0%**
uncorrected to **55.5%** corrected. Ignoring the conversion systematically understates
freight.

It's the same move as `TC_per_t_zinc = TC_per_dmt_conc / (grade × recovery)`: the quoted
unit is not the economic unit, and doing the conversion properly is half the job.

---

## Project B — the thesis in five lines

Every European power desk knows the shape of the idea: push gas expensive enough
relative to coal-plus-carbon and generation switches fuel, capping TTF's further upside.
That belief is old and it is not the pitch. The pitch is that the level at which it's
supposed to bind is never published — it's a property of two plant efficiencies no
exchange quotes — and the belief has an honest, testable alternative: TTF might simply
mean-revert on its own, with "switching" attached to the chart after the fact.

```
ttf_switch = (eta_gas/eta_coal) x coal_th + eua x (eta_gas x EF_coal/eta_coal - EF_gas)
```

### The test, not the idea, is the finding

Regressing TTF's 20-day forward return on distance from `ttf_switch`, on
**non-overlapping** windows (daily overlap would reuse the same 20-day outcome twenty
times and manufacture the t-stat), against a placebo — distance from TTF's own 250-day
trailing median, which contains no switching economics at all:

| regressor | coefficient | t-stat | reading |
|---|---|---|---|
| switching distance | negative | significant | survives the honest sample |
| placebo (trailing median) | ~0 | not significant | does not survive alone |
| both, horse race | switching keeps sign | placebo does not | one mechanism, not two |

n = 98 non-overlapping windows out of 1,945 overlapping daily rows since 2018 — a small
sample on purpose, and the page refuses a verdict below `MIN_OBS_FOR_VERDICT = 60`.

### What this replaced

The module was originally built around the Richards Bay → ARA arb
(`API2 CIF ARA − API4 FOB Richards Bay − freight − ETS`), on the thesis that Indian
demand pulled the marginal RB cargo off the European netback after 2022. **API4 Richards
Bay is absent from the export**, so that spread was never computable here and the thesis
was abandoned before publication rather than faked with a proxy. API2, TTF, EUA and
EURUSD remain and support the switching-ceiling test instead.

---

## Project C — the thesis in five lines

```
arb = P_ARA($/t) − P_USGC($/t) − freight($/t) − spec_bridge − financing − losses
      with  P_USGC($/t) = P_USGC($/gal) × 42 × bbl_per_tonne
      and   freight($/t) = WS/100 × flat_rate(route, year)
```

Since 2022 Europe has lost Russian diesel: the transatlantic flow reversed and lengthened
in tonne-miles. Two terms in this line aren't what they look like.

### Volume is not mass

The US leg is quoted in $/gallon, the European leg in $/tonne. The conversion runs
through a density, ~7.45 bbl/t. On a leg at ~780 $/t, moving from 7.45 to 7.50 shifts the
price by **5.25 $/t** — often more than the whole arb. **The conversion factor everyone
treats as a constant weighs as much as the signal.**

### Worldscale points are not a cost

The flat rate resets every January 1st. The decomposition is an exact identity:

```
Δfreight = [ ΔWS·FR_prev  +  WS_prev·ΔFR  +  ΔWS·ΔFR ] / 100
             └ market ┘      └ reset ┘       └ cross ┘
```

At constant WS points, WS 150 and a flat rate moving from 20 to 24 give **+6.00 $/t** of
cost, none of it market. On the synthetic set, the largest jump is worth **11.78 $/t**,
of which **11.96 is reset** and **−0.13 is market**, against an arb whose mean is
−1.02 $/t: the reset weighs ten times the signal, and it's invisible to anyone just
watching the points.

`signals/worldscale.py` refuses by construction to convert points into $/t without a
dated flat rate — it raises rather than falling back on the previous year.

### If flat rates are missing, the project still holds

- **C-3**: both price legs stay free and exchange-traded, the arb can still be built.
- **C-2**: invert the TCE engine to reconstruct freight from voyage economics — distances,
  cubic consumption, bunkers, port days. `voyage/tce.py` finally stops being dead
  infrastructure. A test verifies that the freight → TCE → freight loop closes exactly.

---

## Structure

```
src/freight/
  portfolio.py           portfolio registry — the only source app/Home.py reads;
                        one more Project here is enough to make it appear on the platform
  chains/ironore.py     project A: moisture, decomposition, variance explained,
                        negative-residual episodes, hedging effect, carry
  chains/coal.py        project B: switching level (coal/gas/carbon reconciled to
                        EUR/MWh electricity), non-overlapping ceiling test vs. a
                        trailing-median placebo
  chains/products.py    project C: volume/mass conversion, Worldscale decomposition,
                        open-days illusion, TCE inversion
  chains/grain_seasonal.py, freight_incidence.py, bunker_basis.py,
  chains/marginal_ship.py, index_basis.py, cii_ballast.py
                        projects D-I: the six freight-native chains
src/agri/
  portfolio.py           the canonical Project type and the nine agri entries;
                        src/freight/portfolio.py re-exports it and prepends A-I
  chains/                the nine agri, softs and biofuels chains (T1-1 ... T3-4)
  core/                  the shared toolkit: voyage economics, regression, seasonality
  data/snapshot.py       @cached: live Bloomberg export when present locally,
                        pre-computed parquet snapshot on the public deployment
  ingest/contract.py    data contract — nothing goes in without a filled-in contract
  ingest/loader.py      raw exports -> canonical long format (date, ticker, value)
  ingest/series.py      long format -> series, + coverage table
  ingest/fixture.py     SYNTHETIC GENERATOR for project A, tickers prefixed SYNTH_
  ingest/fixture_products.py  same for project C — the flat-rate jumps are IMPOSED
  ingest/audit.py       coverage audit over data/raw/
  voyage/               TCE, consumption, distances, owner's indifference C3*/C5*
  backtest/             execution engine, attribution, parameter sensitivity
  signals/worldscale.py WS-points -> USD/t conversion (waiting on flat rates)
  events/               event post-mortem template
app/Home.py             portfolio home page, grouped by sector, generated
                        from src/freight/portfolio.py — never hand-edited
app/pages/               one Streamlit dashboard per ready chain
docs/NOUVELLE_CHAINE.md  template for the seven files a new chain touches
tests/                  golden tests, hand-computed values, no data required
scripts/smoke_*.py    end-to-end pipeline on synthetic data
data/raw/               raw exports, immutable, never hand-edited
```

## Running it

```bash
make install   # venv + editable install, engine + dashboard
make test      # golden tests — no data needed, safe to run any time
make smoke     # full pipeline for both projects on SYNTHETIC datasets
make app       # Streamlit dashboard
make audit     # once data/raw/ is filled and data_dictionary.csv is complete
```

## Method rules

- **A data gap is information**, not noise to smooth over. No forward-fill before the
  audit step. `gap_policy` accepts only `none` — the refusal is coded, not just written
  down.
- **No series enters a calculation without a filled-in contract**: ticker, native unit,
  frequency, source, plus a dated `verified` flag. The `exchange_code` and `bbg_ticker`
  columns are deliberately separate: the exchange's contract code is publicly verifiable,
  the Bloomberg ticker has to be confirmed on the terminal via `CTM` then `DES`. **No
  Bloomberg ticker gets written into this repo without having been seen.**
- **Calendar alignment is explicit.** The decomposition works on the intersection of the
  four series and shows how many dates survived.
- **Synthetic mode is flagged in red** in the dashboard, and its tickers are prefixed
  `SYNTH_`. No number produced in this mode should leave the repo.

## Test status

**692 tests, all green, with no market data at all.** Every expected value is hand-computed
in the comment preceding it. Four tests carry their own argument:

- `test_omitting_the_control_biases_the_freight_coefficient` — why `ols()` taking
  multiple regressors matters at all; the same reasoning is what makes project B's
  switching-vs-placebo horse race meaningful rather than decorative
- `test_ceiling_test_recovers_a_designed_reversion_and_rejects_a_flat_placebo` —
  project B's switching-ceiling mechanism, on data where the reversion is imposed by hand
- `test_reset_moves_cost_with_zero_market_move` — project C's Worldscale result, in its
  barest form
- `test_implied_freight_round_trips_through_the_tce_engine` — if this loop doesn't close,
  the whole C-2 variant collapses
