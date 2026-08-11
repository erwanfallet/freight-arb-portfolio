# Physical arbitrage — freight as the decisive term

A portfolio, not three isolated projects: every chain added follows the same template
(`docs/NOUVELLE_CHAINE.md`), rebuilds its margin from raw series, and confronts it with
an **official, free physical-flow series**. The question is never just "is the arb
open," it's **"did the cargo actually leave."**

| | Project | Sector | Thesis | Status |
|---|---|---|---|---|
| **A** | Iron ore | Dry bulk — ores | The 65-62% Fe premium is partly a disguised Capesize freight spread | Engine + dashboard ready, waiting on the 4 series |
| **B** | Atlantic coal | Dry bulk — energy | The API2 − API4 arb lost its binding constraint in 2022 | Engine + dashboard ready, waiting on series |
| **C** | Transatlantic distillate | Tankers — refined products | Volume is not mass, and Worldscale points are not a cost | Engine + dashboard ready, C-2/C-3 variants coded |
| **D** | TBD | Dry bulk — agriculture | — | Sector identified, nothing coded |
| **E** | TBD | Gas (LNG) | — | Sector identified, nothing coded |

This table is a human-readable summary — the source of truth is
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

```
arb_ARA = API2 (CIF ARA) − API4 (FOB Richards Bay) − freight(C4) − financing − ETS
```

The textbook says this arb can't stay wide open: competition and freight pull it back
toward zero. Since 2022, South African coal has been going to India rather than Europe,
so the marginal Richards Bay cargo is no longer priced off Rotterdam. The equation lost
its binding term. Since the CFR India price is licensed data, price equality isn't
proven directly: the reorientation is shown in **flow**, and presented as the weaker
result it is.

### The control that decides everything

2022 is also the year of the European gas shock. Attributing the break to India without
controlling for TTF is getting the mechanism wrong. On the synthetic set — whose true
post-break slope is **0.15** by construction — the regression gives:

| | coefficient on freight, post-break |
|---|---|
| without control | **0.71** — the conclusion would be that freight still binds |
| with TTF control | **0.18** — close to the truth, freight no longer binds |

The control isn't an econometrician's refinement: without it, the conclusion is
reversed.

### The two technical layers

**Calorific value.** API2 and API4 are both 6,000 kcal/kg NAR reference grades: the
reference arb is CV-neutral **by construction** and stays correct. The problem is it has
stopped describing the physical cargo, whose real CV has drifted to ~5,700-5,800. Freight
is paid per tonne, coal is sold per kcal: at 5,750 kcal, freight per
tonne-equivalent-6,000 is worth 1.0435× the quoted freight.

**Maritime ETS.** Since 2024, a voyage with one end outside the EU — Richards Bay →
Rotterdam — is covered on 50% of emissions, phased in at 40% in 2024, 70% in 2025 and
100% from 2026. Effective coverage: 20%, 35%, 50%. The allowance is quoted in EUR and the
arb in USD, so FX is a term in the calculation.

**Order of magnitude, not to be oversold:** with realistic parameters, this term is worth
on the order of 0.2 $/t in 2024 and up to ~0.9 $/t at full phase-in. On an arb of a few
dollars, that's significant without being dominant. It's a term **nobody prices in**,
which is not the same thing as a term that decides everything.

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
  chains/coal.py        project B: ARA arb, ETS layer with phase-in and FX, CV energy
                        basis, controlled MCO, break test
  chains/products.py    project C: volume/mass conversion, Worldscale decomposition,
                        open-days illusion, TCE inversion
  ingest/contract.py    data contract — nothing goes in without a filled-in contract
  ingest/loader.py      raw exports -> canonical long format (date, ticker, value)
  ingest/series.py      long format -> series, + coverage table
  ingest/fixture.py     SYNTHETIC GENERATOR for project A, tickers prefixed SYNTH_
  ingest/fixture_coal.py  same for project B — the 2022 break is IMPOSED by hand
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

90 tests, all green, with no market data at all: 16 for project A, 22 for B, 20 for C,
the rest for the shared base. Every expected value is hand-computed in the comment
preceding it. Three tests carry their own argument:

- `test_omitting_the_control_biases_the_freight_coefficient` — why TTF is mandatory in
  project B
- `test_reset_moves_cost_with_zero_market_move` — project C's Worldscale result, in its
  barest form
- `test_implied_freight_round_trips_through_the_tce_engine` — if this loop doesn't close,
  the whole C-2 variant collapses
