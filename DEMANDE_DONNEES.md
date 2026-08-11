# Data request — the list, the files, the access

Operational document. Open it in front of the terminal, or send it to whoever grants access.

**Code status:** all three projects are written, tested, and wired up. 90 tests green, no
market data required to run them. All three dashboards already render end to end on
synthetic datasets. **Only the columns of numbers are missing.**

---

## 1. On one page

| Priority | Project | Series | Licence needed |
|---|---|---|---|
| **P0** | A — iron ore | 4 | **none** |
| **P0** | B — coal | 6 | **none** |
| P1 | C — distillate | 4 | none for the arb, **Worldscale for the main result** |
| P2 | all | 4 flow series | none, all public |

Translation: **with a standard Bloomberg terminal, no special entitlement needed, A and B
are entirely doable.** C is 70% doable, the rest depends on a single lock.

---

## 2. Three checks to make before downloading anything

Each takes five minutes and can change the plan.

| # | Check | Why it's decisive |
|---|---|---|
| **V1** | Volume and open interest on the **SGX M65F** contract (65% Fe ore) | If the contract is too thinly traded, the computed 65-62 premium is a quoting artefact, not a market price. Fall back to a **62-58** premium (FEF against the SGX 58% Fe contract): ten minutes of code now, two wasted weeks if discovered at the end |
| **V2** | Is the **TC2 / TC14** contract quoted in **USD per tonne** or in **Worldscale points**? | In USD/t: project C's Worldscale lock disappears. In WS points: the flat-rate table is needed, and it's the only real blocker across the three projects |
| **V3** | Does the **`BALTIC`** function respond on the terminal? | If yes: spot routes C3, C5, C4, TC14 live instead of FFAs. Project A loses a caveat (the FFA-vs-index basis) and project B finds its C4 freight |

Note the answers here:

```
V1 — M65F volume / OI:
V2 — TC2 quoted in:
V3 — BALTIC responds:
```

---

## 3. Files to produce

One file per project, dropped into `data/raw/`. **Wide format accepted** — it's the
natural format for a Bloomberg export.

```
data/raw/ironore.csv
data/raw/coal.csv
data/raw/products.csv
```

Expected content: a `date` column, then one column per series, named with the exact
ticker used.

```csv
date,SGX_FEF_M1,SGX_M65F_M1,SGX_C3F_M1,SGX_C5F_M1
2024-01-02,138.25,153.40,20.15,10.05
2024-01-03,137.10,152.05,20.40,10.10
```

### Three rules, and they matter more than speed

1. **Don't fill in any gap.** Public holiday, suspended quote, dead series: the pipeline
   knows how to handle them and reports how many dates survive at the intersection of the
   calendars. A gap filled by hand is undetectable afterward.
2. **Don't apply any unit conversion.** Drop in the native unit and declare it in
   `data_dictionary.csv`. A conversion done in Excel leaves no trace, and all three
   projects turn precisely on unit conversions being done correctly.
3. **The longest history available.** Ideally 2019 for A and B, 2021 for C. Without
   several regimes, there's nothing to show.

Then: fill in `data_dictionary.csv` (columns `exchange_code`, `bbg_ticker`, `verified`,
`verified_date`), then `REAL_TICKERS` at the top of the corresponding Streamlit page.

---

## 4. Project A — iron ore. **4 series, no licence.**

File: `data/raw/ironore.csv`

| # | Series | Exchange code | Unit | Freq. | Prio |
|---|---|---|---|---|---|
| A1 | 62% Fe ore CFR China | SGX **FEF**, nearest contract | USD/dmt | daily | P0 |
| A2 | 65% Fe ore CFR China | SGX **M65F** | USD/dmt | daily | P0 |
| A3 | Freight route C3 Tubarão → Qingdao | FFA SGX **C3F** | USD/wmt | daily | P0 |
| A4 | Freight route C5 W. Australia → Qingdao | FFA SGX **C5F** | USD/wmt | daily | P0 |
| A5 | Capesize 5TC (fallback) | SGX **CWF** | USD/day | daily | P2 |

**Watch the units, it's the heart of the project:** A1 and A2 are in **dry** tonnes
(dmt), A3 and A4 in **wet** tonnes (wmt). Whatever you do, don't harmonise them — the
conversion is exactly what the code does, and ignoring it is the error the project
demonstrates.

Free fallback if the terminal is unavailable: Barchart, roots `KW3` (C3F), `KWD` (C5F),
`KWC` (5TC).

---

## 5. Project B — Atlantic coal. **6 series, no licence.**

File: `data/raw/coal.csv`

| # | Series | Exchange code | Unit | Freq. | Prio |
|---|---|---|---|---|---|
| B1 | API2 CIF ARA | ICE **ATW** | USD/t | daily | P0 |
| B2 | API4 FOB Richards Bay | ICE **AFR** | USD/t | daily | P0 |
| B3 | Freight C4 Richards Bay → Rotterdam | Baltic route C4 | USD/t | daily | P0 |
| B4 | **European gas TTF** | ICE TTF | EUR/MWh | daily | **P0** |
| B5 | EUA price | ICE EUA | EUR/tCO2 | daily | P1 |
| B6 | EURUSD | any source | — | daily | P1 |
| B7 | Freight C7 Bolivar → Rotterdam | EEX C7 | USD/t | daily | P2 |

**B4 is P0 and it's not negotiable.** The project's thesis is that the ARA arb lost its
constraint in 2022 because of the reorientation toward India. But 2022 is also the
European gas shock. On the synthetic set, where the true slope is known (0.15), the
regression **without** the TTF control gives 0.71 and **with** control gives 0.18:
without TTF, the conclusion is reversed. One free series decides the project's validity.

**B3 is the known weak point.** The existence of a liquid C4 futures contract isn't
confirmed. If C4 isn't available: fall back to Capesize 5TC plus a route ratio, which
degrades the precision of the freight level and must be stated in the dashboard.

**B5-B6 feed the ETS layer.** The allowance is in EUR and the arb in USD, so FX is a term
in the calculation. Order of magnitude of the ETS term: ~0.2 $/t in 2024, ~0.9 $/t at
full phase-in. Real, not to be oversold.

---

## 6. Project C — transatlantic distillate. **4 series, plus one lock.**

File: `data/raw/products.csv`

| # | Series | Source / code | Unit | Freq. | Prio |
|---|---|---|---|---|---|
| C1 | ULSD spot US Gulf Coast | **EIA**, free, long history | USD/gal | daily | P0 |
| C2 | ARA gasoil | ICE Low Sulphur Gasoil futures | USD/t | daily | P0 |
| C3 | Freight route TC14 USGC → Continent | ICE / CME, Barchart root `IT2` for TC2 | WS points **or** USD/t | daily | P0 |
| C4 | **Worldscale flat rates** for the route | Worldscale Association | USD/t at WS100 | annual | **the lock** |
| C5 | VLSFO Rotterdam or Houston | Ship & Bunker, vlsfo.com | USD/t | daily | P1 |
| C6 | Average MR TCE | Clarksons, or a broker | USD/day | daily | P1 |

**C1 and C2 are in different units, and that's the point.** The US leg is in $/gallon,
the European leg in $/tonne. The conversion runs through a density (~7.45 bbl/t). On a
leg at ~780 $/t, moving from 7.45 to 7.50 shifts the price by **~5 $/t** — often more
than the whole arb. **The conversion factor everyone treats as a constant weighs as much
as the signal.** That's project C's central result.

**C4, the lock.** Tanker freight is quoted in WS points, and `$/t = WS/100 ×
flat_rate`, with the flat rate resetting every January 1st. On the synthetic set, the
largest jump moves the cost by **11.78 $/t**, of which **11.96 is attributable to the
reset alone** and **−0.13 to the market** — against an arb whose mean is −1.02 $/t. The
reset weighs ten times the signal, and it's invisible to anyone watching the WS points.

**If C4 is unavailable**, two fallbacks are already coded:

- **C-3**: the arb still gets built with C1 through C3. Only the Worldscale result falls
  away.
- **C-2**: freight isn't bought, it's **computed**. C5 and C6 are enough: the TCE engine
  is inverted to back out the freight rate a market TCE implies, and it's compared
  against the quoted freight wherever there's a point of comparison. "I didn't buy the
  freight, I reconstructed it" is a stronger argument than a subscription.

What to actually ask for on C4: **the route's flat-rate table, year by year, even just
for past years.** One value per year is enough.

---

## 7. Flow series — P2, free, but this is the portfolio's throughline

What sets these three projects apart: each one confronts the margin with an official
physical-flow series. The question isn't just "is the arb open," it's **"did the cargo
leave."**

| Project | Series | Source | Freq. |
|---|---|---|---|
| A | Chinese ore imports by origin | Chinese customs (GACC) | monthly |
| B | European imports of South African coal | Eurostat | monthly |
| B | Indian coal imports by origin | Indian trade statistics | monthly |
| C | US distillate exports by destination | EIA | monthly |

All free. The monthly frequency is a resolution limit **stated up front** in each
project, not a surprise discovered afterward.

---

## 8. Bloomberg entitlements — what to test, and what it changes

| Entitlement | What it unlocks | Verdict |
|---|---|---|
| **Standard terminal, no extras** | All the futures: SGX FEF, M65F, C3F, C5F, CWF; ICE ATW, AFR, TTF, EUA, Gasoil | **Enough for A and B in full** |
| **Baltic** (`BALTIC`) | Spot routes C3, C5, C4, TC14. Removes project A's FFA-vs-index basis, resolves project B's C4 freight | Strong convenience, not blocking |
| **Worldscale** | The flat-rate table | **The only real lock**, and it only concerns C |
| **Platts** | IODEX spot, premiums | Not needed: the SGX futures cover A |
| **Argus** | EBOB, underlying API indices | Not needed in the distillate version of C |
| **Clarksons SIN** | MR TCE, distances, fleet specifications | Would make C-2 solid rather than approximate |
| **Kpler / Vortexa** | Vessel-by-vessel flow, daily | The biggest possible gain: would remove all three projects' monthly resolution limit |

**Procedure for each ticker:** start from the exchange's contract code, go through the
contract-table menu (`CTM`), confirm with `DES` that the series is indeed the one
expected, and only then write the ticker into `data_dictionary.csv` with `verified =
yes` and the date. **No Bloomberg ticker gets written into this repo without having been
seen on a screen.**

---

## 9. Ready-to-copy message

> Hello,
>
> I need an export of daily closing-price history, over the longest history available,
> in CSV and **in native units, with no reprocessing**.
>
> **Iron ore and dry freight (SGX)**: 62% Fe futures (FEF), 65% Fe futures (M65F), Capesize
> FFA routes C3 and C5, and the Capesize 5TC.
>
> **Coal and energy (ICE)**: API2 Rotterdam (ATW), API4 Richards Bay (AFR), TTF, EUA, and
> the Baltic route C4 freight if you have access to it.
>
> **Oil products**: ICE Low Sulphur Gasoil, and the TC14 route freight — please specify
> whether it's quoted in Worldscale points or in USD per tonne.
>
> Two side questions: is the BALTIC function accessible on the terminal, and is the
> Worldscale flat-rate table available, even just for past years?
>
> Thank you very much.

Nothing in there is confidential or exotic: these are listed futures contracts.

---

## 10. What to do if a series is missing

| Missing | Fallback | Cost |
|---|---|---|
| M65F illiquid | 62-58 premium with the SGX 58% Fe contract | Less elegant, but liquid |
| C3F or C5F | Capesize 5TC + route ratio | Loss of level precision, to be stated |
| C4 | Capesize 5TC + route ratio | Same |
| Worldscale flat rates | C-3 and C-2 variants, already coded | The Worldscale result falls away, the project still holds |
| MR TCE | Estimate from bunkers and a public charter rate | C-2 becomes indicative |
| Flow series | Nothing — the validation sections stay pending | The projects run, validation is missing |

**None of these fallbacks prevents delivering all three projects.** The only result
genuinely lost without the licence is the Worldscale reset result.
