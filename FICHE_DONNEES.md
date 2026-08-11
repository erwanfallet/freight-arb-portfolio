# Data sheet — what to get, for A, B and C

The point of this document: so you can go ask for access, or dig through a database you
come across, **without having to ask me again what to look for**.

Reminder of the rule: **no Bloomberg ticker gets written here.** What's written is the
**exchange contract code**, which is publicly verifiable. On the terminal, you go from
the exchange code to the Bloomberg ticker via the contract-table menu (`CTM`), then `DES`
to confirm the series is the right one, and only then do you write it into
`data_dictionary.csv` with `verified = yes` and the date.

---

## 0. The input format, so it works on the first try

You drop CSVs into `data/raw/`. Two formats accepted, auto-detected.

**Long format** (preferred):

```csv
date,ticker,value
2024-01-02,SGX_FEF_M1,138.25
2024-01-03,SGX_FEF_M1,137.10
```

**Wide format** (a typical Bloomberg export):

```csv
date,SGX_FEF_M1,SGX_M65F_M1,SGX_C3F_M1,SGX_C5F_M1
2024-01-02,138.25,153.40,20.15,10.05
```

Three rules:

- **Never fill in a gap by hand.** A public holiday, a suspended quote, a dead series:
  the pipeline knows how to handle them and will tell you how many dates survived at the
  intersection of the calendars. A gap filled by hand is invisible and undetectable.
- **Don't convert units before dropping the file in.** Drop in the native unit, declare
  it in `data_dictionary.csv`, and let the code convert. A conversion done in Excel
  leaves no trace.
- **One ticker = one stable name.** The same ticker in the file and in the dictionary.

Then fill in `REAL_TICKERS` at the top of `app/pages/1_Iron_Ore_Premium.py` and the
dashboard switches from synthetic mode to real data.

---

## 1. Project A — iron ore. **Four series, and it's up and running.**

This is the least demanding of the three projects. Nothing here needs Platts, Baltic, or
Argus.

| # | What's needed | Precise identification | Where | Status |
|---|---|---|---|---|
| A1 | 62% Fe CFR China price, daily, USD/dmt | SGX TSI Iron Ore 62% Fe futures, code **FEF**, nearest contract | SGX, or Bloomberg via `CTM` | Widely used code, to confirm |
| A2 | 65% Fe CFR China price, daily, USD/dmt | SGX MB Iron Ore 65% Fe futures, code **M65F** | same | Existence confirmed |
| A3 | Freight route C3 Tubarão → Qingdao, USD/wmt | Capesize FFA route C3, code SGX **C3F** | SGX; free fallback Barchart root `KW3` | Confirmed |
| A4 | Freight route C5 W. Australia → Qingdao, USD/wmt | Capesize FFA route C5, code SGX **C5F** | SGX; free fallback Barchart root `KWD` | Confirmed |

**The longest history possible**, ideally 2019 to today: it needs to cover both the wide
high-grade premium regime of 2019 and its collapse in 2022-2024, otherwise the
decomposition only has one regime to show and the project loses its interest.

**The check to make first, before downloading anything else:** volume and open interest
on **M65F**. If the contract is too thinly traded, the series is noisy, the computed
premium is a quoting artefact, and the fallback is a 62-58 premium (FEF against the SGX
58% Fe contract) — less elegant, but liquid. That switch costs ten minutes in the code,
and two weeks if discovered at the end.

### Nice-to-have series, not blocking

| What's needed | Why | Where |
|---|---|---|
| Chinese ore imports by origin, monthly | S4's flow validation: does the Brazilian share lead the C3 − C5 widening | Chinese customs (GACC), free |
| Capesize 5TC | Level fallback if C3/C5 are a problem | Barchart root `KWC` |
| Moisture by origin | Assumption A-H2, not a series | Producers' public specifications |

---

## 2. Project B — Atlantic coal. **Two free price series, one weak point.**

| # | What's needed | Identification | Where | Status |
|---|---|---|---|---|
| B1 | API2 CIF ARA, daily | ICE Rotterdam Coal futures, root **ATW** | ICE; free on investing.com and TradingView (`ICEEUR-ATW1!`) | Confirmed |
| B2 | API4 FOB Richards Bay, daily | ICE Richards Bay Coal futures, root **AFR** | same (`ICEEUR-AFR1!`) | Confirmed |
| B3 | Freight C4 Richards Bay → Rotterdam | Baltic route C4 | **to look for** | **Weak point.** The existence of a liquid C4 futures contract isn't confirmed |
| B4 | Freight C7 Bolivar → Rotterdam | EEX Baltic Capesize C7 futures | EEX | Existence confirmed |
| B5 | EUA price | ICE EUA futures | ICE; free on investing.com | To confirm |
| B6 | **European gas TTF** | ICE TTF futures | ICE; free | **Non-negotiable, see below** |
| B7 | EURUSD | any FX source | free everywhere | The ETS allowance is in EUR, the arb in USD: it's a term in the calculation |
| B8 | Bunker consumption for the RB → ARA voyage | Assumption, not a series | public fleet reports | Parameter B-H5. A Capesize at ~40 t/day over ~24 days of crossing gives the order of magnitude |

**Why B6 is non-negotiable.** Project B's thesis is that the ARA arb lost its constraint
in 2022 because the marginal Richards Bay cargo goes to India. But 2022 is also the year
of the European gas shock, which sent European coal demand soaring. **Without
controlling for TTF, the result will be attributed to the wrong mechanism**, and the
first competent reader of the email will spot it. That's project B's real intellectual
risk, and it's solved with one free series.

**What's new about project B, and that nobody prices in:** the European maritime ETS
cost since 2024 makes freight to Europe structurally more expensive than to India at
equal distance. Effective coverage of 20% in 2024, 35% in 2025, 50% from 2026 for a
non-EU voyage.

**And the order of magnitude, not to be oversold.** With realistic parameters — a
Capesize, ~960 t of bunkers on the crossing, 150,000 t of cargo, an allowance around 80
EUR — this term is worth on the order of **0.2 $/t in 2024** and **~0.9 $/t at full
phase-in**. On an arb of a few dollars that's significant, but it's not the term that
decides. Its value lies elsewhere: **nobody prices it in**, which isn't the same thing as
a dominant term. In the email, present it as a forgotten term, never as the explanation
for the break — otherwise the first reader who does the back-of-envelope math will catch
you on it, and you lose the rest of the message with it.

**Flow:** European imports of South African coal (Eurostat, monthly, free) and Indian
imports by origin (Indian trade statistics).

---

## 3. Project C — transatlantic products. **Two blockers, to resolve before coding.**

### Check number one, and it takes five minutes

The **TC2** contract exists at both ICE **and** CME ("TC2 37k mt"), and Barchart has its
history under root `IT2`. The decisive question:

> **Is this contract quoted in USD per tonne, or in Worldscale points?**

- **In USD/t** → the Worldscale problem disappears, project C can be built with exchange
  data.
- **In WS points** → the flat rates are needed to convert, and that's where it's
  blocked: I haven't found any free public source for the Worldscale table. It's
  published every November by the Worldscale Association for the following year, and
  adjustments go through subscriber notices.

### The second blocker

The **EBOB** price (Eurobob oxy barges ARA) is an Argus assessment, under licence. ICE
futures settled against this assessment probably exist, but I haven't verified the code
and I'm not going to invent it.

### The three ways forward, in order of preference

| Path | What's needed | What the project becomes |
|---|---|---|
| **C-1** | Worldscale flat rates + an EBOB price | The original project, with its best result: the January 1st flat-rate reset shifts the arb threshold at constant WS points |
| **C-3** | **Nothing beyond what's free.** Switch to transatlantic distillate: ICE ARA Gasoil against CME NY ULSD/Heating Oil, TC14 freight | Same trade, opposite direction, two liquid and free price legs. And a better story since 2022: Europe lost Russian diesel, the flow reversed and lengthened in tonne-miles |
| **C-2** | Distances, consumption, VLSFO, hire | Freight is no longer bought, it's **computed**: `voyage/tce.py` reconstructs the MR cost from the bottom up, and it's compared against the quoted route. Harder, more original, and it finally gives the module a use |

My take, if the flat rates don't come through: **C-3 + C-2 combined**. "I didn't buy the
freight, I computed it" is a stronger credibility argument than a subscription.

### What's already secured for C, for free

| What's needed | Where | Status |
|---|---|---|
| Conventional spot gasoline NY Harbor, daily | EIA | Confirmed, long history. **Careful**: the EIA's RBOB futures series stops after April 5, 2024 |
| US gasoline imports by country of origin, monthly | EIA | Confirmed — the official, free flow validation |
| PADD1 gasoline stocks, weekly | EIA | Confirmed |
| VLSFO Rotterdam / Singapore | Ship & Bunker, vlsfo.com, OilPriceAPI (free tier) | Current prices confirmed; a clean long history still to confirm |

---

## 4. Which database unlocks what

You don't yet know what you have access to. Here's the reading grid, so the day someone
offers you access you immediately know whether it's useful.

| Database | What it unlocks | Priority |
|---|---|---|
| **Bloomberg with Baltic entitlement** | Spot routes C3, C5, C4, TC2 live, instead of FFAs. Removes the FFA-vs-index basis (A-H3), so removes a caveat from project A. **Unlocks C-1 on the freight side.** | High |
| **Bloomberg with no special entitlement** | All the futures: SGX FEF, M65F, C3F, C5F, ICE ATW, AFR, EUA, TTF. **Fully enough for A and for B.** | Enough to get started |
| **Platts / S&P Global Commodity Insights** | IODEX 62 and 65 spot assessments, aluminium premiums, and adjusted Worldscale flat rates | Medium — the SGX futures already cover A |
| **Argus Direct** | EBOB, and the underlying API2/API4 indices | **High for C-1**, not needed for A and B |
| **Worldscale Association** | The flat-rate table. **C-1's only lock.** | High for C only |
| **Clarksons SIN** | Time-charter rates, distances, fleet specifications. **This is what would make C-2 solid** instead of approximate | High if going with C-2 |
| **Kpler or Vortexa** | Vessel-by-vessel flow, near real-time. Would replace all the monthly customs series with daily data, and **would remove the resolution limit stated in S4** | The biggest gain across the three projects |
| **CEIC or Wind** | Chinese customs data properly historised, without scraping | Real convenience on A |
| **LSEG / Refinitiv Eikon, Datastream** | Broad coverage, often futures and indices with no separate entitlement | Good substitute for Bloomberg |
| **Barchart / TradingView / investing.com, free** | Daily history of listed futures, a few years back | **Already enough to prototype A and B** |

### If asked "what exactly do you need?"

Short answer to give as-is:

> Daily closing-price series, over the longest history available, for four futures
> contracts: SGX iron ore 62% Fe (FEF) and 65% Fe (M65F), and the Capesize FFA routes C3
> and C5. In addition, the ICE coal futures API2 (ATW) and API4 (AFR), plus TTF and EUA.
> In CSV, native units, no reprocessing.

That's a mundane, non-confidential request that needs no exotic entitlement.

---

## 5. The three checks to make as soon as you have a terminal

1. **Does `BALTIC` respond?** If yes, you have the spot routes, project C regains its
   original form and project A loses a caveat.
2. **Is the TC2 contract quoted in USD/t or in WS points?** Decisive for project C.
3. **M65F volume and open interest.** Decisive for project A — it's the only thing that
   could force a change of index.

None of these three answers stops work from starting: project A's engine is written, the
16 golden tests are green, and the dashboard already runs end to end on synthetic data.
Only four columns of numbers are missing.
