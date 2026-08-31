"""Canonical project registry — every project, its tier, its state.

`app/Home.py` renders this list and nothing else: adding a `Project` here puts it on the
platform, grouped by tier, whether it is built or not.

Deliberately Python rather than a config file: a project's metadata sits next to the type
that constrains its shape, and nothing here can drift from what the dashboard shows.
"""
from __future__ import annotations

from dataclasses import dataclass

STATUS_READY = "ready"        # engine and dashboard exist, tested
STATUS_PLANNED = "planned"    # scoped, nothing written

DATA_SYNTHETIC = "synthetic"  # fabricated to impose the phenomenon, waiting on real data
DATA_REAL = "real"            # runs on the Bloomberg export
DATA_HYBRID = "hybrid"        # main legs real, a minor term stays parameterised (documented in the engine)

TIER_1 = "AGRI · TIER 1 — sourced disagreements"
TIER_2 = "AGRI · TIER 2 — inferred structural tensions"
TIER_3 = "AGRI · TIER 3 — disagreements open as of August 2026"

# Data-access risk, as established by the spec's gates.
GATE_NONE = "none"            # free, public series
GATE_MEDIUM = "medium"        # one licensed series, fallback coded
GATE_HIGH = "high"            # test before writing any code


@dataclass(frozen=True)
class Project:
    id: str                       # "freight_cf" — matches chains/<id>.py
    code: str                     # "T1-1" — the spec reference
    tier: str
    title: str
    thesis: str                   # the one-line claim, bold on the card
    disagreement: str             # where the disagreement comes from, and between whom
    pivot: str                    # the tipping point — the page's deliverable
    mail_question: str            # the question only an insider can settle
    targets: str                  # the target pool
    data_gate: str
    data_fallback: str | None     # what happens if the gate fails
    status: str
    dashboard_page: str | None
    chain_module: str | None
    n_tests: int | None
    data_mode: str = DATA_SYNTHETIC


PROJECTS: list[Project] = [
    # ======================================================================
    # TIER 1 — sourced, quotable disagreements
    # ======================================================================
    Project(
        id="freight_cf",
        code="T1-1",
        tier=TIER_1,
        title="Freight inside the C&F calculation",
        thesis=(
            "On the marginal cargo freight does not add noise to the arb, it decides it — "
            "and the internal rate that sets it is a volume policy wearing an accounting "
            "convention."
        ),
        disagreement=(
            "Mat Halsall interview (Commodity Conversations, 25 Nov 2024): at Louis Dreyfus, "
            "recurring arguments between trading desks and the freight department, with "
            "traders disputing the rate without knowing its components. The desk says "
            "\"your rate is not the market\"; freight replies \"you are looking at an index, "
            "not a cost\". SETTLED ON REAL DATA: reading the published P8 rate without "
            "charging ballast implies a TCE above the peak of the 2021 dry bulk boom on "
            "99 % of the last five years — arithmetically untenable. The ceiling comes from "
            "the segment of the series the export quotes in USD per day, isolated first as "
            "a data defect."
        ),
        pivot=(
            "Three costs the desks do not argue about: the ballast is worth ~20 USD/t and "
            "twice as much out of Santos as out of the PNW, the market bounds it below at "
            "a median 0.28 without settling it, and the smoothed internal rate sits on one "
            "side of spot for up to 126 sessions"
        ),
        mail_question=(
            "On Santos-Qingdao the ballast leg is worth about 20 USD per tonne of grain — "
            "~1.3 M on a Panamax cargo — and it is stable across seventeen years, because "
            "ballast is time rather than fuel. Tested against the route's own recorded TCE "
            "peak, reading the rate at zero ballast is tenable on only 1% of days, so the "
            "trading desk's position does not survive; but the binding lower bound has a "
            "median of 0.28, not 1. The market rules zero out and does not rule full cost "
            "in. On the marginal band that parameter decides whether the cargo clears at "
            "all, so the internal rate is effectively setting tonnage. Is that owned "
            "explicitly — calibrated against a volume or utilisation target — or negotiated "
            "as P&L attribution, with the volume effect falling between the two mandates?"
        ),
        targets="Freight desks (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra) and grain/oilseed traders",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "The arb's price legs (FOB Santos, CIF China) are absent, so the page covers the "
            "freight term alone and says so explicitly."
        ),
        status=STATUS_READY,
        dashboard_page="pages/4_T1_1_Freight_CF.py",
        chain_module="agri.chains.freight_cf",
        n_tests=92,
        data_mode=DATA_HYBRID,
    ),
    Project(
        id="hedge_cost",
        code="T1-2",
        tier=TIER_1,
        title="The full cost of hedging — cocoa and coffee",
        thesis="The binding constraint is not the price, it is the collateral.",
        disagreement=(
            "Barry Callebaut H1 2024/25: initial margins up ninefold, backwardation cost 60 % "
            "dearer at the peak. Coffee, Nov 2025: roughly 7 bn USD of margin calls in one "
            "month; at Montesanto Tavares the cost of carrying hedges went from 74 % to "
            "158 % of trade receivables, judged unsustainable by their own lawyers. VERIFIED "
            "ON REAL ICE PRICES: NY cocoa did peak at 12 565 USD/t on 18 Dec 2024, tying up "
            "1.08 bn USD of cash on a 100 kt book — no longer reconstructed orders of "
            "magnitude but the market's own numbers on the day."
        ),
        pivot="IM* — the initial margin at which hedging capacity falls below the physical book",
        mail_question=(
            "At what level of initial margin does your desk stop adding physical because the "
            "hedge no longer finances? A formalised limit, or one discovered along the way?"
        ),
        targets="ofi/Olam, ECOM, Volcafe, Sucden Coffee, Touton, Barry Callebaut, Cargill Cocoa, Freepoint softs",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "No real deferred contract available, so roll cost is neutralised "
            "(deferred = front) and shown as a limitation rather than estimated."
        ),
        status=STATUS_READY,
        dashboard_page="pages/5_T1_2_Hedge_Cost.py",
        chain_module="agri.chains.hedge_cost",
        n_tests=43,
        data_mode=DATA_HYBRID,
    ),
    # ======================================================================
    # TIER 2 — inferred tensions. "It seems to me", never "I read that".
    # ======================================================================
    Project(
        id="crush_tracking",
        code="T2-3",
        tier=TIER_2,
        title="The board crush is not a price, it is a yield in disguise",
        thesis=(
            "The coefficients 0.022 and 0.11 are not unit conversions but frozen yields "
            "(44 lb and 11 lb per bushel). Hedging on the board means accepting them as "
            "your own and keeping the difference as an open position."
        ),
        disagreement=(
            "Inferred tension. Board traders treat the CBOT crush as a hedgeable proxy for a "
            "plant's economics; plant people reply that domestic meal basis, real yields and "
            "logistics break the hedge exactly when it matters. MEASURED ON REAL CBOT DATA: "
            "in the tightest margin decile, a 0.5 lb per bushel error on meal yield — 1.2 % "
            "of the yield the contract assumes — is enough to erase the entire net margin."
        ),
        pivot=(
            "The yield precision the board silently demands, in pounds per bushel — and its "
            "collapse in the tight-margin regime"
        ),
        mail_question=(
            "How far does your real meal yield drift from the board's 44 lb over a campaign, "
            "and does anyone hedge that gap separately — or does it just sit in the result?"
        ),
        targets="US and Brazilian crushers (ADM, Bunge, Cargill, LDC, CHS), oilseed risk managers",
        data_gate=GATE_NONE,
        data_fallback=(
            "Local cash prices are absent from the export, so no tracking error is measured; "
            "the page is built not to need them — the inversion only requires the board."
        ),
        status=STATUS_READY,
        dashboard_page="pages/6_T2_3_Crush_Tracking.py",
        chain_module="agri.chains.crush_tracking",
        data_mode=DATA_HYBRID,
        n_tests=36,
    ),
    Project(
        id="white_premium",
        code="T2-4",
        tier=TIER_2,
        title="The white premium, or what a price can and cannot tell you",
        thesis=(
            "The LEVEL of the refining rent is not identifiable from prices — a conversion "
            "factor nobody publishes weighs as much as the answer. Its VARIATION is: it "
            "shifted by roughly 60 USD/t."
        ),
        disagreement=(
            "Inferred tension. The white premium (No.5 − No.11) is presented as the refining "
            "margin; it seems to me it mostly contains a residual of positioning and delivery "
            "constraints. MEASURED ON REAL ICE No.11/No.5 with a real Henry Hub energy proxy: "
            "the polarisation adjustment that would zero the rent is 1.0852, just above the "
            "plausible band [1.06 ; 1.08] — price alone cannot settle it. But the ranking of "
            "years is identical at both bounds (rank correlation 1.0000), and richness moves "
            "from −26 USD/t in 2021 to +35 in 2024: a shift 5.5 times larger than the "
            "parameter uncertainty."
        ),
        pivot=(
            "The price the market pays for the act of refining (~70 USD/t), observed with no "
            "cost assumption at all — a refiner does the subtraction themselves"
        ),
        mail_question=(
            "Is your all-in refining cost of the order of 70 USD/t? And what changed on your "
            "side between 2021 and 2024, when the price paid for refining shifted by roughly "
            "60 USD/t?"
        ),
        targets="Destination refiners (Al Khaleej, ASR, Tereos, Südzucker) plus Sucden, Czarnikow, Alvean, Wilmar, ED&F Man",
        data_gate=GATE_NONE,
        data_fallback=(
            "Refining labour and freight remain parameterised flat rates — no refinery cost "
            "accounting is public as a time series."
        ),
        status=STATUS_READY,
        dashboard_page="pages/7_T2_4_White_Premium.py",
        chain_module="agri.chains.white_premium",
        n_tests=24,
        data_mode=DATA_HYBRID,
    ),
    Project(
        id="plant_option",
        code="T2-5",
        tier=TIER_2,
        title="The plant as an option on the margin",
        thesis=(
            "A curtailment rule of the form \"margin below zero for N periods\" implements a "
            "hysteresis band whether it means to or not — and a band corresponds to a precise "
            "round-trip switching cost."
        ),
        disagreement=(
            "Not a market dispute but a critique of a rule that can be pointed at: the "
            "`consecutive_below(margin, 0, N=4)` signal used on the zinc and lithium pages, "
            "which silently assumes stopping and restarting is free. INVERTED ON THE REAL "
            "CHINESE CRUSH MARGIN: the rule stops at −39 CNY/t and restarts at +43, which "
            "implies a round trip of about 143 CNY/t of beans crushed. The reader does not "
            "have to accept the model, only compare one number to their own."
        ),
        pivot="The implied round-trip switching cost, in CNY per tonne of beans crushed",
        mail_question=(
            "Is your shutdown rule a margin threshold, or does the restart cost move it "
            "explicitly? Does 143 CNY/t look like your order of magnitude?"
        ),
        targets="Desk management, corporate development, crushing and smelting operators",
        data_gate=GATE_NONE,
        data_fallback=(
            "Shutdown, restart and idling costs are not public anywhere — they are the "
            "page's sliders, and the inversion is what makes that acceptable."
        ),
        status=STATUS_READY,
        dashboard_page="pages/8_T2_5_Plant_Option.py",
        chain_module="agri.chains.plant_option",
        n_tests=36,
        data_mode=DATA_HYBRID,
    ),
    Project(
        id="oil_substitution",
        code="T2-6",
        tier=TIER_2,
        title="The palm–soy substitution bound does not exist",
        thesis=(
            "Tested in the one window where the exchange rate contaminates nothing — the "
            "seven years of the ringgit peg — the hypothesis comes out INVERTED: narrow gaps "
            "revert in 12 days, wide ones do not revert at all."
        ),
        disagreement=(
            "Inferred tension. It seems to me crushers hold palm/soy elasticity to be strong "
            "and formulators hold it to be sticky — reformulating a recipe takes months. "
            "MEASURED ON REAL BURSA + CBOT: palm quotes in ringgit and the export has no "
            "USDMYR, so the spread is computable ONLY over the 1998–2005 peg, where the "
            "exchange rate is a constant fixed by decree. On that clean window there is no "
            "substitution bound — and the naive test that appears to find one is in fact "
            "selecting the 2004–05 era."
        ),
        pivot=(
            "The absence of mean reversion beyond 54 USD/t of deviation — therefore: fading "
            "a wide palm–soy spread has no empirical support"
        ),
        mail_question=(
            "At what palm–soy gap does your phone actually ring for a recipe change today? "
            "My clean window stops in 2005, before biodiesel, and the bound may well have "
            "moved since."
        ),
        targets="Vegetable oil crushers and refiners (Wilmar, Musim Mas, Golden Agri, Bunge, Cargill), food formulators",
        data_gate=GATE_NONE,
        data_fallback=(
            "USDMYR is absent from the export (FREE, a single ticker) so the test is limited "
            "to the seven peg years. Fetching it would unlock thirty years instead of seven: "
            "the highest-value missing series in the portfolio."
        ),
        status=STATUS_READY,
        dashboard_page="pages/9_T2_6_Oil_Substitution.py",
        chain_module="agri.chains.oil_substitution",
        data_mode=DATA_REAL,
        n_tests=32,
    ),
    # ======================================================================
    # TIER 3 — disagreements still open
    # ======================================================================
    Project(
        id="feedstock_lcfs",
        code="T3-1",
        tier=TIER_3,
        title="Two subsidies that contradict each other",
        thesis=(
            "Both camps argue about the LCFS credit price, which cannot settle it: across "
            "the programme's entire realised range it moves the discount imported UCO must "
            "hold by only 3.2 c/lb. What decides is the UCO–soyoil price spread."
        ),
        disagreement=(
            "Three quarters of US renewable diesel capacity was built on the coasts, sited to "
            "run on imported feedstock. Then 45Z excludes non-North-American feedstock from "
            "the credit while California's LCFS keeps paying for low carbon intensity "
            "regardless of origin — one policy penalises exactly what the other rewards. "
            "Camp A (coastal plants): the LCFS premium suffices, imports hold. Camp B (the "
            "soy complex): it does not, and soyoil takes the share."
        ),
        pivot=(
            "The discount in c/lb that imported UCO must hold under soyoil — a feedstock "
            "buyer confirms or denies it against their own book in ten seconds"
        ),
        mail_question=(
            "At 75 USD/t CO2e on the LCFS, I find imported UCO must sell about 4.5 c/lb "
            "under soyoil purely to offset a tax credit it cannot claim, and that across the "
            "programme's whole history that number can only have varied by 3.2 c/lb. Is the "
            "discount you see delivered USGC of that order? And below what collected price "
            "do you simply stop loading?"
        ),
        targets="Bunge, ADM, LDC, Cargill, CHS plus the bio desks at Vitol, Gunvor, Freepoint, Trafigura — the portfolio's largest target pool",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "UCO prices are unavailable (Platts/PGA) — the page is built never to need them: "
            "the deliverable is a relative discount, not an absolute price. The LCFS credit "
            "is absent from the export (published by CARB) and is treated as an axis."
        ),
        status=STATUS_READY,
        dashboard_page="pages/10_T3_1_Feedstock_LCFS.py",
        chain_module="agri.chains.feedstock_lcfs",
        data_mode=DATA_HYBRID,
        n_tests=74,
    ),
    Project(
        id="sugar_mix",
        code="T3-2",
        tier=TIER_3,
        title="The \"Brazilian cost floor\" is an exchange rate in disguise",
        thesis=(
            "A production cost is denominated in reais. Translated into cents per pound for "
            "a New York reader, a CONSTANT cost produces a floor that varies by 20.8 c/lb — "
            "more than the market's own range — purely through USDBRL."
        ),
        disagreement=(
            "Hedgepoint (Feb 2026): the mix should fall towards 46 % to cut the surplus, but "
            "mill limits and sugar already sold forward prevent it. Czarnikow (Jun 2026): "
            "mills enter far less hedged than the previous four seasons, and 2026/27 pricing "
            "stayed below BRL 2 000/t, below the cost of production. VERIFIED ON REAL NY11 + "
            "USDBRL: Czarnikow's claim holds — sugar is worth 1 843 BRL/t at the last print, "
            "and more than 80 % of 2026 traded below that threshold."
        ),
        pivot=(
            "The NY11 floor implied by a constant cost in reais — a series, not a level, and "
            "its rank correlation with the inverse exchange rate is exactly 1"
        ),
        mail_question=(
            "Does your team reason on a Brazilian cost floor in cents per pound, or recompute "
            "it on every move in the real? And for 2026/27, at what entry hedge ratio did "
            "your mills actually start the season?"
        ),
        targets="Sugar desks at Sucden, Czarnikow, Alvean, Wilmar, LDC Sugar, ED&F Man, Copersucar, BP Bioenergy",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "UNICA mix and CEPEA ethanol are absent from the export (both FREE) so the "
            "conditional elasticity is left unestimated and its specification shown as-is "
            "rather than simulated. Everything else runs on real NY11 + USDBRL."
        ),
        status=STATUS_READY,
        dashboard_page="pages/11_T3_2_Sugar_Mix.py",
        chain_module="agri.chains.sugar_mix",
        data_mode=DATA_HYBRID,
        n_tests=22,
    ),
    Project(
        id="china_soy",
        code="T3-4",
        tier=TIER_3,
        title="The windows in which no origin works",
        thesis=(
            "Rather than testing for a political signature — which needs auction data nobody "
            "publishes — date the periods when the origination budget is NEGATIVE: a free "
            "bean, freighted free, would still not make the crush pay. Any cargo arriving "
            "then is non-commercial by arithmetic."
        ),
        disagreement=(
            "Sinograin sold about half of the 504 000 t offered at its largest auction since "
            "January; traders quoted by Reuters read it as making room for US cargoes "
            "(Aug 2026). Against that, ADM raised its 2026 outlook betting China keeps buying "
            "US beans. MEASURED ON REAL CBOT + DCE + USDCNY: 2.0 % of sessions since 2018 "
            "show a negative origination budget, all concentrated in 2023, including a "
            "29-day window from 7 June to 5 July."
        ),
        pivot=(
            "The dated calendar of impossible windows — dates that can be checked against an "
            "arrival book, not a coefficient"
        ),
        mail_question=(
            "Did you fix China cargoes during the June–July 2023 windows when the basis plus "
            "freight budget was negative? And if so, was the crush margin really the "
            "constraint, or was another link in the chain carrying the result?"
        ),
        targets="Oilseed origination (COFCO, Sinograin, Bunge, LDC, Cargill), China soy desks",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "Sinograin auctions and GACC customs are absent, so the logit signature test is "
            "not run. The origination budget replaces it and depends on NEITHER the basis NOR "
            "the freight — the two missing series drop out of the calculation instead of "
            "entering it."
        ),
        status=STATUS_READY,
        dashboard_page="pages/12_T3_4_China_Soy.py",
        chain_module="agri.chains.china_soy",
        data_mode=DATA_REAL,
        n_tests=22,
    ),
]


def by_tier() -> dict[str, list[Project]]:
    grouped: dict[str, list[Project]] = {}
    for project in PROJECTS:
        grouped.setdefault(project.tier, []).append(project)
    return grouped


def ready_projects() -> list[Project]:
    return [p for p in PROJECTS if p.status == STATUS_READY]


def total_tests() -> int:
    return sum(p.n_tests or 0 for p in PROJECTS)


def get(project_id: str) -> Project:
    for project in PROJECTS:
        if project.id == project_id:
            return project
    raise KeyError(f"unknown project id: {project_id!r}")


__all__ = [
    "DATA_HYBRID",
    "DATA_REAL",
    "DATA_SYNTHETIC",
    "GATE_HIGH",
    "GATE_MEDIUM",
    "GATE_NONE",
    "PROJECTS",
    "Project",
    "STATUS_PLANNED",
    "STATUS_READY",
    "TIER_1",
    "TIER_2",
    "TIER_3",
    "by_tier",
    "get",
    "ready_projects",
    "total_tests",
]
