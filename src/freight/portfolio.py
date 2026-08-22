"""Single registry for the whole portfolio — twelve projects, one source of truth.

`app/Home.py` renders this list and nothing else: adding a `Project` here puts it on the
platform, grouped by tier, whether it is built or not.

Deliberately Python rather than a config file: a project's metadata sits next to the type
that constrains its shape, and nothing here can drift from what the dashboard shows.

NOTE ON THE PACKAGE NAME. This package is still called `freight` for the same reason a
ticker outlives the company that issued it: renaming it would break the deployment for no
gain. The portfolio has since grown well past freight — grains, softs, biofuels, gas — but
the throughline is unchanged and it is worth stating plainly: **every project turns on the
quoted unit not being the economic unit.** Wet tonne versus dry tonne, kcal versus tonne,
gallon versus tonne, cents per bushel versus dollars, USD per day versus USD per tonne,
ringgit versus dollar. That is the portfolio, and the sector is incidental.
"""
from __future__ import annotations

# The canonical dataclass lives in `agri.portfolio`, which is where it was first written.
# Re-exported here so that the twelve projects are declared against one type.
from agri.portfolio import (  # noqa: F401
    DATA_HYBRID,
    DATA_REAL,
    DATA_SYNTHETIC,
    GATE_HIGH,
    GATE_MEDIUM,
    GATE_NONE,
    STATUS_PLANNED,
    STATUS_READY,
    Project,
)
from agri.portfolio import PROJECTS as AGRI_PROJECTS

TIER_FREIGHT = "FREIGHT — Dry bulk and refined products — freight as the deciding term"


FREIGHT_PROJECTS: list[Project] = [
    Project(
        id="iron_ore",
        code="A",
        tier=TIER_FREIGHT,
        title="The 65–62 premium contains a freight spread — but at half weight",
        thesis=(
            "Both indices are quoted CFR China, so the C3 − C5 freight differential is "
            "already inside the high-grade premium. Correct for moisture — freight is paid "
            "on the wet tonne, the index is quoted on the dry tonne — and decompose. What "
            "is left is a residual, and it is not renamed."
        ),
        disagreement=(
            "Inferred tension. Steel desks read the 65–62 premium as a pure quality signal "
            "about mill margins and blast-furnace productivity; freight desks point out "
            "that 65 % Fe is largely Brazilian (route C3) and 62 % largely Australian "
            "(route C5), so a Capesize spread sits inside the number before any quality "
            "argument starts."
        ),
        pivot=(
            "How much of the premium the freight differential accounts for, and the "
            "residual left over once moisture is handled correctly"
        ),
        mail_question=(
            "When the 65–62 premium moves, does your desk decompose it against C3 − C5 "
            "first, or is it read straight as a quality signal?"
        ),
        targets="Iron ore desks at Vale, Rio Tinto, BHP, Glencore, Trafigura; Capesize freight desks",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "The C3 route in the export is monthly (64 points) against a daily C5 — the "
            "frequency mismatch is surfaced by the resampling rules rather than silently "
            "forward-filled."
        ),
        status=STATUS_READY,
        dashboard_page="pages/1_Iron_Ore_Premium.py",
        chain_module="freight.chains.ironore",
        data_mode=DATA_REAL,
        n_tests=25,
    ),
    Project(
        id="coal",
        code="B",
        tier=TIER_FREIGHT,
        title="The switching ceiling predicts, and the efficiencies it is built on cannot",
        thesis=(
            "Distance from the coal-switching level does predict TTF's next 20 days — "
            "one-sided in the direction the physics requires, and surviving the Stambaugh "
            "bias this regressor is unusually exposed to. But the unpublished efficiency "
            "pair that supposedly defines the level is PROVABLY irrelevant to that "
            "prediction, and raw thermal parity with no carbon price in it predicts just "
            "as well."
        ),
        disagreement=(
            "Tested, not inferred, and the test cuts against its own construction. The "
            "regressor contains TTF in its numerator, so OLS is biased toward the finding: "
            "a Nelson-Kim bootstrap puts the honest p-value at 0.018 against ~0.001 read "
            "naively. What survives is one-sided (above the switch t = -2.6, below it "
            "nothing) — which mean reversion cannot mimic. What does not survive is the "
            "arithmetic itself: across the whole efficiency grid the level swings 30 to 47 "
            "EUR/MWh and the share of days above it 16% to 75%, while the t-statistic "
            "moves 0.03, because the efficiencies enter only affinely and a t-statistic is "
            "invariant under an affine map."
        ),
        pivot=(
            "The same efficiency grid moving the level by 45% and the prediction by "
            "nothing — two functionals of one object, one unidentified and one immune"
        ),
        mail_question=(
            "The data says there is a ceiling and cannot say it is a *switching* ceiling: "
            "the efficiency pair is provably irrelevant to the prediction, and raw thermal "
            "parity predicts as well as the full carbon-inclusive level. Does your desk "
            "use the switching level as a level — where to place a trade — or as a signal "
            "for when to place it? Because on this evidence it can support one of those "
            "and not the other."
        ),
        targets="European power and gas desks (Uniper, RWE, EDF Trading, Vitol, Glencore coal)",
        data_gate=GATE_NONE,
        data_fallback=(
            "API4 Richards Bay is absent from the export, so the original API2 − API4 arb "
            "is not computable. API2, TTF, EUA and EURUSD are all real and support the "
            "switching-ceiling test instead. What that test cannot reach: EU coal "
            "generation and plant availability are absent, so the saturation point — "
            "where finite coal capacity stops the ceiling binding — is untestable, and "
            "43 above-switch windows cannot separate the full switching level from raw "
            "thermal parity."
        ),
        status=STATUS_READY,
        dashboard_page="pages/2_Coal_Gas_Switching.py",
        chain_module="freight.chains.coal",
        data_mode=DATA_REAL,
        n_tests=24,
    ),
    Project(
        id="products",
        code="C",
        tier=TIER_FREIGHT,
        title="The transatlantic distillate arb is a unit conversion before it is a trade",
        thesis=(
            "The US leg is quoted in dollars per gallon and the European leg in dollars per "
            "tonne. The density conversion moves the number by more than the arb itself, so "
            "getting it wrong does not produce an error — it produces a plausible, wrong arb."
        ),
        disagreement=(
            "Inferred tension. The transatlantic diesel arb is discussed as a single spread, "
            "as though both legs were commensurable. They are quoted in different units on "
            "different exchanges, and the conversion factor between them is not a constant "
            "of nature — it depends on the density of the specific grade."
        ),
        pivot=(
            "The density assumption the published arb implies, compared with the range real "
            "diesel grades actually occupy"
        ),
        mail_question=(
            "Which density does your transatlantic diesel arb use, and is it a desk "
            "convention or re-derived per cargo grade?"
        ),
        targets="Refined product desks (Vitol, Trafigura, Gunvor, Mercuria, BP, Shell Trading)",
        data_gate=GATE_NONE,
        data_fallback=(
            "NYMEX ULSD (cents per gallon) and ICE gasoil (USD per tonne) are both in the "
            "export with 9 000+ observations each — the arb is fully computable."
        ),
        status=STATUS_READY,
        dashboard_page="pages/3_Products_Transatlantic.py",
        chain_module="freight.chains.products",
        data_mode=DATA_REAL,
        n_tests=28,
    ),
    Project(
        id="grain_seasonal",
        code="D",
        tier=TIER_FREIGHT,
        title="The largest cargo flow in the market leaves no mark on freight",
        thesis=(
            "Brazilian soybean exports grew roughly sixfold since 1999 and the Panamax "
            "seasonal amplitude nearly tripled — but not in the harvest months, which sit "
            "at the year's own median in every sub-period. A freight seasonal is a "
            "positioning seasonal, not a demand one."
        ),
        disagreement=(
            "Inferred tension. The harvest is the obvious candidate for a freight "
            "seasonal, and it is how the flow is usually described. MEASURED ON 27 YEARS "
            "OF THE BALTIC PANAMAX INDEX: the trough is February and the peak October — "
            "both outside the Brazilian window. March through June are statistically "
            "indistinguishable from the annual level, while February and October are "
            "not. An anticipated flow is absorbed by repositioning before it reaches the "
            "rate."
        ),
        pivot=(
            "The amplitude tripled while the harvest months moved by 0.011 of the annual "
            "level — a difference that does not require knowing the counterfactual"
        ),
        mail_question=(
            "Brazilian exports went up roughly sixfold in 25 years and I cannot find any "
            "footprint in the Panamax seasonal. Is that because you pre-position the "
            "fleet months ahead, or because the BPI is too global to show a single-basin "
            "flow?"
        ),
        targets="Grain chartering desks (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra), Panamax owners and operators",
        data_gate=GATE_NONE,
        data_fallback=(
            "The P8 Santos-Qingdao route is missing 2023 and 2024 entirely in the export, "
            "which rules out a route-level seasonal. The BPI carries 27 complete years "
            "and the test runs on it instead — at the cost of basin dilution, which is "
            "stated as the limit of what the page can prove."
        ),
        status=STATUS_READY,
        dashboard_page="pages/13_D_Grain_Seasonal.py",
        chain_module="freight.chains.grain_seasonal",
        data_mode=DATA_REAL,
        n_tests=16,
    ),
    Project(
        id="freight_incidence",
        code="E",
        tier=TIER_FREIGHT,
        title="The buyer does not pay the freight, and the index says so",
        thesis=(
            "The 62% Fe index is quoted CFR China, so the freight is already inside it — "
            "which makes the incidence a regression coefficient rather than an opinion. "
            "At every horizon where the sample has power, full pass-through is rejected: "
            "the delivered price does not move with the freight it contains."
        ),
        disagreement=(
            "Inferred tension. Quoting a price CFR presents freight as something the "
            "buyer is charged for, and the delivered-price convention answers the "
            "incidence question implicitly. MEASURED ON DAILY IODEX 62 AND C5: the "
            "coefficient is -0.06 daily and -0.26 weekly, and one sits outside both "
            "intervals. The test is strong enough to matter — under full pass-through "
            "the correlation would be 0.18 against a 0.066 band, 2.7x the noise floor."
        ),
        pivot=(
            "Full pass-through rejected where the sample has power; zero not rejected — "
            "so the answer is bounded, not settled, and the bound is the deliverable"
        ),
        mail_question=(
            "The 62% CFR shows no freight pass-through at daily or weekly horizons and "
            "full pass-through is statistically rejected. Is freight a cost you absorb "
            "into the netback, or does it get renegotiated into the price on term "
            "business — and does that differ between your FOB and CFR books?"
        ),
        targets="Iron ore desks and marketing at Vale, Rio Tinto, BHP; Capesize freight desks at Glencore, Trafigura, Cargill",
        data_gate=GATE_NONE,
        data_fallback=(
            "The IODEX series starts in January 2023, leaving 14 quarterly observations — "
            "so the contract horizon, where incidence would most plausibly settle, cannot "
            "be tested and no verdict is drawn from it. C5 is a front-month FFA rather "
            "than the spot route, which attenuates toward the null found here: the bias "
            "runs against the conclusion, not for it."
        ),
        status=STATUS_READY,
        dashboard_page="pages/14_E_Freight_Incidence.py",
        chain_module="freight.chains.freight_incidence",
        data_mode=DATA_REAL,
        n_tests=15,
    ),
    Project(
        id="bunker_basis",
        code="F",
        tier=TIER_FREIGHT,
        title="You hedge bunkers with crude, and here is what you keep",
        thesis=(
            "Crude is the cheapest and deepest instrument bunkers actually get hedged "
            "with, which makes the hedge ratio a measurable number rather than a "
            "convention. It has fallen by a factor of three since 2016, and even at its "
            "best crude explained barely a fifth of the daily bunker variance."
        ),
        disagreement=(
            "Inferred tension. Hedging bunkers with crude is standard desk practice, "
            "and the ratio used is often a inherited convention rather than a "
            "re-estimated one. MEASURED ON DAILY VLSFO SINGAPORE AND BRENT: the hedge "
            "ratio was 0.79 in 2016-2017 and 0.23 in 2024-2025 — a position sized on "
            "the earlier window is three times too large today, and crude never "
            "explained more than 21% of the daily variance even at its peak."
        ),
        pivot=(
            "The hedge ratio's own drift across eight two-year windows, and the ceiling "
            "on variance explained even in the best of them"
        ),
        mail_question=(
            "The crude hedge ratio on VLSFO Singapore has fallen from about 0.8 in "
            "2016-2017 to about 0.2 now, and even in the best window crude only "
            "explained roughly half the daily variance it was hedging. How do you size "
            "a bunker hedge today — still crude as the primary instrument, or has the "
            "desk moved toward gasoil, a blended proxy, or something else?"
        ),
        targets=(
            "Bunker purchasing and fuel risk desks (Cargill Fuel & Freight Risk, "
            "trading-house bunker desks at Vitol, Glencore, Trafigura; ship-operator "
            "bunker procurement)"
        ),
        data_gate=GATE_NONE,
        data_fallback=(
            "The series is labelled VLSFO back to 2009, nine years before the 0.5% "
            "sulphur grade it names existed as a bunker fuel. What the pre-2020 window "
            "actually tracks is not established here, and the drift measured across it "
            "is reported as a property of the quoted series rather than a claim about "
            "the fuel itself."
        ),
        status=STATUS_READY,
        dashboard_page="pages/15_F_Bunker_Basis.py",
        chain_module="freight.chains.bunker_basis",
        data_mode=DATA_REAL,
        n_tests=14,
    ),
    Project(
        id="marginal_ship",
        code="G",
        tier=TIER_FREIGHT,
        title="How much less efficient could a ship be, and still cover its fuel bill here",
        thesis=(
            "Solved exactly from the real P8 route rate and the real VLSFO price: the "
            "fuel-consumption multiplier at which a reference panamax's own fuel bill "
            "would consume all its freight revenue never approached 1 in nearly five "
            "years — not even during the sharpest VLSFO spike on record — and the "
            "margin's day-to-day variance traces more to the bunker price than to the "
            "freight rate itself."
        ),
        disagreement=(
            "Inferred tension. A strong freight market is usually read as the thing "
            "buying a chartering desk fuel-cost headroom. MEASURED ON THE REAL P8 ROUTE, "
            "2021-2022 AND 2025-2026: the breakeven multiplier ranged 1.46 to 3.86 and "
            "never approached 1, and only 38% of its variance traces to the freight "
            "rate — 62% traces to the bunker price, which the freight market does not "
            "set and does not track closely."
        ),
        pivot=(
            "The margin's tightest point on record lands inside the same VLSFO spike "
            "documented on project F's page, and even there stayed above 1.4"
        ),
        mail_question=(
            "On the P8 route since late 2021, a panamax could have burned 1.5 to nearly "
            "4 times its reference consumption and still covered its fuel bill from "
            "freight revenue alone — and that margin moves more with the bunker price "
            "than with the freight rate itself. Does your desk actually track a "
            "fuel-cost headroom like this when assessing which tonnage to fix, or does "
            "opex and capital cost dominate the decision so completely that fuel "
            "efficiency alone rarely binds?"
        ),
        targets="Grain and freight chartering desks (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra)",
        data_gate=GATE_NONE,
        data_fallback=(
            "The P8 route rate is missing 2023 and 2024 entirely in this export, the "
            "same gap documented in project D — this test covers the 2021-2022 "
            "boom-to-slump transition and the 2025-2026 window, not a continuous "
            "history."
        ),
        status=STATUS_READY,
        dashboard_page="pages/16_G_Marginal_Ship.py",
        chain_module="freight.chains.marginal_ship",
        data_mode=DATA_REAL,
        n_tests=11,
    ),
    Project(
        id="index_basis",
        code="H",
        tier=TIER_FREIGHT,
        title="The index tracks the route worst on the days that matter least",
        thesis=(
            "A route-specific hedger fears a benchmark index decouples exactly when "
            "the route moves hardest. Measured on the real BPI and the real P8 route: "
            "the index explains MORE of the route's variance on its biggest-move days "
            "(R²=10.4%, significant) than on calm ones (R²=0.3%, not significant) — but "
            "the unexplained dollars grow just as fast as the move itself, so the "
            "better statistical fit buys no more absolute protection."
        ),
        disagreement=(
            "Inferred tension. A route-specific hedge using a global index is assumed "
            "to work worst exactly when it is needed most. MEASURED ON 624 OVERLAPPING "
            "DAYS OF THE REAL BPI AND P8 ROUTE RATE: split by the size of the route's "
            "own move, R² rises monotonically from 0.3% (calm) to 10.4% (top decile) — "
            "the opposite of the naive fear. The unexplained residual still grows 40x "
            "over the same split, almost exactly matching the 42x growth in the raw "
            "move, so the improved fit is a statistical artifact of scale, not better "
            "dollar protection."
        ),
        pivot=(
            "R² rises with the size of the route's move; the residual in USD/t grows "
            "in almost the same proportion as the move itself — the correction that "
            "keeps the first result from being read as good news"
        ),
        mail_question=(
            "On the days the P8 route moves hardest, the BPI index actually tracks it "
            "more closely than on a quiet day — but the dollars left unhedged grow "
            "just as fast as the move itself, so the better statistical fit does not "
            "translate into better protection. On your largest route-specific moves, "
            "is there usually an identifiable driver — a large single fixture, port "
            "congestion, an itinerary shift — that a desk would see operationally but "
            "that never shows up in the index?"
        ),
        targets=(
            "Freight derivatives and FFA risk desks (Freight Investor Services, SSY "
            "Futures, Clarksons Securities) — the paper-hedging side of freight risk"
        ),
        data_gate=GATE_NONE,
        data_fallback=(
            "The P8 route rate is missing 2023 and 2024 entirely, the same gap "
            "documented in project D — this test covers the 2021-2022 and 2025-2026 "
            "windows only. A finer slice than the top decile (the 90th-95th percentile "
            "of moves, 32 observations) shows a striking R² and is deliberately not "
            "reported as a verdict."
        ),
        status=STATUS_READY,
        dashboard_page="pages/17_H_Index_Basis.py",
        chain_module="freight.chains.index_basis",
        data_mode=DATA_REAL,
        n_tests=8,
    ),
    Project(
        id="cii_ballast",
        code="I",
        tier=TIER_FREIGHT,
        title="The ballast leg the regulator counts the same as a laden one",
        thesis=(
            "CII's attained metric divides emissions by capacity and distance sailed — "
            "neither term asks whether the ship was carrying cargo. Slowing only the "
            "ballast leg buys a 31% better rating with zero additional cargo "
            "transported, and that same slowdown costs 6% of the voyage's real annual "
            "net contribution at current freight and bunker prices."
        ),
        disagreement=(
            "Inferred tension. A rating designed to measure carbon efficiency is "
            "assumed to track transport efficiency. COMPUTED FROM THE VOYAGE MODEL ON "
            "THE REAL P8 SANTOS-QINGDAO ROUTE: slowing the ballast leg from 13 to 8 "
            "knots, with the laden leg and every tonne of cargo on it unchanged, "
            "improves attained AER by 31% — and loading more or less cargo changes AER "
            "by exactly zero, since deadweight is nameplate capacity, not cargo "
            "carried."
        ),
        pivot=(
            "The same 31% rating gain costs 6% of annual net contribution at real "
            "prices — the rating can be gamed for free, the P&L notices"
        ),
        mail_question=(
            "Slowing only the ballast leg on this route buys a 31% better attained AER "
            "with zero additional cargo moved — but at real freight and bunker prices "
            "it costs about 6% of the voyage's annual net contribution. Does a bad CII "
            "rating actually show up in your fixture terms or charter-party clauses in "
            "a way that would justify eating that cost, or is the rating something the "
            "fleet reports and the chartering decision otherwise ignores?"
        ),
        targets=(
            "Dry bulk shipowners and operators (Oldendorff, Star Bulk, Golden Ocean, "
            "Pacific Basin) — the rating sits with the owner, not the charterer"
        ),
        data_gate=GATE_NONE,
        data_fallback=(
            "The official IMO rating-boundary table (reference lines and reduction "
            "factors by ship type and size) is deliberately not reproduced — citing "
            "exact regulatory thresholds without full certainty in them would be worse "
            "than not answering. Every result is an attained-AER percentage change, "
            "which needs none of that table."
        ),
        status=STATUS_READY,
        dashboard_page="pages/18_I_CII_Ballast.py",
        chain_module="freight.chains.cii_ballast",
        data_mode=DATA_REAL,
        n_tests=12,
    ),
]


PROJECTS: list[Project] = FREIGHT_PROJECTS + AGRI_PROJECTS


def by_tier() -> dict[str, list[Project]]:
    """Group projects by tier, preserving the order tiers first appear in PROJECTS."""
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
    "AGRI_PROJECTS",
    "DATA_HYBRID",
    "DATA_REAL",
    "DATA_SYNTHETIC",
    "FREIGHT_PROJECTS",
    "GATE_HIGH",
    "GATE_MEDIUM",
    "GATE_NONE",
    "PROJECTS",
    "Project",
    "STATUS_PLANNED",
    "STATUS_READY",
    "TIER_FREIGHT",
    "by_tier",
    "get",
    "ready_projects",
    "total_tests",
]
