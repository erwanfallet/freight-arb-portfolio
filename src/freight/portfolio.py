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

TIER_FREIGHT = "Dry bulk and refined products — freight as the deciding term"


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
        title="The coal-to-gas switching price is not a property of the fuels",
        thesis=(
            "A European generator does not choose between coal and gas on fuel price. It "
            "chooses on fuel plus carbon, per MWh of electricity, at plant efficiencies "
            "that differ by a factor of nearly two. Three units and two currencies stack "
            "up before any comparison is possible."
        ),
        disagreement=(
            "Inferred tension. The switching price is widely quoted as a single number, as "
            "if it were a property of the two fuels. It is not: it is a property of the "
            "two plants' efficiencies and of the carbon price, and it moves when neither "
            "fuel moves."
        ),
        pivot=(
            "The carbon price at which the switch flips at today's fuel prices — one "
            "number a power trader confirms or denies immediately"
        ),
        mail_question=(
            "Which pair of efficiencies does your switching calculation actually use, and "
            "how often is it re-estimated against the plants that are really at the margin?"
        ),
        targets="European power and gas desks (Uniper, RWE, EDF Trading, Vitol, Glencore coal)",
        data_gate=GATE_NONE,
        data_fallback=(
            "API4 Richards Bay is absent from the export, so the original API2 − API4 arb "
            "is not computable. API2, TTF, EUA and EURUSD are all real and support a better "
            "question."
        ),
        status=STATUS_READY,
        dashboard_page="pages/2_Coal_Atlantic_Arb.py",
        chain_module="freight.chains.coal",
        data_mode=DATA_REAL,
        n_tests=30,
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
