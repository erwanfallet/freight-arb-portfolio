"""Single source of truth for the portfolio — every arbitrage project this repo covers
or plans to cover, across every commodity vertical (dry bulk, tankers, agri, LNG, ...).

`app/Home.py` renders this list and nothing else — add a new `Project` here and it shows
up on the platform automatically, grouped by sector, whether it's built yet or not.

This is deliberately not a database or a config file: it's Python, so a project's
metadata sits next to the type that enforces its shape, and nothing here can drift out
of sync with what the dashboard actually renders.
"""
from __future__ import annotations

from dataclasses import dataclass

STATUS_READY = "ready"        # engine + dashboard exist, tested, waiting on real data
STATUS_PLANNED = "planned"    # sector identified, nothing built yet


@dataclass(frozen=True)
class Project:
    id: str                      # "iron_ore" — matches chains/<id>.py and data/raw/<id>.csv
    letter: str                  # "A" — short handle used in conversation and docs
    sector: str                  # groups projects on the platform page
    title: str                   # "Minerai de fer"
    thesis: str                  # one-line claim, bold on the card
    mechanism: str                # 2-4 sentences, the trick that makes it non-obvious
    flow_validation: str | None   # which free official physical-flow series checks it
    status: str                   # STATUS_READY | STATUS_PLANNED
    status_detail: str            # shown under the status badge
    dashboard_page: str | None    # path relative to app/, None if not built
    chain_module: str | None      # "freight.chains.ironore", None if not built
    n_tests: int | None           # golden tests for this project alone


PROJECTS: list[Project] = [
    Project(
        id="iron_ore",
        letter="A",
        sector="Vrac sec — minerais",
        title="Minerai de fer",
        thesis="Le premium 65-62 % Fe est en partie un spread de fret Capesize.",
        mechanism=(
            "Les deux indices sont cotés CFR Chine, donc le différentiel de fret "
            "C3 − C5 est déjà à l'intérieur du premium haute teneur. On corrige "
            "l'humidité (le fret se paie sur tonne humide, l'indice se cote sur tonne "
            "sèche) puis on décompose : ce qui reste est un résidu qu'on ne rebaptise pas."
        ),
        flow_validation="Importations chinoises de minerai par origine (douanes GACC, mensuel)",
        status=STATUS_READY,
        status_detail="Moteur + dashboard prêts, 16 golden tests, en attente des 4 séries",
        dashboard_page="pages/1_Iron_Ore_Premium.py",
        chain_module="freight.chains.ironore",
        n_tests=16,
    ),
    Project(
        id="coal",
        letter="B",
        sector="Vrac sec — énergie",
        title="Charbon Atlantique",
        thesis="L'arb API2 − API4 a perdu sa contrainte contraignante en 2022.",
        mechanism=(
            "Depuis 2022 le charbon sud-africain part vers l'Inde plutôt que vers "
            "l'Europe : la cargaison marginale de Richards Bay n'est plus celle qui "
            "fixe Rotterdam. Sans contrôle par le TTF (choc gazier européen, même "
            "année), on attribue à l'Inde un décrochage qui vient en partie du gaz — "
            "sur le jeu de test, la pente passe de 0,71 à 0,18 selon qu'on contrôle ou non."
        ),
        flow_validation="Importations Eurostat (UE) + douanes indiennes de charbon sud-africain (mensuel)",
        status=STATUS_READY,
        status_detail="Moteur + dashboard prêts, 22 golden tests, en attente des séries",
        dashboard_page="pages/2_Coal_Atlantic_Arb.py",
        chain_module="freight.chains.coal",
        n_tests=22,
    ),
    Project(
        id="products",
        letter="C",
        sector="Tankers — produits raffinés",
        title="Distillat transatlantique",
        thesis="Le volume n'est pas la masse, et les points Worldscale ne sont pas un coût.",
        mechanism=(
            "La jambe US se cote au gallon, l'européenne à la tonne : une densité "
            "traitée comme une constante pèse souvent plus que l'arb lui-même. Et le "
            "fret tanker se cote en points Worldscale, convertis via un flat rate "
            "réinitialisé chaque 1er janvier — sur le jeu de test, ce seul reset pèse "
            "dix fois le vrai signal de marché."
        ),
        flow_validation="Exportations américaines de distillat par destination (EIA, mensuel)",
        status=STATUS_READY,
        status_detail="Moteur + dashboard prêts, 20 golden tests, variantes C-2/C-3 codées",
        dashboard_page="pages/3_Products_Transatlantic.py",
        chain_module="freight.chains.products",
        n_tests=20,
    ),
    Project(
        id="agri",
        letter="D",
        sector="Vrac sec — agriculture",
        title="À définir",
        thesis="Piste ouverte, thèse et piège d'unité pas encore choisis.",
        mechanism=(
            "Candidats naturels dans la même famille que A et B : un arb origine-origine "
            "sur soja ou maïs (Santos/US Gulf, route Panamax P6), avec un piège plausible "
            "sur l'humidité de récolte ou la conversion boisseau/tonne."
        ),
        flow_validation=None,
        status=STATUS_PLANNED,
        status_detail="Secteur identifié, rien codé — voir docs/NOUVELLE_CHAINE.md",
        dashboard_page=None,
        chain_module=None,
        n_tests=None,
    ),
    Project(
        id="lng",
        letter="E",
        sector="Gaz",
        title="À définir",
        thesis="Piste ouverte, thèse et piège d'unité pas encore choisis.",
        mechanism=(
            "Le LNG ne se mesure jamais en DWT (limité par le volume, pas la masse, "
            "~0,45 t/m³) — un piège d'unité de plus dans la même famille que le "
            "gallon/tonne du projet C. Arb Atlantique-Pacifique (US Gulf vs Asie) le "
            "candidat le plus évident, sur les routes BLNG."
        ),
        flow_validation=None,
        status=STATUS_PLANNED,
        status_detail="Secteur identifié, rien codé — voir docs/NOUVELLE_CHAINE.md",
        dashboard_page=None,
        chain_module=None,
        n_tests=None,
    ),
]


def by_sector() -> dict[str, list[Project]]:
    """Group projects by sector, preserving the order sectors first appear in PROJECTS."""
    grouped: dict[str, list[Project]] = {}
    for project in PROJECTS:
        grouped.setdefault(project.sector, []).append(project)
    return grouped


def ready_projects() -> list[Project]:
    return [p for p in PROJECTS if p.status == STATUS_READY]


def total_tests() -> int:
    return sum(p.n_tests or 0 for p in PROJECTS)
