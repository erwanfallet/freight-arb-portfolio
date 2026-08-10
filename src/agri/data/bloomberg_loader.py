"""Lecteur de l'export Bloomberg reel de l'utilisateur.

Chaque onglet Bloomberg est un bloc Date/Valeur (parfois plusieurs par onglet, cote a
cote). Ce module localise le bloc, verifie l'unite attendue, et rend une pd.Series
propre — sans jamais deviner un ticker : la correspondance onglet -> serie est explicite
dans `SERIES_SPECS`, construite a partir de l'inspection reelle du fichier (pas d'un nom
suppose).

Aucun forward-fill ici : c'est une lecture, pas une modelisation. Un trou reste un trou,
meme regle qu'`ingest/contract.py` du portefeuille fret.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import openpyxl
import pandas as pd

DEFAULT_PATH = Path("/Users/erwanfallet/Desktop/Data Bloomberg.xlsx")


class BloombergLoaderError(ValueError):
    """Onglet ou bloc introuvable, ou lecture incoherente."""


@dataclass(frozen=True)
class SeriesSpec:
    """Un bloc Date/Valeur identifie dans l'export, avec son unite declaree.

    `block_index` distingue plusieurs series dans un meme onglet (ex. l'onglet "JET"
    contient le swap M1 en bloc 0 et le prompt spot en bloc 1).

    `scale` corrige un facteur de cotation connu et verifie a la main — jamais devine.
    Les futures grains CBOT (soja/mais/ble) sont cotes en **cents par boisseau** dans cet
    export (ex. 1156.50 = 11,565 USD/bu, verifie contre le niveau de marche reel) ; sans
    diviser par 100, toute formule attendant un prix en USD/bu (board_crush_usd_bu,
    financing_cost_usd_t...) sort un resultat cent fois trop grand — le genre d'erreur qui
    ne plante jamais.

    `valid_from` ecarte une periode dont l'unite economique a change dans le temps.
    USDBRL avant juillet 1994 cote le cruzeiro/cruzeiro real d'avant le Plano Real
    (hyperinflation bresilienne, valeurs ~0,0004 en 1992) : une monnaie differente, pas
    une valeur aberrante a corriger — on l'exclut plutot que de la reechelonner.
    """

    sheet: str
    block_index: int
    unit: str
    real_id: str
    note: str = ""
    scale: float = 1.0
    valid_from: str | None = None
    valid_to: str | None = None
    min_valid: float | None = None


# Construit a partir de l'inspection reelle du fichier (dump exhaustif des onglets),
# jamais d'un nom de ticker suppose. Voir DEMANDE_DONNEES.md et data_coverage.json pour
# la tracabilite complete.
SERIES_SPECS: dict[str, SeriesSpec] = {
    "jet_swap_m1": SeriesSpec(
        sheet="JET", block_index=0, unit="c/gal MAIS CONVENTION INSTABLE",
        real_id="US Gulf Coast Jet Fuel 54 Grade Swap M1",
        note=(
            "DEFAUT DE DONNEE CONFIRME : la serie alterne entre USD/gal (valeurs ~3-4) et "
            "c/gal (valeurs ~300-400) a plusieurs reprises dans son historique (sauts "
            "detectes en 2019-02, 2019-03, 2020-08/09/11, 2023-07, 2026-05). Ne pas "
            "utiliser telle quelle pour un calcul de spread — soit normaliser via "
            "`normalize_unit_jumps`, soit preferer 'jet_spot' qui est propre."
        ),
    ),
    "jet_spot": SeriesSpec(
        sheet="JET", block_index=1, unit="c/gal",
        real_id="US Gulf Coast Jet Fuel 54 Prompt Spot",
        note="Verifie propre : aucun saut de convention detecte sur 8542 observations (1990-2026).",
    ),
    "ulsd": SeriesSpec(
        sheet="HO1 Comdty", block_index=0, unit="c/gal",
        real_id="NYMEX ULSD / Heating Oil front month",
        note=(
            "COTE EN CENTS PAR GALLON — donc en VOLUME. La jambe europeenne (ice_gasoil) "
            "est en USD par TONNE, donc en MASSE. Le facteur entre les deux est une "
            "densite, pas une constante : voir `freight.chains.products`, ou l'incertitude "
            "sur cette densite vaut 91 % de la variabilite de l'arb qu'elle sert a calculer."
        ),
    ),
    "ice_gasoil": SeriesSpec(
        sheet="QS1 Comdty", block_index=0, unit="USD/t",
        real_id="ICE Gasoil (Low Sulphur) front month",
        note=(
            "Jambe europeenne de l'arb distillat transatlantique, cotee a la TONNE. Sert "
            "aussi de proxy MGO dans `agri.chains.freight_cf` faute d'assessment bunker "
            "port par port — deux usages, une seule serie, et les deux sont documentes."
        ),
    ),
    "ttf": SeriesSpec(
        sheet="TZT1 Comdty", block_index=0, unit="EUR/MWh",
        real_id="ICE Endex Dutch TTF Natural Gas Futures front month",
    ),
    "henry_hub": SeriesSpec(
        sheet="NG1 Comdty", block_index=0, unit="USD/mmBtu",
        real_id="NYMEX Henry Hub Natural Gas front month",
    ),
    "eurusd": SeriesSpec(
        sheet="EUR USD", block_index=0, unit="EUR par USD (verifier le sens)",
        real_id="EURUSD",
    ),
    "brent": SeriesSpec(
        sheet="CO1 Comdty", block_index=0, unit="USD/bbl", real_id="ICE Brent front month",
    ),
    "wti": SeriesSpec(
        sheet="CL1 Comdty", block_index=0, unit="USD/bbl", real_id="NYMEX WTI front month",
    ),
    "dubai": SeriesSpec(
        sheet="Middle East Dubai Crude FOB Fat", block_index=0, unit="USD/bbl",
        real_id="Middle East Dubai Crude FOB Fateh Cargo Spot",
    ),
    "sofr": SeriesSpec(
        sheet="SOFRRATE Index", block_index=0, unit="fraction décimale (0,0433 = 4,33 %)",
        real_id="SOFR", scale=0.01,
        note=(
            "DEFAUT TROUVE ET CORRIGE : Bloomberg cote le SOFR en POURCENTS (5,40 au pic "
            "du resserrement 2023), pas en fraction decimale. Additionne tel quel a un "
            "spread deja exprime en decimal (250 bps -> 0,025), il produisait un taux "
            "tout compris de 243 % et gonflait le cout de financement d'un facteur ~100. "
            "`scale=0.01` rend le contrat du loader uniforme : **tout taux sort en "
            "fraction decimale**, prete a multiplier un montant."
        ),
    ),
    # --- Softs ICE : cacao, cafe, sucre. Tous verifies sans saut de convention. ---
    "cocoa_ny": SeriesSpec(
        sheet="CC1 Comdty", block_index=0, unit="USD/t", real_id="ICE Cocoa New York front month",
        note="Max observe 12 565 USD/t — coincide avec le pic reel d'avril 2024, confirme que la serie n'est pas re-echelonnee.",
    ),
    "cocoa_london": SeriesSpec(
        sheet="QC1 Comdty", block_index=0, unit="GBP/t", real_id="ICE Cocoa London front month",
    ),
    "coffee_arabica": SeriesSpec(
        sheet="KC1 Comdty", block_index=0, unit="c/lb", real_id="ICE Coffee Arabica (C) front month",
    ),
    "coffee_robusta": SeriesSpec(
        sheet="DF1 Comdty", block_index=0, unit="USD/t", real_id="ICE Robusta Coffee front month",
        note=(
            "Identite deduite de la position dans la sequence CC1/QC1/KC1/DF1/SB1/QW1, "
            "PAS confirmee via DES. Coherence de niveau : max 5817 USD/t proche du record "
            "reel robusta 2024 (~5800 USD/t) — plausible mais a verifier avant usage engageant."
        ),
    ),
    "sugar_no11": SeriesSpec(
        sheet="SB1 Comdty", block_index=0, unit="c/lb, base 96 pol", real_id="ICE Sugar No.11 front month",
    ),
    "sugar_no5": SeriesSpec(
        sheet="QW1 Comdty", block_index=0, unit="USD/t", real_id="ICE Sugar No.5 (white) front month",
    ),
    # --- CBOT grains : COTES EN CENTS/BOISSEAU dans cet export, scale=0.01 (voir SeriesSpec.scale) ---
    "cbot_soybean": SeriesSpec(
        sheet="S 1 Comdty", block_index=0, unit="USD/bu", real_id="CBOT Soybean front month",
        scale=0.01, note="Brut en c/bu (ex. 1156.50) ; scale=0.01 -> 11,565 USD/bu.",
    ),
    "cbot_corn": SeriesSpec(
        sheet="C 1 Comdty", block_index=0, unit="USD/bu", real_id="CBOT Corn front month",
        scale=0.01, note="Brut en c/bu ; scale=0.01.",
    ),
    "cbot_wheat": SeriesSpec(
        sheet="W 1 Comdty", block_index=0, unit="USD/bu", real_id="CBOT Wheat (SRW) front month",
        scale=0.01, note="Brut en c/bu ; scale=0.01.",
    ),
    "cbot_soymeal": SeriesSpec(
        sheet="SM1 Comdty", block_index=0, unit="USD/short ton", real_id="CBOT Soybean Meal front month",
        note="Deja en USD/short ton natif — pas de scale (niveaux 120-550 coherents sans conversion).",
    ),
    "cbot_soyoil": SeriesSpec(
        sheet="BO1 Comdty", block_index=0, unit="c/lb", real_id="CBOT Soybean Oil front month",
        note="Deja en c/lb natif — pas de scale (niveaux 14-90 coherents sans conversion).",
    ),
    "palm_oil_myr": SeriesSpec(
        sheet="KO1 Comdty", block_index=0, unit="MYR/t", real_id="Bursa Malaysia Crude Palm Oil front month",
        note=(
            "COTEE EN RINGGITS MALAIS, pas en dollars. Niveaux 657-8163 sur 1995-2026, "
            "coherents avec du MYR/tonne (la palme en USD/t vaut 400-1500 sur la meme "
            "periode). L'export ne contient AUCUNE serie USDMYR : tout spread palme-soja "
            "melangerait donc deux devises, ce qui est exactement l'erreur que ce "
            "portefeuille traque. Voir `oil_substitution` : la seule fenetre exploitable "
            "est celle de la parite fixe 1998-2005, ou le change est une constante connue."
        ),
    ),
    # --- DCE Chine ---
    "dce_soymeal": SeriesSpec(
        sheet="AE1 Comdty", block_index=0, unit="CNY/t", real_id="DCE Soybean Meal front month",
    ),
    "dce_soyoil": SeriesSpec(
        sheet="SH1 Comdty", block_index=0, unit="CNY/t", real_id="DCE Soybean Oil front month",
    ),
    # --- Change ---
    "usdbrl": SeriesSpec(
        sheet="USDBRL Curncy", block_index=0, unit="BRL par USD", real_id="USDBRL",
        valid_from="1994-07-01",
        note="Avant juillet 1994 : cruzeiro/cruzeiro real pre-Plano Real (hyperinflation) — monnaie differente, exclue plutot que reechelonnee.",
    ),
    "usdcny": SeriesSpec(
        sheet="USDCNY Curncy", block_index=0, unit="CNY par USD", real_id="USDCNY",
    ),
    # --- Fret sec : la route P8 est coupee en DEUX REGIMES D'UNITE ---
    # C'est le meme defaut que 'jet_swap_m1', mais il tombe ici sur le sujet meme de la
    # page T1-1 : la confusion entre un TCE en USD/JOUR et un taux de voyage en USD/TONNE.
    "p8_route_usd_t": SeriesSpec(
        sheet="P8 FFA 66kt Santos to Qingdao M", block_index=0, unit="USD/t",
        real_id="P8 FFA 66kt Santos->Qingdao M0, segment taux de voyage",
        valid_from="2021-11-18", min_valid=5.0,
        note=(
            "SEGMENT USD/TONNE UNIQUEMENT (2021-11-18 -> 2026-08-08, 774 prints, "
            "35,8-84,5 USD/t). Avant cette date la meme cellule cote un TCE en USD/JOUR "
            "(2021-07-01 -> 2021-10-30, 103 prints, 24 500-38 000). Les deux segments "
            "n'ont AUCUNE date commune et sont separes par 19 jours de trou : le facteur "
            "de conversion ne peut donc pas etre calibre sur la jonction, le marche ayant "
            "bouge entre les deux. Un print nul au 2022-04-30 est ecarte par min_valid."
        ),
    ),
    "p8_route_tce_2021": SeriesSpec(
        sheet="P8 FFA 66kt Santos to Qingdao M", block_index=0, unit="USD/jour (TCE)",
        real_id="P8 FFA 66kt Santos->Qingdao M0, segment TCE",
        valid_to="2021-10-30",
        note=(
            "Le segment amont de la MEME cellule, en USD/jour. Conserve separement parce "
            "qu'un TCE est l'entree naturelle du modele de voyage, alors qu'un taux en "
            "USD/t en est la sortie — les melanger est exactement l'erreur que T1-1 mesure."
        ),
    ),
    "bpi": SeriesSpec(
        sheet="BPI Baltic Exchange Panamax Ind", block_index=0, unit="points d'indice",
        real_id="Baltic Panamax Index (moyenne TC P1A/P2A/P3A/P4)",
        note=(
            "POINTS D'INDICE, pas des USD : le BPI est une moyenne ponderee de quatre "
            "routes timecharter, publiee en points. Il ne se convertit pas en USD/t sans "
            "passer par le 5TC en USD/jour, absent de l'export. Utilise ici comme "
            "contexte de cycle, jamais comme terme d'un calcul de cout."
        ),
    ),
    "vlsfo_singapore": SeriesSpec(
        sheet="GX Very Low Sulphur Fuel Oil Bu", block_index=0, unit="USD/t",
        real_id="VLSFO Bunker Singapore (SGSIN) 1800 Prompt",
    ),
}


def _find_date_blocks(rows: list[tuple]) -> list[tuple[int, int]]:
    """Toutes les positions (ligne, colonne) ou une cellule 'Date' apparait, dans les 12
    premieres lignes — un onglet peut porter plusieurs series cote a cote."""
    blocks = []
    for i, row in enumerate(rows[:12]):
        for j, cell in enumerate(row):
            if isinstance(cell, str) and cell.strip().lower() in ("date", "dates"):
                blocks.append((i, j))
    return blocks


def load_raw_series(
    key: str, *, path: Path = DEFAULT_PATH, dropna: bool = True
) -> pd.Series:
    """Charge une serie par sa cle dans `SERIES_SPECS`. Aucun ffill, aucune interpolation.

    `dropna=True` (par defaut) retire les dates sans valeur — frequent en tete de fichier
    Bloomberg (ex. une ligne datee sans print encore arrive). Passer False pour voir les
    trous bruts.
    """
    if key not in SERIES_SPECS:
        raise BloombergLoaderError(
            f"cle inconnue : {key!r}. Cles disponibles : {sorted(SERIES_SPECS)}"
        )
    spec = SERIES_SPECS[key]
    if not path.exists():
        raise BloombergLoaderError(f"fichier introuvable : {path}")

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if spec.sheet not in wb.sheetnames:
        raise BloombergLoaderError(
            f"onglet {spec.sheet!r} absent du fichier — l'export a-t-il change depuis "
            f"data_coverage.json ? Onglets disponibles : {wb.sheetnames}"
        )

    ws = wb[spec.sheet]
    rows = list(ws.iter_rows(values_only=True))
    blocks = _find_date_blocks(rows)
    if spec.block_index >= len(blocks):
        raise BloombergLoaderError(
            f"bloc {spec.block_index} demande mais l'onglet {spec.sheet!r} n'en contient "
            f"que {len(blocks)}"
        )
    header_row, col = blocks[spec.block_index]

    dates: list[date] = []
    values: list[float] = []
    for row in rows[header_row + 1 :]:
        if col >= len(row):
            continue
        d = row[col]
        v = row[col + 1] if col + 1 < len(row) else None
        if isinstance(d, (datetime, date)):
            dates.append(pd.Timestamp(d))
            values.append(v)

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=key).sort_index()
    if spec.scale != 1.0:
        series = series * spec.scale
    if spec.valid_from is not None:
        series = series.loc[pd.Timestamp(spec.valid_from) :]
    if spec.valid_to is not None:
        series = series.loc[: pd.Timestamp(spec.valid_to)]
    if spec.min_valid is not None:
        # Un print physiquement impossible (un fret nul, un prix negatif) est ecarte
        # plutot que laisse circuler : il ne represente pas un etat du marche, et il
        # casse silencieusement toute division en aval.
        series = series[series > spec.min_valid]
    if dropna:
        series = series.dropna()
    return series


def load(key: str, *, path: Path = DEFAULT_PATH) -> pd.Series:
    """Raccourci standard : `load('ttf')`, `load('henry_hub')`, etc."""
    return load_raw_series(key, path=path)


def detect_unit_jumps(series: pd.Series, *, factor_threshold: float = 20.0) -> pd.Series:
    """Detecte les sauts de convention d'unite : jour a jour, un facteur > 20x ou < 0.05x.

    Un vrai mouvement de marche ne multiplie jamais un prix par 100 du jour au lendemain ;
    un changement de convention d'affichage (USD vs cents) le fait. C'est le controle qui
    a permis de trouver le defaut de 'jet_swap_m1' : la serie alterne entre les deux
    conventions plusieurs fois dans son historique. Renvoie les dates ou le saut se
    produit, avec le ratio observe — vide si la serie est propre.
    """
    ratio = series / series.shift(1)
    return ratio[(ratio > factor_threshold) | (ratio < 1.0 / factor_threshold)]


def available_range(key: str, *, path: Path = DEFAULT_PATH) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Premiere date, derniere date, nombre d'observations non nulles — pour le panneau
    de diagnostics d'une page (staleness, profondeur d'historique)."""
    series = load_raw_series(key, path=path)
    if series.empty:
        raise BloombergLoaderError(f"serie {key!r} vide apres lecture")
    return series.index.min(), series.index.max(), len(series)
