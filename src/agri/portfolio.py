"""Registre unique du portefeuille agri — chaque projet, son tier, son état.

`app/Home.py` rend cette liste et rien d'autre : ajouter un `Project` ici le fait
apparaître sur la plateforme, groupé par tier, qu'il soit construit ou non.

Volontairement du Python et pas un fichier de configuration : les métadonnées d'un projet
sont à côté du type qui contraint leur forme, et rien ici ne peut dériver de ce que le
dashboard affiche réellement.
"""
from __future__ import annotations

from dataclasses import dataclass

STATUS_READY = "ready"        # moteur + dashboard existent, testés
STATUS_PLANNED = "planned"    # cadré, rien codé

DATA_SYNTHETIC = "synthétique"   # jeu fabriqué pour imposer le phénomène, en attente de données réelles
DATA_REAL = "réel"               # tourne sur l'export Bloomberg de l'utilisateur
DATA_HYBRID = "hybride"          # jambes principales réelles, un terme minoritaire reste paramétré (documenté dans le moteur)

TIER_1 = "T1 — désaccords sourcés"
TIER_2 = "T2 — tensions structurelles inférées"
TIER_3 = "T3 — désaccords ouverts en août 2026"
TIER_REAL = "Données Bloomberg réelles — oil & LNG"

# Niveau de risque sur l'accès aux données, tel qu'établi par les gates de la spec.
GATE_NONE = "aucun"           # séries gratuites et publiques
GATE_MEDIUM = "moyen"         # une série payante, repli codé
GATE_HIGH = "élevé"           # à tester avant tout code


@dataclass(frozen=True)
class Project:
    id: str                       # "freight_cf" — correspond à chains/<id>.py
    code: str                     # "T1-1" — la référence de la spec
    tier: str
    title: str
    thesis: str                   # l'affirmation en une ligne, en gras sur la carte
    disagreement: str             # d'où vient le désaccord, et entre qui
    pivot: str                    # le point de bascule — le livrable de la page
    mail_question: str            # la question que seul un insider peut trancher
    targets: str                  # le vivier de cibles
    data_gate: str
    data_fallback: str | None     # ce qu'on fait si le gate échoue
    status: str
    dashboard_page: str | None
    chain_module: str | None
    n_tests: int | None
    data_mode: str = DATA_SYNTHETIC   # DATA_REAL pour les projets branchés sur l'export Bloomberg


PROJECTS: list[Project] = [
    # ======================================================================
    # TIER 1 — désaccords sourcés, citables
    # ======================================================================
    Project(
        id="freight_cf",
        code="T1-1",
        tier=TIER_1,
        title="Le fret dans le calcul C&F",
        thesis="Sur la bande frontière, le fret ne bruite pas l'arb : il le détermine.",
        disagreement=(
            "Interview Mat Halsall (Commodity Conversations, 25 nov. 2024) : chez Louis "
            "Dreyfus, disputes récurrentes entre desks de trading et département fret, "
            "les traders contestant le taux sans en connaître les composantes. Le desk "
            "dit « votre taux n'est pas le marché » ; le fret répond « vous regardez un "
            "index, pas un coût ». TRANCHÉ SUR DONNÉES RÉELLES : lire le taux P8 publié "
            "sans facturer le ballast implique un TCE supérieur au pic du boom vraquier "
            "2021 sur 99 % des cinq dernières années — arithmétiquement intenable. La "
            "borne vient du segment de la série que l'export cote en USD/jour, isolé "
            "d'abord comme défaut de données."
        ),
        pivot="La part de repositionnement à vide que le marché price réellement dans le taux publié",
        mail_question=(
            "Quelle part de repositionnement à vide votre taux interne facture-t-il "
            "réellement, et est-elle négociée avec le desk trading ou imposée par le "
            "département fret ?"
        ),
        targets="Desks fret (Cargill Ocean Transportation, Bunge, LDC, COFCO, Viterra) ET traders grains/oléagineux",
        data_gate=GATE_MEDIUM,
        data_fallback="Jambes de prix de l'arb (FOB Santos, CIF Chine) absentes -> la page porte sur le terme de fret seul, dit explicitement",
        status=STATUS_READY,
        dashboard_page="pages/4_T1_1_Fret_CF.py",
        chain_module="agri.chains.freight_cf",
        n_tests=74,
        data_mode=DATA_HYBRID,
    ),
    Project(
        id="hedge_cost",
        code="T1-2",
        tier=TIER_1,
        title="Le coût complet de la couverture — cacao et café",
        thesis="La contrainte contraignante n'est pas le prix, c'est le collatéral.",
        disagreement=(
            "Barry Callebaut S1 2024/25 : marges initiales multipliées par neuf, coût de "
            "backwardation 60 % plus cher au pic. Café nov. 2025 : ~7 Md USD d'appels de "
            "marge en un mois ; chez Montesanto Tavares, le coût de maintien des "
            "couvertures passe de 74 % à 158 % des créances clients, jugé insoutenable "
            "par leurs propres avocats. VÉRIFIÉ SUR PRIX ICE RÉELS : le cacao NY a "
            "réellement culminé à 12 565 USD/t le 18/12/2024, mobilisant 1,08 Md USD de "
            "trésorerie sur un book de 100 kt — ce ne sont plus des ordres de grandeur "
            "reconstruits, ce sont les chiffres du marché ce jour-là."
        ),
        pivot="IM* — la marge initiale à laquelle la capacité de couverture tombe sous le book physique",
        mail_question=(
            "À quel niveau d'initial margin votre desk arrête d'ajouter du physique parce "
            "que la couverture ne se finance plus ? Limite formalisée, ou découverte en route ?"
        ),
        targets="ofi/Olam, ECOM, Volcafe, Sucden Coffee, Touton, Barry Callebaut, Cargill Cocoa, Freepoint softs",
        data_gate=GATE_MEDIUM,
        data_fallback="Pas d'échéance différée réelle disponible -> coût de roll neutralisé (deferred=front), affiché comme limite plutôt qu'estimé",
        status=STATUS_READY,
        dashboard_page="pages/5_T1_2_Cout_Hedge.py",
        chain_module="agri.chains.hedge_cost",
        n_tests=43,
        data_mode=DATA_HYBRID,
    ),
    # ======================================================================
    # TIER 2 — tensions inférées. « Il me semble que », jamais « j'ai lu que ».
    # ======================================================================
    Project(
        id="crush_tracking",
        code="T2-3",
        tier=TIER_2,
        title="Le board crush n'est pas un prix, c'est un rendement déguisé en prix",
        thesis=(
            "Les coefficients 0,022 et 0,11 ne sont pas des conversions d'unité mais des "
            "rendements figés (44 lb / 11 lb par boisseau). Se couvrir au board, c'est les "
            "accepter comme les siens et garder l'écart en position nue."
        ),
        disagreement=(
            "Tension inférée. Les traders board tiennent le crush CBOT pour un proxy "
            "hedgeable de l'économie d'une usine ; les gens d'usine répondent que le basis "
            "tourteau intérieur, le rendement réel et la logistique cassent le hedge "
            "exactement quand il faut. MESURÉ SUR CBOT RÉEL : dans le décile de marge le "
            "plus tendu, un écart de 0,5 lb de tourteau par boisseau — 1,2 % du rendement "
            "que le contrat suppose — suffit à effacer toute la marge nette."
        ),
        pivot=(
            "La précision de rendement que le board exige sans le dire, en livres par "
            "boisseau — et son effondrement dans le régime de marge tendue"
        ),
        mail_question=(
            "De combien votre rendement tourteau réel s'écarte-t-il des 44 lb du board sur "
            "une campagne, et est-ce que quelqu'un couvre cet écart séparément — ou est-ce "
            "qu'il reste dans le résultat ?"
        ),
        targets="Trituration US/Brésil (ADM, Bunge, Cargill, LDC, CHS), risk managers oléagineux",
        data_gate=GATE_NONE,
        data_fallback=(
            "Prix cash locaux absents de l'export -> aucun tracking error mesuré ; la page "
            "est construite pour ne pas en avoir besoin, l'inversion ne demande que le board."
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
        title="Le white premium, ou ce qu'un prix peut dire et ce qu'il ne peut pas",
        thesis=(
            "Le NIVEAU de la rente de raffinage n'est pas identifiable à partir des prix — "
            "un facteur de conversion que personne ne publie pèse autant que la réponse. Sa "
            "VARIATION, elle, l'est entièrement : elle a basculé d'environ 60 USD/t."
        ),
        disagreement=(
            "Tension inférée. Le premium blanc (No.5 − No.11) est présenté comme la marge "
            "de raffinage ; il me semble qu'il contient surtout un résidu de positionnement "
            "et de contraintes de livraison. MESURÉ SUR ICE No.11/No.5 RÉELS avec un proxy "
            "énergie Henry Hub réel : l'ajustement de polarisation qui annulerait la rente "
            "vaut 1,0852, juste au-dessus de la plage plausible [1,06 ; 1,08] — le prix seul "
            "ne tranche pas. Mais le classement des années est identique aux deux bornes "
            "(corrélation de rang 1,0000), et la richness passe de −26 USD/t en 2021 à "
            "+35 en 2024 : un basculement 5,5 fois plus grand que l'incertitude."
        ),
        pivot=(
            "Le prix que le marché paie pour l'acte de raffiner (~70 USD/t), observé sans "
            "aucune hypothèse de coût — un raffineur fait la soustraction lui-même"
        ),
        mail_question=(
            "Est-ce que votre coût de raffinage tout compris est de l'ordre de 70 USD/t ? Et "
            "qu'est-ce qui a changé de votre côté entre 2021 et 2024, quand le prix payé "
            "pour raffiner a basculé d'environ 60 USD/t ?"
        ),
        targets="Raffineurs de destination (Al Khaleej, ASR, Tereos, Südzucker) + Sucden, Czarnikow, Alvean, Wilmar, ED&F Man",
        data_gate=GATE_NONE,
        data_fallback="Main-d'œuvre et fret de raffinage restent des forfaits paramétrés — aucune comptabilité analytique de raffinerie n'est publique",
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
        title="L'usine comme option sur la marge",
        thesis="Une usine souvent en marge négative peut valoir plus qu'une usine stablement rentable.",
        disagreement=(
            "Tension inférée, et c'est le débat stratégique en cours du secteur "
            "(restructuration Cargill, consolidation Bunge/Viterra). Asset-heavy ou "
            "asset-light : les actifs créent-ils de la valeur d'option de trading, ou "
            "détruisent-ils le ROIC ? Halsall note que les marges sont rares dans l'agri "
            "sans actifs. TESTÉ SUR LA VRAIE MARGE DE CRUSH CBOT (soja/tourteau/huile, "
            "aucun terme paramétré) : le verdict ADF+KPSS rejette la stationnarité sur "
            "toute fenêtre — un résultat en soi, pas un échec de calibration : la marge "
            "réelle traverse de vraies ruptures de régime (Covid, guerre en Ukraine, RVO) "
            "qu'un OU à paramètres fixes ne peut pas absorber."
        ),
        pivot="La bande d'hystérésis [M_off, M_on] — la vraie frontière d'exercice",
        mail_question=(
            "Votre règle d'arrêt est-elle un seuil de marge, ou est-ce que le coût de "
            "redémarrage la déplace explicitement ?"
        ),
        targets="Niveau senior, direction de desk, corporate development",
        data_gate=GATE_NONE,
        data_fallback="La marge réelle échoue le test de stationnarité -> calibration OU affichée comme indicative (strict=False), jamais comme frontière ferme",
        status=STATUS_READY,
        dashboard_page="pages/8_T2_5_Usine_Option.py",
        chain_module="agri.chains.plant_option",
        n_tests=36,
        data_mode=DATA_HYBRID,
    ),
    Project(
        id="oil_substitution",
        code="T2-6",
        tier=TIER_2,
        title="La borne de substitution palme-soja n'existe pas",
        thesis=(
            "Testé dans la seule fenêtre où le change ne contamine rien — les sept ans de "
            "parité fixe du ringgit — l'hypothèse ressort INVERSÉE : les écarts étroits "
            "reviennent en 12 jours, les larges ne reviennent pas du tout."
        ),
        disagreement=(
            "Tension inférée. Il me semble que les triturateurs tiennent l'élasticité "
            "palme/soja pour forte et les formulateurs pour collante — reformuler une "
            "recette prend des mois. MESURÉ SUR BURSA + CBOT RÉELS : la palme cote en "
            "ringgits et l'export n'a aucun USDMYR, donc le spread n'est calculable QUE sur "
            "la parité fixe 1998-2005, où le change est une constante décrétée. Sur cette "
            "fenêtre propre, aucune borne de substitution — et le test naïf qui semble en "
            "trouver une sélectionne en réalité l'ère 2004-2005."
        ),
        pivot=(
            "L'absence de retour à la moyenne au-delà de 54 USD/t d'écart — donc : fader un "
            "spread palme-soja large n'a aucun support empirique"
        ),
        mail_question=(
            "À partir de quel écart palme-soja votre téléphone sonne réellement pour un "
            "changement de recette aujourd'hui ? Ma fenêtre propre s'arrête en 2005, avant "
            "le biodiesel, et la borne a très bien pu bouger depuis."
        ),
        targets="Triturateurs et raffineurs d'huiles (Wilmar, Musim Mas, Golden Agri, Bunge, Cargill), formulateurs agroalimentaires",
        data_gate=GATE_NONE,
        data_fallback=(
            "USDMYR absent de l'export (GRATUIT, un seul ticker) -> le test se limite aux "
            "sept ans de parité fixe. Le récupérer débloquerait trente ans d'historique au "
            "lieu de sept : c'est la donnée la plus rentable du portefeuille."
        ),
        status=STATUS_READY,
        dashboard_page="pages/9_T2_6_Substitution_Huiles.py",
        chain_module="agri.chains.oil_substitution",
        data_mode=DATA_REAL,
        n_tests=32,
    ),
    # ======================================================================
    # TIER 3 — désaccords ouverts. Ne jamais prendre parti : quantifier la bascule.
    # ======================================================================
    Project(
        id="feedstock_lcfs",
        code="T3-1",
        tier=TIER_3,
        title="Deux subventions qui se contredisent",
        thesis=(
            "Les deux camps argumentent sur le prix du crédit LCFS, qui ne peut pas "
            "trancher : sur toute l'amplitude réalisée du programme, il ne déplace la "
            "décote exigée de l'UCO importé que de 3,2 c/lb. Ce qui décide, c'est le "
            "spread de prix UCO-soyoil."
        ),
        disagreement=(
            "Trois quarts de la capacité de renewable diesel a été construite sur les "
            "côtes, sitée pour tourner sur du feedstock importé. Puis 45Z exclut du "
            "crédit tout feedstock non nord-américain, pendant que le LCFS californien "
            "continue de payer le faible CI sans regarder l'origine — une politique "
            "pénalise exactement ce que l'autre récompense. Camp A (usines côtières) : "
            "la prime LCFS suffit, les imports tiennent. Camp B (complexe soja) : elle "
            "ne suffit pas, le soyoil rafle la part."
        ),
        pivot=(
            "La décote en c/lb que l'UCO importé doit tenir sous le soyoil — un acheteur "
            "de feedstock la confirme ou la dément sur son propre book en dix secondes"
        ),
        mail_question=(
            "À 75 $/t CO2e sur le LCFS, je trouve que l'UCO importé doit se vendre ~4,5 "
            "c/lb sous le soyoil juste pour compenser le 45Z qu'il ne peut pas réclamer, "
            "et que sur toute l'histoire du programme ce nombre n'a pu varier que de 3,2 "
            "c/lb. Est-ce que la décote que vous voyez rendu USGC est de cet ordre ? Et "
            "en dessous de quel prix collecté cessez-vous simplement de charger ?"
        ),
        targets="Bunge, ADM, LDC, Cargill, CHS + desks bio de Vitol, Gunvor, Freepoint, Trafigura — le plus grand vivier du portefeuille",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "Prix UCO indisponibles (Platts/PGA) — la page est construite pour ne jamais "
            "en avoir besoin : le livrable est une décote relative, pas un prix absolu. "
            "Crédit LCFS absent de l'export (publié par CARB) : traité comme un axe."
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
        title="Le « plancher de coût brésilien » est une série de change",
        thesis=(
            "Un coût de production se libelle en réaux. Traduit en cents par livre pour un "
            "lecteur new-yorkais, un coût CONSTANT produit un plancher qui varie de 20,8 "
            "c/lb — plus que la fourchette du marché lui-même — uniquement via l'USDBRL."
        ),
        disagreement=(
            "Hedgepoint (févr. 2026) : le mix devrait tomber vers 46 % pour réduire "
            "l'excédent, mais les limites d'usine et le sucre déjà vendu à terme l'en "
            "empêchent. Czarnikow (juin 2026) : les mills entrent beaucoup moins couverts "
            "que les quatre saisons précédentes, et le pricing 2026/27 est resté sous "
            "BRL 2 000/t, sous le coût de production. VÉRIFIÉ SUR NY11 + USDBRL RÉELS : "
            "l'affirmation de Czarnikow tient — le sucre vaut 1 843 BRL/t au dernier cours, "
            "et plus de 80 % de 2026 s'est traité sous ce seuil."
        ),
        pivot=(
            "Le plancher NY11 impliqué par un coût constant en réaux — une série, pas un "
            "niveau, et sa corrélation de rang avec l'inverse du change vaut exactement 1"
        ),
        mail_question=(
            "Est-ce que votre équipe raisonne sur un plancher de coût brésilien en cents/lb, "
            "ou est-ce qu'elle le recalcule à chaque mouvement du real ? Et sur 2026/27, à "
            "quel taux de couverture d'entrée de saison vos moulins sont-ils entrés ?"
        ),
        targets="Sucden, Czarnikow, Alvean, Wilmar, LDC Sugar, ED&F Man, Copersucar, BP Bioenergy",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "Mix UNICA et éthanol CEPEA absents de l'export (tous deux GRATUITS) -> "
            "l'élasticité conditionnelle reste non estimée et la spécification est affichée "
            "telle quelle plutôt que simulée. Le reste tourne sur NY11 + USDBRL réels."
        ),
        status=STATUS_READY,
        dashboard_page="pages/11_T3_2_Sucre_Mix.py",
        chain_module="agri.chains.sugar_mix",
        data_mode=DATA_HYBRID,
        n_tests=22,
    ),
    Project(
        id="china_soy",
        code="T3-4",
        tier=TIER_3,
        title="Les fenêtres où aucune origine ne fonctionne",
        thesis=(
            "Plutôt que de tester une signature politique — ce qui demande des données "
            "d'enchères que personne ne publie —, on date les périodes où le budget "
            "d'origination est NÉGATIF : une fève gratuite, transportée gratuitement, ne "
            "rendrait pas le crush rentable. Toute cargaison arrivée là est non commerciale "
            "par construction arithmétique."
        ),
        disagreement=(
            "Sinograin a vendu environ la moitié des 504 000 t proposées à sa plus grosse "
            "enchère depuis janvier ; des traders cités par Reuters y voient de la place "
            "faite pour des cargaisons US (août 2026). En face, ADM relève ses perspectives "
            "2026 en pariant que la Chine continue d'acheter du soja US. MESURÉ SUR CBOT + "
            "DCE + USDCNY RÉELS : 2,0 % des séances depuis 2018 affichent un budget "
            "d'origination négatif, toutes concentrées en 2023, dont une fenêtre de 29 jours "
            "du 7 juin au 5 juillet."
        ),
        pivot=(
            "Le calendrier daté des fenêtres impossibles — des dates confrontables à un "
            "carnet d'arrivées, pas un coefficient"
        ),
        mail_question=(
            "Avez-vous fixé des cargaisons Chine pendant les fenêtres de juin-juillet 2023 "
            "où le budget basis + fret était négatif ? Et si oui, la marge de crush "
            "était-elle vraiment la contrainte, ou un autre maillon portait-il le résultat ?"
        ),
        targets="Origination oléagineux (COFCO, Sinograin, Bunge, LDC, Cargill), desks soja Chine",
        data_gate=GATE_MEDIUM,
        data_fallback=(
            "Enchères Sinograin et douanes GACC absentes -> le test de signature logit n'est "
            "pas passé. Le budget d'origination le remplace et ne dépend NI du basis NI du "
            "fret, les deux séries manquantes : elles sortent du calcul au lieu d'y entrer."
        ),
        status=STATUS_READY,
        dashboard_page="pages/12_T3_4_Chine_Soja.py",
        chain_module="agri.chains.china_soy",
        data_mode=DATA_REAL,
        n_tests=22,
    ),
    # ======================================================================
    # DONNEES REELLES — nes de l'export Bloomberg de l'utilisateur, pas du cadre
    # T1/T2/T3 sourcé/inféré/ouvert. Deux defauts de donnee trouves en les construisant :
    # voir jet_crack (convention d'unite instable sur jet_swap_m1) et le piege d'unite
    # double de lng_netback (energie + rendement de liquefaction).
    # ======================================================================
]


def by_tier() -> dict[str, list[Project]]:
    """Groupe les projets par tier, dans l'ordre d'apparition des tiers."""
    grouped: dict[str, list[Project]] = {}
    for project in PROJECTS:
        grouped.setdefault(project.tier, []).append(project)
    return grouped


def get(project_id: str) -> Project:
    for project in PROJECTS:
        if project.id == project_id:
            return project
    raise KeyError(f"projet inconnu : {project_id!r} (connus : {[p.id for p in PROJECTS]})")


def ready_projects() -> list[Project]:
    return [p for p in PROJECTS if p.status == STATUS_READY]


def projects_at_gate(level: str) -> list[Project]:
    """Les projets dont l'accès aux données porte un risque donné — pilote l'ordre des gates."""
    return [p for p in PROJECTS if p.data_gate == level]
