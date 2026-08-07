"""Page d'accueil du portefeuille. `streamlit run app/Home.py`

Ne contient aucune donnée en dur : tout vient de `freight.portfolio`. Pour ajouter un
projet au portefeuille, on ajoute un `Project` dans `src/freight/portfolio.py` — cette
page n'a plus jamais besoin d'être modifiée à la main.
"""
from __future__ import annotations

import streamlit as st

from freight.portfolio import PROJECTS, STATUS_READY, by_sector, total_tests

st.set_page_config(page_title="Physical arb portfolio", layout="wide")

st.title("Arbitrages physiques — le fret comme terme décisif")

st.markdown(
    """
Une règle commune à tout le portefeuille, quel que soit le secteur : reconstruire la
marge depuis les séries brutes, puis la confronter à une **série de flux physiques
officielle et gratuite** — douanes, EIA, Eurostat. La question n'est jamais seulement
« l'arb a-t-il l'air ouvert », c'est **« la cargaison est-elle effectivement partie »**.
"""
)

n_ready = sum(1 for p in PROJECTS if p.status == STATUS_READY)
n_planned = len(PROJECTS) - n_ready
c1, c2, c3 = st.columns(3)
c1.metric("Projets prêts", n_ready)
c2.metric("Projets en piste", n_planned)
c3.metric("Golden tests", total_tests())

st.divider()

for sector, projects in by_sector().items():
    st.subheader(sector)
    cols = st.columns(len(projects))
    for col, project in zip(cols, projects):
        with col:
            badge = "🟢 prêt" if project.status == STATUS_READY else "⚪ à construire"
            st.markdown(f"**{project.letter} — {project.title}** &nbsp; {badge}")
            st.markdown(f"*{project.thesis}*")
            with st.expander("Mécanisme"):
                st.markdown(project.mechanism)
                if project.flow_validation:
                    st.caption(f"Validation de flux : {project.flow_validation}")
            st.caption(project.status_detail)
            if project.dashboard_page:
                st.page_link(project.dashboard_page, label="Ouvrir le dashboard →")
    st.divider()

st.markdown(
    """
### Règles de méthode, valables pour tout le portefeuille

- **Un trou de données est une information**, pas du bruit à lisser. Aucun forward-fill
  avant l'étape d'audit — `gap_policy` n'accepte que `none`, le refus est codé.
- **Aucune série n'entre dans un calcul sans contrat rempli** : ticker, unité native,
  fréquence, source, et un drapeau `verified` daté.
- **L'unité de cotation n'est pas l'unité économique.** Tonne humide contre tonne sèche,
  kcal contre tonne, gallon contre tonne — cette conversion est la moitié du travail de
  chaque projet, pas un détail.
- **Un résidu s'appelle un résidu.** On ne le rebaptise pas « tension physique » pour
  rendre la conclusion plus vendable.
- **Le mode synthétique est signalé**, jamais confondu avec un résultat réel. Aucun
  chiffre produit dessus ne doit sortir du repo.

Pour ajouter un nouveau secteur (agriculture, LNG, ...) : `docs/NOUVELLE_CHAINE.md`
décrit le gabarit exact que suivent A, B et C.
"""
)
