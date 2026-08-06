"""Page d'accueil du portefeuille. `streamlit run app/Home.py`"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Physical freight arb portfolio", layout="wide")

st.title("Arbitrages physiques — le fret comme terme décisif")

st.markdown(
    """
Trois chaînes, une règle commune : reconstruire la marge depuis les séries brutes, puis
la confronter à une **série de flux physiques officielle et gratuite** — les douanes,
l'EIA, Eurostat. La question n'est jamais seulement « l'arb est-il ouvert », c'est
« la cargaison est-elle effectivement partie ».
"""
)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("A — Minerai de fer")
    st.markdown(
        """
**Le premium 65-62 % Fe est en partie un spread de fret Capesize.**

Les deux indices sont CFR, donc C3 − C5 est déjà dedans. On décompose, on corrige
l'humidité (le fret se paie sur tonne humide, l'indice se cote sur tonne sèche), et on
regarde ce qui reste.

Validation de flux : douanes chinoises par origine.

*Statut : moteur et dashboard prêts, en attente des séries.*
"""
    )

with col2:
    st.subheader("B — Charbon Atlantique")
    st.markdown(
        """
**L'arb API2 − API4 a perdu sa contrainte contraignante en 2022.**

Le fret ne ferme plus le spread parce que la cargaison marginale de Richards Bay est
cotée sur l'Inde, pas sur Rotterdam. Plus une couche ETS maritime européenne que
personne n'intègre.

Validation de flux : Eurostat + douanes indiennes.

*Statut : à cadrer après A.*
"""
    )

with col3:
    st.subheader("C — Produits transatlantiques")
    st.markdown(
        """
**Le fret coté en points Worldscale n'est pas un coût.**

La conversion en $/t passe par un flat rate réinitialisé chaque janvier : le coût saute
au 1er janvier à points constants.

Validation de flux : imports EIA par pays d'origine.

*Statut : bloqué sur la disponibilité des flat rates Worldscale.*
"""
    )

st.divider()

st.markdown(
    """
### Règles de méthode, valables pour les trois

- **Un trou de données est une information**, pas du bruit à lisser. Aucun forward-fill
  avant l'étape d'audit.
- **Aucune série n'entre dans un calcul sans contrat rempli** : ticker, unité native,
  fréquence, source, et un drapeau `verified` daté.
- **L'unité de cotation n'est pas l'unité économique.** Le fret se paie sur la tonne
  humide et le minerai se cote sur la tonne sèche ; un TC se cote par tonne de concentré
  et se vit par tonne de métal. Ces conversions sont la moitié du travail.
- **Un résidu s'appelle un résidu.** On ne le rebaptise pas « tension physique » pour
  rendre la conclusion plus vendable.
"""
)
