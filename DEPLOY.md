# Déployer le dashboard sur Streamlit Community Cloud

Le repo est prêt côté code (`requirements.txt` à la racine, testé dans un venv propre).
Ce qui reste est hors de ce que je peux faire moi-même : ça passe par **ton** compte
GitHub et **ton** compte Streamlit — je ne crée pas de comptes et je ne publie pas de
contenu public en ton nom sans confirmation explicite.

**`gh` (CLI GitHub) est installé sur cette machine mais le token est expiré** — il faudra
de toute façon repasser par toi pour `gh auth login` ou par l'interface web GitHub.

## Option 1 — Rapide, mais public

Le tiers gratuit de Streamlit Community Cloud exige un dépôt GitHub **public**. N'importe
qui avec le lien (ou qui tombe dessus en cherchant) peut lire `PROJECT_NOTES.md`, les
thèses, la méthode — tout ce qui est le cœur de la démarche de cold outreach.

1. Créer un repo GitHub (public), par exemple `freight-arb-portfolio`
2. Depuis `freight-project/` :
   ```bash
   git remote add origin https://github.com/<ton-user>/freight-arb-portfolio.git
   git add -A
   git commit -m "Portefeuille A/B/C + plateforme"
   git push -u origin main
   ```
3. Sur [share.streamlit.io](https://share.streamlit.io) : se connecter avec GitHub, "New app",
   choisir le repo, branche `main`, fichier principal `app/Home.py`, déployer.

## Option 2 — Privé

Deux façons de garder ça privé :

- **Repo GitHub privé + Streamlit Cloud payant** (le tiers gratuit ne déploie pas depuis
  un repo privé)
- **Rester en local** : `make app` lance le dashboard sur `localhost:8501` — largement
  suffisant pour un partage en visio ou un screenshare, sans rien publier

## Ce qui est déjà prêt, quel que soit le choix

- `requirements.txt` — vérifié dans un venv propre, installe le paquet et l'app d'un coup
- `app/Home.py` comme point d'entrée, déjà celui que `make app` utilise en local
- 90 tests verts, rien de cassé par l'ajout de la plateforme
