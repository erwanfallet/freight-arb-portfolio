# Deploying the dashboard on Streamlit Community Cloud

The repo is ready on the code side (`requirements.txt` at the root, tested in a clean
venv). What's left is outside what I can do myself: it goes through **your** GitHub
account and **your** Streamlit account — I don't create accounts and I don't publish
public content on your behalf without explicit confirmation.

**`gh` (the GitHub CLI) is installed on this machine but the token has expired** — either
way you'll need to run `gh auth login` yourself, or go through the GitHub web interface.

## Option 1 — Fast, but public

The free tier of Streamlit Community Cloud requires a **public** GitHub repo. Anyone with
the link (or who stumbles on it while searching) can read `PROJECT_NOTES.md`, the
theses, the method — everything at the heart of the cold-outreach approach.

1. Create a GitHub repo (public), e.g. `freight-arb-portfolio`
2. From `freight-project/`:
   ```bash
   git remote add origin https://github.com/<your-user>/freight-arb-portfolio.git
   git add -A
   git commit -m "A/B/C portfolio + platform"
   git push -u origin main
   ```
3. On [share.streamlit.io](https://share.streamlit.io): sign in with GitHub, "New app",
   pick the repo, branch `main`, main file `app/Home.py`, deploy.

## Option 2 — Private

Two ways to keep this private:

- **Private GitHub repo + paid Streamlit Cloud** (the free tier doesn't deploy from a
  private repo)
- **Stay local**: `make app` launches the dashboard on `localhost:8501` — plenty for a
  video call or screen share, without publishing anything

## What's already ready, whichever option you pick

- `requirements.txt` — verified in a clean venv, installs the package and the app in one shot
- `app/Home.py` as the entry point, already what `make app` uses locally
- 90 tests green, nothing broken by adding the platform
