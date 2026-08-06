.PHONY: install test audit app smoke clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,app]"

install: $(VENV)/bin/activate

# les golden tests ne demandent aucune donnee : a lancer n'importe quand
test: install
	$(PY) -m pytest -q

# audit de couverture, des que data/raw/ contient des exports et que
# data_dictionary.csv est rempli avec verified=oui
audit: install
	$(PY) -m freight.ingest.audit --raw data/raw --dictionary data_dictionary.csv \
		--out data/interim/audit_report.csv

# verifie que le pipeline tourne de bout en bout sur le jeu SYNTHETIQUE
smoke: install
	$(PY) scripts/smoke_ironore.py

app: install
	$(VENV)/bin/streamlit run app/Home.py

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
