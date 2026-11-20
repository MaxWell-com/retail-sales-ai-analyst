PY ?= python

.PHONY: setup data run test eval lint docker

setup:   ## install dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:    ## generate the synthetic sell-out panel and build the DuckDB warehouse (~20 s)
	$(PY) scripts/make_synthetic.py
	$(PY) scripts/build_db.py

run:     ## start the web app on http://localhost:8765 (needs Ollama with $$VT_MODEL pulled)
	$(PY) -m app.server

test:    ## 362 tests without the LLM: question parsing + every tool vs independent SQL
	$(PY) -m pytest

eval:    ## end-to-end eval through the LLM with strict number checks
	$(PY) -m eval.run_eval

lint:
	$(PY) -m ruff check app scripts tests eval

docker:  ## app + Ollama in containers
	docker compose up --build
