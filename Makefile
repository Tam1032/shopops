PY ?= python

.PHONY: install test eval eval-model up demo

install:
	pip install -e '.[dev]'

test:
	$(PY) -m pytest -q

eval:            ## offline: keyword baseline, no API key needed
	$(PY) -m shopops.evals --model baseline

eval-model:      ## real model from LLM_* env, judged by JUDGE_* env
	$(PY) -m shopops.evals --model env $(if $(JUDGE_MODEL),--judge judge-env,)

up:
	docker compose up --build

demo:            ## customers post 3 new reviews; the webhook sends them to n8n
	curl -s -X POST 'localhost:8000/admin/release?n=3'
