.PHONY: help sync fmt lint type test check perf mutate db-up db-down migrate db-roles dashboards synth

UV ?= uv
PATHS := packages tests strategies migrations scripts

help:
	@echo "sync     install the workspace (uv)"
	@echo "fmt      format code"
	@echo "check    ruff + mypy --strict + pytest (the phase gate)"
	@echo "mutate   break each safety rule once; the tests must catch every one"
	@echo "db-up    start postgres (timescale+pgvector), redis, grafana"
	@echo "migrate  run alembic migrations"
	@echo "db-roles give Grafana's read-only role its login (secrets/db_readonly_password.txt)"
	@echo "dashboards  regenerate grafana/provisioning/dashboards/json from scripts/dashboards.py"

sync:
	$(UV) sync --python 3.12

fmt:
	$(UV) run ruff format $(PATHS)
	$(UV) run ruff check --fix $(PATHS)

lint:
	$(UV) run ruff format --check $(PATHS)
	$(UV) run ruff check $(PATHS)

type:
	$(UV) run mypy packages tests strategies scripts

test:
	$(UV) run pytest

check: lint type test

mutate:
	$(UV) run python scripts/mutate.py

perf:
	$(UV) run pytest -m slow -s -n 0

db-up:
	docker compose up -d postgres redis grafana

db-down:
	docker compose down

migrate:
	$(UV) run alembic upgrade head

db-roles:
	@test -s secrets/db_readonly_password.txt || (echo "create secrets/db_readonly_password.txt first" && exit 1)
	@echo "ALTER ROLE autotrader_readonly LOGIN PASSWORD :'pw';" | docker compose exec -T postgres \
		psql -q -U autotrader -d autotrader -v ON_ERROR_STOP=1 -v pw="$$(cat secrets/db_readonly_password.txt)" \
		&& echo "autotrader_readonly can log in"

dashboards:
	$(UV) run python scripts/dashboards.py

synth:
	$(UV) run at data synth --symbol SYNTH --days 30 --out data/synthetic
