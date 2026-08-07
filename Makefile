# Supply Chain Platform Infrastructure Makefile

.PHONY: up down logs test help

up:
	docker compose up -d

down:
	docker compose down -v

logs:
	docker compose logs -f

test:
	python scripts/check_env.py

help:
	@echo "Available targets:"
	@echo "  make up    - Start local infrastructure (Postgres, Redis, MLflow, API)"
	@echo "  make down  - Stop and remove infrastructure containers and volumes"
	@echo "  make logs  - Tail container logs"
	@echo "  make test  - Run environment health check (python scripts/check_env.py)"
