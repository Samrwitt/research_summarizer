.PHONY: up build down logs backend-check ai-health

up:
	docker compose up --build

build:
	docker compose build

down:
	docker compose down

logs:
	docker compose logs -f

backend-check:
	python backend/manage.py check

ai-health:
	python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8001/health').read().decode())"
