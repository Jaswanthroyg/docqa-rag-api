.PHONY: up down test lint eval load
up:    ; docker compose up --build -d
down:  ; docker compose down
test:  ; pytest -q
lint:  ; ruff check .
eval:  ; python eval/run_eval.py --base-url http://localhost:8000
load:  ; locust -f locustfile.py --host http://localhost:8000
