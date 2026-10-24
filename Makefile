install:
	uv sync --frozen
train:
	uv run python -m churn.train
monitor:
	uv run python -m churn.monitor
api:
	uv run uvicorn churn.api:app --host 127.0.0.1 --port 8000 --log-config logging.json
dashboard:
	uv run streamlit run dashboard/app.py
test:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest --cov=churn
