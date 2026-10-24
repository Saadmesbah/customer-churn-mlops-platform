# Verification record

Executed on 2026-10-24 with Python 3.12.14, using the committed uv lockfile.

| Check | Result |
|---|---|
| Ruff lint and formatting | Passed |
| Pytest suite | 20 passed; 96% statement coverage |
| Real UCI ingestion and pinned ZIP checksum | Passed |
| Schema, upstream modification, cache corruption, invalid API inputs | Passed |
| Logistic regression and XGBoost tuning | Passed; 6 configurations, 3 grouped CV folds |
| MLflow experiment artifacts and champion registry alias | Passed against local SQLite backend |
| Real model/API prediction parity | Passed |
| Missing and corrupt model readiness | Passed; 503 response |
| Train/holdout profile overlap | None; asserted in integration test |
| Evidently reference/holdout report | Generated; 0 of 13 drifted columns |
| Explicit simulated usage shift | Generated; usage drift statistic exceeds its threshold |
| Live local FastAPI and Streamlit, including prediction and metrics views | Passed; actual screenshots captured and inspected |
| Docker build and Compose runtime | Not executed: Docker CLI/daemon unavailable |
| PostgreSQL-backed MLflow runtime | Not executed locally; covered by configured Compose CI smoke job |
| GitHub publication and Actions execution | Pending repository creation; connected integration has no repository creation operation |
| Python 3.11 | Declared compatible; not executed here |

The passing tests emit 12 dependency deprecation/MLflow warnings; there are no failed tests. MLflow uses explicit cloudpickle serialization for the trusted training pipeline. The dependency lock constrains Pandas below 3 to avoid an observed Evidently import incompatibility.

No fabricated metrics, production traffic, live deployment, latency SLA, or CI success is claimed. The tagged GHCR publishing workflow is configured but has not been executed. See README for deployment boundaries and model limitations.
