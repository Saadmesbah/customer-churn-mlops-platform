# Verification record

Executed on 2026-10-24 with Python 3.12.14, using the committed uv lockfile.

| Check | Result |
|---|---|
| Ruff lint and formatting | Passed |
| Pytest suite | 21 passed; 96% statement coverage |
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
| Docker build and Compose runtime | Passed in GitHub Actions: six-service stack, API and dashboard health checks |
| PostgreSQL-backed MLflow runtime | Passed in Docker CI, including remote tracking, training and registry |
| GitHub publication and Actions execution | Published; quality, real lifecycle integration and Docker smoke checks passed |
| Python 3.11 | Declared compatible; not executed here |

The passing tests emit 12 dependency deprecation/MLflow warnings; there are no failed tests. MLflow uses explicit cloudpickle serialization for the trusted training pipeline. The dependency lock constrains Pandas below 3 to avoid an observed Evidently import incompatibility.

CI evidence: https://github.com/Saadmesbah/customer-churn-mlops-platform/actions/runs/40124403150 (tested application commit `778654e`). The first Docker run exposed a SQLAlchemy driver mismatch: the ambiguous `postgresql://` URI selected Psycopg 3, while the project installs Psycopg 2. The Compose URI now explicitly uses `postgresql+psycopg2://`, and a regression test verifies that the configured database driver imports.

Docker is unavailable in the editing environment, so container verification was performed on GitHub's Linux runner. A Windows 11 local run remains optional and was not performed here. No production traffic, public runtime deployment or latency SLA is claimed. The tagged GHCR publishing workflow is configured but has not been executed. See README for deployment boundaries and model limitations.
