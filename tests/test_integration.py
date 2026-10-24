import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from churn.api import create_app
from churn.monitor import monitor
from churn.train import train


@pytest.mark.integration
def test_real_lifecycle(tmp_path, monkeypatch):
    tracking = f"sqlite:///{tmp_path}/tracking.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking)
    output = tmp_path / "artifacts"
    result = train(tmp_path / "data", output)
    assert result["cv_roc_auc"] >= 0.75
    assert 0 <= result["metrics"]["roc_auc"] <= 1
    assert sum(map(sum, result["metrics"]["confusion_matrix"])) == result["test_rows"]
    client = MlflowClient(tracking_uri=tracking)
    assert (
        client.get_model_version_by_alias("customer-churn", "champion").version == result["version"]
    )
    ref = pd.read_csv(output / "reference.csv")
    cur = pd.read_csv(output / "current.csv")
    assert not set(map(tuple, ref.values)) & set(map(tuple, cur.values))
    with TestClient(create_app(output)) as api:
        assert api.get("/health").status_code == 200
        assert api.get("/model-info").json()["run_id"] == result["run_id"]
        payload = json.loads((output / "example.json").read_text())
        response = api.post("/predict", json=payload)
        assert response.status_code == 200
        probability = response.json()["predictions"][0]["churn_probability"]
        import joblib

        expected = joblib.load(output / "model.joblib").predict_proba(
            pd.DataFrame(payload["customers"])
        )[0, 1]
        assert probability == pytest.approx(float(expected))
    report = tmp_path / "reports"
    monitor(output / "reference.csv", output / "current.csv", report)
    assert (report / "drift.html").stat().st_size > 1000
    assert json.loads((report / "drift.json").read_text())["metrics"]
    # Controlled covariate shift is a separate, explicitly labelled scenario.
    shifted = cur.copy()
    shifted["seconds_of_use"] += 100000
    shifted.to_csv(tmp_path / "shifted.csv", index=False)
    monitor(output / "reference.csv", tmp_path / "shifted.csv", tmp_path / "shift_report")
    assert (tmp_path / "shift_report/drift.json").read_text() != (report / "drift.json").read_text()
    shifted_report = json.loads((tmp_path / "shift_report/drift.json").read_text())
    usage = next(
        m for m in shifted_report["metrics"] if m["config"].get("column") == "seconds_of_use"
    )
    assert usage["value"] > usage["config"]["threshold"]
    evidence = Path("evidence")
    evidence.mkdir(exist_ok=True)
    shutil.copy(output / "metadata.json", evidence / "evaluation.json")
    shutil.copy(report / "drift.json", evidence / "drift.json")
    (output / "model.joblib").write_bytes(b"tampered")
    with TestClient(create_app(output)) as api:
        assert api.get("/health").status_code == 503
