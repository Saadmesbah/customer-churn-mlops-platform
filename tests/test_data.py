import hashlib
import io
import json
import zipfile
from unittest.mock import patch

import pandas as pd
import pandera as pa
import pytest

from churn.data import ingest, validate


def test_valid_data(customer):
    assert validate(pd.DataFrame([customer])).shape == (1, 13)


@pytest.mark.parametrize(
    "change",
    [
        {"complains": 2},
        {"age": -1},
        {"age_group": 6},
        {"seconds_of_use": float("nan")},
        {"unexpected": 1},
    ],
)
def test_reject_bad_features(customer, change):
    with pytest.raises(pa.errors.SchemaErrors):
        validate(pd.DataFrame([{**customer, **change}]))


def test_reject_missing(customer):
    del customer["age"]
    with pytest.raises(pa.errors.SchemaErrors):
        validate(pd.DataFrame([customer]))


def test_empty():
    with pytest.raises(ValueError):
        validate(pd.DataFrame())


def test_invalid_target(customer):
    with pytest.raises(pa.errors.SchemaErrors):
        validate(pd.DataFrame([{**customer, "churn": 2}]), target=True)


def test_ingestion_normalization_and_checksum(tmp_path, customer, monkeypatch):
    frame = pd.DataFrame([{**customer, "churn": 1}])
    frame.columns = [c.replace("_", "  ") for c in frame.columns]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("Customer Churn.csv", frame.to_csv(index=False))
    monkeypatch.setattr("churn.data.ARCHIVE_SHA256", hashlib.sha256(buf.getvalue()).hexdigest())
    with patch("churn.data.urlopen", return_value=io.BytesIO(buf.getvalue())):
        assert len(ingest(tmp_path)) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["archive_sha256"] == hashlib.sha256(buf.getvalue()).hexdigest()
    assert len(ingest(tmp_path)) == 1
    (tmp_path / "churn.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        ingest(tmp_path)


def test_reject_infinite_features(customer):
    customer["seconds_of_use"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate(pd.DataFrame([customer]))


def test_reject_changed_upstream(tmp_path):
    with patch("churn.data.urlopen", return_value=io.BytesIO(b"unexpected archive")):
        with pytest.raises(ValueError, match="Upstream archive checksum"):
            ingest(tmp_path)
