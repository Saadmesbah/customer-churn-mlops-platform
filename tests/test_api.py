import pytest
from fastapi.testclient import TestClient

from churn.api import create_app


def test_missing_model_readiness(tmp_path, customer):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/model-info").status_code == 503
        assert client.post("/predict", json={"customers": [customer]}).status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"customers": []},
        {"customers": [{}]},
        {"customers": [{}] * 1001},
        {"customers": [], "secret": 1},
    ],
)
def test_invalid_requests(tmp_path, payload):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_negative_input(tmp_path, customer):
    customer["age"] = -1
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/predict", json={"customers": [customer]}).status_code == 422
