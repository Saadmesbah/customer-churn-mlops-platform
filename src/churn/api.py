"""Read-only model serving with bounded requests and readiness checks."""

import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from churn.data import validate

log = logging.getLogger("churn.api")


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    call_failure: float = Field(ge=0)
    complains: int = Field(ge=0, le=1)
    subscription_length: float = Field(ge=0)
    charge_amount: int = Field(ge=0, le=10)
    seconds_of_use: float = Field(ge=0)
    frequency_of_use: float = Field(ge=0)
    frequency_of_sms: float = Field(ge=0)
    distinct_called_numbers: float = Field(ge=0)
    age_group: int = Field(ge=1, le=5)
    tariff_plan: int = Field(ge=1, le=2)
    status: int = Field(ge=1, le=2)
    age: float = Field(ge=0, le=120)
    customer_value: float = Field(ge=0)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[Customer] = Field(min_length=1, max_length=1000)


def create_app(artifact_dir=None):
    directory = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = None
        app.state.metadata = None
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
            if (
                hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()
                != metadata["model_sha256"]
            ):
                raise ValueError("Model integrity mismatch")
            # Only load artifacts produced by the trusted training pipeline.
            app.state.model = joblib.load(directory / "model.joblib")
            app.state.metadata = metadata
        except (OSError, ValueError):
            log.exception("model_load_failed")
        yield

    app = FastAPI(title="Customer Churn MLOps", version="1.0.0", lifespan=lifespan)

    def ready():
        if app.state.model is None:
            raise HTTPException(503, "Model unavailable; run the training pipeline")

    @app.get("/health")
    def health():
        ready()
        return {"status": "ready", "version": app.state.metadata["version"]}

    @app.get("/model-info")
    def model_info():
        ready()
        return app.state.metadata

    @app.post("/predict")
    def predict(batch: Batch):
        ready()
        started = time.perf_counter()
        frame = validate(pd.DataFrame([c.model_dump() for c in batch.customers]))
        probabilities = app.state.model.predict_proba(frame)[:, 1]
        log.info(
            json.dumps(
                {
                    "event": "prediction",
                    "batch_size": len(frame),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "version": app.state.metadata["version"],
                }
            )
        )
        return {
            "model_version": app.state.metadata["version"],
            "predictions": [
                {"churn_probability": float(p), "churn": bool(p >= app.state.metadata["threshold"])}
                for p in probabilities
            ],
        }

    return app


app = create_app()
