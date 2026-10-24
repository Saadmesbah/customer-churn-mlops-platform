"""Versioned public ingestion and shared training/serving schema."""

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pandera.pandas as pa

URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
ARCHIVE_SHA256 = "696c3a1812267980751f30d19193a1b430ac4ad76172bb089e664c96813ead66"
FEATURES = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
]
BOUNDS = {
    "complains": (0, 1),
    "charge_amount": (0, 10),
    "age_group": (1, 5),
    "tariff_plan": (1, 2),
    "status": (1, 2),
    "age": (0, 120),
}


def schema(target=False):
    columns = {
        c: pa.Column(
            float,
            checks=pa.Check.in_range(*BOUNDS[c]) if c in BOUNDS else pa.Check.ge(0),
            coerce=True,
        )
        for c in FEATURES
    }
    if target:
        columns["churn"] = pa.Column(int, pa.Check.isin([0, 1]), coerce=True)
    return pa.DataFrameSchema(columns, strict=True)


def validate(frame, target=False):
    if frame.empty:
        raise ValueError("Dataset must not be empty")
    result = schema(target).validate(frame, lazy=True)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("All features must be finite")
    return result


def ingest(directory=Path("data")):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "churn.csv"
    manifest = directory / "manifest.json"
    if not path.exists():
        with urlopen(URL, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Upstream archive checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            member = next(n for n in archive.namelist() if n.endswith(".csv"))
            frame = pd.read_csv(archive.open(member))
        frame.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in frame.columns]
        frame = validate(frame, target=True)
        frame.to_csv(path, index=False)
        manifest.write_text(
            json.dumps(
                {
                    "source": URL,
                    "doi": "10.24432/C5JW3Z",
                    "license": "CC-BY-4.0",
                    "archive_sha256": hashlib.sha256(raw).hexdigest(),
                    "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
                indent=2,
            )
        )
    if (
        not manifest.exists()
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != json.loads(manifest.read_text())["csv_sha256"]
    ):
        raise ValueError("Cached dataset checksum mismatch; remove cache and ingest again")
    return validate(pd.read_csv(path), target=True)
