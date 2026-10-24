"""Explicit batch monitoring; no silent simulation of production traffic."""

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from churn.data import validate


def monitor(reference, current, output="reports"):
    ref = validate(pd.read_csv(reference))
    cur = validate(pd.read_csv(current))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
    snapshot.save_html(str(out / "drift.html"))
    snapshot.save_json(str(out / "drift.json"))
    summary = {
        "reference_rows": len(ref),
        "current_rows": len(cur),
        "scenario": "Supplied batch comparison; drift does not imply model performance degradation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="artifacts/reference.csv")
    parser.add_argument("--current", default="artifacts/current.csv")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    monitor(args.reference, args.current, args.output)
