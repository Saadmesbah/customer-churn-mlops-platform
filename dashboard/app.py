"""Interactive API client and offline evidence viewer."""

import json
import os
from pathlib import Path

import httpx
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Churn Operations", page_icon="📊", layout="wide")
st.title("Customer Churn · Operations")
st.caption("Model serving, evaluation and batch drift evidence")
url = os.getenv("API_URL", "http://localhost:8000")
tab1, tab2, tab3 = st.tabs(["Prediction", "Model performance", "Drift monitoring"])
with tab1:
    st.write("Enter a customer feature record. Schema and bounds are enforced by the API.")
    example = Path("artifacts/example.json")
    payload = st.text_area(
        "JSON request", example.read_text() if example.exists() else '{"customers": []}', height=300
    )
    if st.button("Predict churn", type="primary"):
        try:
            response = httpx.post(
                url + "/predict", json=json.loads(payload), timeout=30, trust_env=False
            )
            response.raise_for_status()
            st.dataframe(response.json()["predictions"])
        except (ValueError, httpx.HTTPError) as exc:
            st.error(str(exc))
with tab2:
    try:
        response = httpx.get(url + "/model-info", timeout=10, trust_env=False)
        response.raise_for_status()
        info = response.json()
        cols = st.columns(4)
        for col, key in zip(cols, ["roc_auc", "f1", "precision", "recall"]):
            col.metric(key.upper(), f"{info['metrics'][key]:.3f}")
        st.write("Confusion matrix: rows = actual, columns = predicted; class order [0, 1]")
        st.dataframe(info["metrics"]["confusion_matrix"])
        st.json(info)
    except httpx.HTTPError as exc:
        st.warning(f"API unavailable: {exc}")
with tab3:
    path = Path("reports/drift.html")
    st.info(
        "Reference = training features; current = holdout features by default. Supply a real new batch to monitor deployment drift."
    )
    if path.exists():
        components.html(path.read_text(), height=850, scrolling=True)
    else:
        st.warning("Run python -m churn.monitor to generate a report.")
