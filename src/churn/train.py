"""Leakage-safe model selection. The holdout is evaluated only after selection."""

import json
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churn.data import FEATURES, ingest

logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","message":"%(message)s"}')
SEED = 42


def metrics(y, probability):
    pred = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train(data_dir="data", output_dir="artifacts"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = ingest(Path(data_dir))
    # Identical predictor profiles must never straddle CV or holdout boundaries.
    original_rows = len(df)
    conflicts = int((df.groupby(FEATURES)["churn"].nunique() > 1).sum())
    df = df.drop_duplicates().reset_index(drop=True)
    groups = pd.util.hash_pandas_object(df[FEATURES], index=False)
    train_idx, test_idx = next(
        StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
            df[FEATURES], df.churn, groups
        )
    )
    x_train, x_test = df[FEATURES].iloc[train_idx], df[FEATURES].iloc[test_idx]
    y_train, y_test = df.churn.iloc[train_idx], df.churn.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("customer-churn")
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"model__C": [0.1, 1.0]},
        ),
        "xgboost": (
            XGBClassifier(n_estimators=150, n_jobs=1, random_state=SEED, eval_metric="logloss"),
            {"model__max_depth": [2, 4], "model__learning_rate": [0.05, 0.1]},
        ),
    }
    results = []
    for name, (estimator, grid) in candidates.items():
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="roc_auc",
            cv=StratifiedGroupKFold(3, shuffle=True, random_state=SEED),
            n_jobs=1,
            error_score="raise",
        )
        with mlflow.start_run(run_name=name) as run:
            search.fit(x_train, y_train, groups=train_groups)
            mlflow.log_params(
                {**search.best_params_, "seed": SEED, "cv_folds": 3, "threshold": 0.5}
            )
            mlflow.log_metric("cv_roc_auc", search.best_score_)
            mlflow.log_dict(
                json.loads(Path(data_dir, "manifest.json").read_text()), "data_manifest.json"
            )
            mlflow.sklearn.log_model(
                search.best_estimator_,
                artifact_path="model",
                serialization_format="cloudpickle",
                signature=infer_signature(x_train, search.best_estimator_.predict(x_train)),
                input_example=x_train.head(2),
            )
            results.append((search.best_score_, name, search.best_estimator_, run.info.run_id))
    eligible = [r for r in results if r[0] >= 0.75]
    if not eligible:
        raise RuntimeError("No candidate achieved CV ROC-AUC >= 0.75; no model published")
    score, name, model, run_id = max(eligible, key=lambda r: (r[0], r[1]))
    evaluation = metrics(y_test, model.predict_proba(x_test)[:, 1])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {"test_" + k: v for k, v in evaluation.items() if k != "confusion_matrix"}
        )
        mlflow.log_dict(evaluation, "evaluation.json")
    version = mlflow.register_model(f"runs:/{run_id}/model", "customer-churn")
    MlflowClient().set_registered_model_alias("customer-churn", "champion", version.version)
    metadata = {
        "model": name,
        "version": version.version,
        "run_id": run_id,
        "cv_roc_auc": score,
        "threshold": 0.5,
        "metrics": evaluation,
        "original_rows": original_rows,
        "conflicting_profiles": conflicts,
        "unique_rows": len(df),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "candidates": [{"model": r[1], "cv_roc_auc": r[0]} for r in results],
        "features": FEATURES,
    }
    joblib.dump(model, out / "model.joblib")
    metadata["model_sha256"] = (
        __import__("hashlib").sha256((out / "model.joblib").read_bytes()).hexdigest()
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    x_train.to_csv(out / "reference.csv", index=False)
    x_test.to_csv(out / "current.csv", index=False)
    (out / "example.json").write_text(
        json.dumps({"customers": [x_test.iloc[0].to_dict()]}, indent=2)
    )
    logging.info("training_completed model=%s cv_auc=%.4f", name, score)
    return metadata


if __name__ == "__main__":
    train()
