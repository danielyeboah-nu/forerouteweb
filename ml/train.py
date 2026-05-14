"""
ForeRoute segment-risk training pipeline.

Trains three models (Decision Tree, Gradient Boosting, MLP) on the
fore_route_segments_large.csv dataset, logs each run to MLflow with autolog,
and registers the best (by validation ROC-AUC) into the MLflow Model Registry
under the name configured by REGISTRY_MODEL_NAME (default: ForeRoute-SegmentRisk).

The label is a stochastic hazard target derived from the same engineered logic
as the original notebook -- this keeps the model honest (no deterministic
leakage), and lets MLflow track the full training/registration loop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "fore_route_segments_large.csv"
DEFAULT_FALLBACK = PROJECT_ROOT / "data" / "fore_route_segments.csv"

EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "foreroute-segment-risk")
REGISTRY_MODEL_NAME = os.environ.get("REGISTRY_MODEL_NAME", "ForeRoute-SegmentRisk")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
RANDOM_STATE = 42

FEATURE_COLS = [
    "temperature",
    "precipitation_type",
    "precipitation_intensity",
    "wind_speed",
    "visibility",
    "humidity",
    "dew_point",
    "road_type",
    "segment_distance_m",
    "lat",
    "lon",
    "hour_of_day",
    "day_of_week",
    "month",
    "prior_year_crash_count",
]
NUMERIC = [
    "temperature",
    "precipitation_intensity",
    "wind_speed",
    "visibility",
    "humidity",
    "dew_point",
    "segment_distance_m",
    "lat",
    "lon",
    "hour_of_day",
    "day_of_week",
    "month",
    "prior_year_crash_count",
]
CATEGORICAL = ["precipitation_type", "road_type"]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        if DEFAULT_FALLBACK.exists():
            print(f"[warn] {path} missing, falling back to {DEFAULT_FALLBACK}")
            path = DEFAULT_FALLBACK
        else:
            raise FileNotFoundError(f"No dataset found at {path} or {DEFAULT_FALLBACK}")
    print(f"[data] loading {path}")
    return pd.read_csv(path)


def build_label(df: pd.DataFrame) -> pd.Series:
    """Stochastic hazard label (mirrors the XAI notebook)."""
    hazard = (
        3.0 * df["precipitation_type"].isin(["snow", "sleet"]).astype(float)
        + 0.8 * df["precipitation_intensity"]
        + 1.4 * (df["precipitation_intensity"] >= 6.0).astype(float)
        + 0.9 * np.maximum(0.0, 2.5 - df["visibility"])
        + 0.6 * np.maximum(0.0, 2.0 - df["temperature"])
        + 0.04 * np.maximum(0.0, df["wind_speed"] - 30.0)
        + 0.02 * np.maximum(0.0, df["humidity"] - 80.0)
    )
    z = (hazard - hazard.mean()) / (hazard.std() + 1e-9)
    prob = 1.0 / (1.0 + np.exp(-(1.1 * z - 0.6)))
    rng = np.random.default_rng(RANDOM_STATE)
    sampled = (rng.random(len(df)) < prob).astype(int)
    flip = rng.random(len(df)) < 0.10
    return pd.Series(np.where(flip, 1 - sampled, sampled), index=df.index, name="risky_condition_label")


def split_data(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None):
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return (
            X.iloc[train_idx].copy(),
            X.iloc[test_idx].copy(),
            y.iloc[train_idx].copy(),
            y.iloc[test_idx].copy(),
        )
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def rebalance(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = X.copy()
    df["_y"] = y.values
    pos = df[df["_y"] == 1]
    neg = df[df["_y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    neg_keep = min(len(neg), int(len(pos) * 1.5))
    out = pd.concat([pos, neg.sample(n=neg_keep, random_state=RANDOM_STATE)], axis=0)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE)
    return out[X.columns].copy(), out["_y"].astype(int).copy()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)]
    )


def model_pipelines() -> dict[str, Pipeline]:
    return {
        "decision_tree": Pipeline(
            [
                ("pre", build_preprocessor()),
                (
                    "clf",
                    DecisionTreeClassifier(
                        max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("pre", build_preprocessor()),
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "mlp": Pipeline(
            [
                ("pre", build_preprocessor()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=600,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


class ProbaWrapper(mlflow.pyfunc.PythonModel):
    """Serves predict_proba positive-class probability so /invocations returns
    floats in [0, 1] instead of hard class labels."""

    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def predict(self, context, model_input, params=None):  # noqa: ARG002
        proba = self.pipe.predict_proba(model_input)
        return proba[:, 1].astype(float)


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="If set, use this column as the binary label instead of building one synthetically.",
    )
    parser.add_argument(
        "--registry-name",
        type=str,
        default=REGISTRY_MODEL_NAME,
        help="Registered model name to upsert under.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument("--no-register", action="store_true", help="skip model registry step")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(args.experiment)
    print(f"[mlflow] tracking_uri={TRACKING_URI} experiment={args.experiment}")

    df = load_dataset(args.data)
    if args.label_col:
        if args.label_col not in df.columns:
            raise SystemExit(f"--label-col '{args.label_col}' not in dataset columns: {list(df.columns)}")
        y = df[args.label_col].astype(int).rename("label")
        print(f"[data] using real label column '{args.label_col}': pos_rate={y.mean():.3f}")
    else:
        y = build_label(df)
    X = df[FEATURE_COLS].copy()
    groups = df["route_id"] if "route_id" in df.columns else None

    X_train_raw, X_test, y_train_raw, y_test = split_data(X, y, groups)
    X_train, y_train = rebalance(X_train_raw, y_train_raw)
    print(
        f"[data] train={len(X_train)} test={len(X_test)} "
        f"train_pos_rate={y_train.mean():.3f} test_pos_rate={y_test.mean():.3f}"
    )

    # Avoid sklearn autologging here. The manual logging below is enough for
    # this project and prevents platform-specific matplotlib/font crashes while
    # autologging tries to create estimator artifacts.
    mlflow.sklearn.autolog(disable=True)

    results: dict[str, dict] = {}
    for name, pipe in model_pipelines().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("model_family", name)
            mlflow.log_param("rows_train", len(X_train))
            mlflow.log_param("rows_test", len(X_test))
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)
            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            proba_test = pipe.predict_proba(X_test)[:, 1]
            signature = infer_signature(X_test, proba_test)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=ProbaWrapper(pipe),
                signature=signature,
                input_example=X_test.head(3),
            )

            print(f"[run ] {name:>20s}  {metrics}")
            results[name] = {
                "run_id": run.info.run_id,
                "metrics": metrics,
            }

    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best = results[best_name]
    print(f"[best] {best_name} run_id={best['run_id']} roc_auc={best['metrics']['roc_auc']:.4f}")

    if args.no_register:
        return

    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(args.registry_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists

    model_uri = f"runs:/{best['run_id']}/model"
    version = client.create_model_version(
        name=args.registry_name,
        source=model_uri,
        run_id=best["run_id"],
        description=(
            f"Best of {len(results)} runs by test ROC-AUC. "
            f"family={best_name} roc_auc={best['metrics']['roc_auc']:.4f}"
        ),
    )
    client.set_registered_model_alias(args.registry_name, "production", version.version)
    print(
        f"[reg ] registered {args.registry_name} v{version.version} "
        f"(alias=production) from {model_uri}"
    )


if __name__ == "__main__":
    main()
