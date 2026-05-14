"""
Regenerate the four Evidently monitoring reports against the *current*
registered model (`ForeRoute-BostonRisk@production`) and the 15-feature
Boston crash dataset. Writes self-contained HTML to ml/reports/.

Reports produced
  · 01_data_drift.html           DataDriftPreset across all 15 inputs
  · 02_data_quality.html         DataSummaryPreset (nulls, ranges, dtypes)
  · 03_output_drift.html         Drift on the model's prediction probability
  · 04_classification_performance.html
                                  PR/ROC, confusion matrix at threshold 0.35

Reference window: the 80% training fold (pre-rebalance).
Current window:   a held-out 20% stratified fold acting as production proxy.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

from evidently import Dataset, DataDefinition, Report
from evidently.core.datasets import BinaryClassification
from evidently.presets import (
    ClassificationPreset,
    DataDriftPreset,
    DataSummaryPreset,
)

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{PROJECT / 'mlflow.db'}"
DATA = PROJECT / "data" / "boston_crash_dataset.csv"
REPORTS_DIR = PROJECT / "reports"

FEATURE_COLS = [
    "temperature", "precipitation_type", "precipitation_intensity",
    "wind_speed", "visibility", "humidity", "dew_point",
    "road_type", "segment_distance_m",
    "lat", "lon", "hour_of_day", "day_of_week", "month",
    "prior_year_crash_count",
]
NUMERIC = [c for c in FEATURE_COLS if c not in ("precipitation_type", "road_type")]
CATEGORICAL = ["precipitation_type", "road_type"]

CLASSIFICATION_THRESHOLD = 0.35  # Matches REPORT.md § 9.4


def load_pipeline():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias("ForeRoute-BostonRisk", "production")
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{mv.run_id}/model")
    return pyfunc_model.unwrap_python_model().pipe, mv.run_id, mv.version


def make_classification_dataset(df: pd.DataFrame) -> Dataset:
    """Wrap a pandas frame as an Evidently classification Dataset with
    explicit numeric/categorical/target/prediction roles."""
    definition = DataDefinition(
        numerical_columns=[c for c in NUMERIC if c in df.columns],
        categorical_columns=[c for c in CATEGORICAL if c in df.columns],
        classification=[
            BinaryClassification(
                target="label",
                prediction_labels="prediction",
                prediction_probas="prediction_proba",
                pos_label=1,
            )
        ] if "label" in df.columns else None,
    )
    return Dataset.from_pandas(df, data_definition=definition)


def make_drift_dataset(df: pd.DataFrame) -> Dataset:
    definition = DataDefinition(
        numerical_columns=[c for c in NUMERIC if c in df.columns],
        categorical_columns=[c for c in CATEGORICAL if c in df.columns],
    )
    return Dataset.from_pandas(df, data_definition=definition)


def save(report: Report, filename: str) -> None:
    out = REPORTS_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    size_mb = out.stat().st_size / 1_048_576
    print(f"  → wrote {out.name} ({size_mb:.1f} MB)")


def main() -> None:
    print(f"[mlflow] tracking_uri={TRACKING_URI}")
    pipe, run_id, version = load_pipeline()
    print(f"[model ] ForeRoute-BostonRisk v{version} from run {run_id}")

    df = pd.read_csv(DATA)
    print(f"[data  ] {len(df):,} rows · {len(FEATURE_COLS)} features")

    X = df[FEATURE_COLS]
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[split ] reference (train) = {len(X_train):,}  ·  current (test) = {len(X_test):,}")

    # Attach predictions for both windows so we can drift-compare outputs
    proba_train = pipe.predict_proba(X_train)[:, 1]
    proba_test = pipe.predict_proba(X_test)[:, 1]

    ref = X_train.copy()
    ref["label"] = y_train.values
    ref["prediction_proba"] = proba_train
    ref["prediction"] = (proba_train >= CLASSIFICATION_THRESHOLD).astype(int)

    cur = X_test.copy()
    cur["label"] = y_test.values
    cur["prediction_proba"] = proba_test
    cur["prediction"] = (proba_test >= CLASSIFICATION_THRESHOLD).astype(int)

    print(f"[score ] threshold={CLASSIFICATION_THRESHOLD}")
    print(f"          ref  pos-rate (label) = {y_train.mean():.3f}   pred-pos-rate = {ref['prediction'].mean():.3f}")
    print(f"          cur  pos-rate (label) = {y_test.mean():.3f}   pred-pos-rate = {cur['prediction'].mean():.3f}")

    # ---- 01 Data drift on inputs --------------------------------------
    print("[01    ] data drift across 15 inputs …")
    r1 = Report([DataDriftPreset(columns=FEATURE_COLS)])
    snap1 = r1.run(reference_data=make_drift_dataset(ref[FEATURE_COLS]),
                   current_data=make_drift_dataset(cur[FEATURE_COLS]))
    save(snap1, "01_data_drift.html")

    # ---- 02 Data quality / schema audit -------------------------------
    print("[02    ] data quality summary …")
    r2 = Report([DataSummaryPreset()])
    snap2 = r2.run(reference_data=make_drift_dataset(ref[FEATURE_COLS]),
                   current_data=make_drift_dataset(cur[FEATURE_COLS]))
    save(snap2, "02_data_quality.html")

    # ---- 03 Output drift — drift on the prediction probability --------
    print("[03    ] output drift on prediction probability …")
    out_def = DataDefinition(numerical_columns=["prediction_proba"])
    out_ref = Dataset.from_pandas(ref[["prediction_proba"]], data_definition=out_def)
    out_cur = Dataset.from_pandas(cur[["prediction_proba"]], data_definition=out_def)
    r3 = Report([DataDriftPreset(columns=["prediction_proba"])])
    snap3 = r3.run(reference_data=out_ref, current_data=out_cur)
    save(snap3, "03_output_drift.html")

    # ---- 04 Classification performance --------------------------------
    print(f"[04    ] classification performance @ threshold {CLASSIFICATION_THRESHOLD} …")
    r4 = Report([ClassificationPreset()])
    snap4 = r4.run(reference_data=make_classification_dataset(ref),
                   current_data=make_classification_dataset(cur))
    save(snap4, "04_classification_performance.html")

    # Drop the legacy regression report — current model is classifier-only.
    legacy = REPORTS_DIR / "05_regression_performance.html"
    if legacy.exists():
        legacy.unlink()
        print(f"[clean ] removed legacy {legacy.name} (model is classifier-only)")

    print(f"[done  ] reports in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
