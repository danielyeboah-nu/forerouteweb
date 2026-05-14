"""Generate monitoring_assignment.ipynb for the ForeRoute-BostonRisk assignment."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "monitoring_assignment.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(md("""# ForeRoute — Model Monitoring with Evidently AI

**Assignment Part 2:** apply model and data monitoring techniques to the ForeRoute prediction application.

This notebook follows the same pattern as Evidently's bicycle-demand monitoring example: define a stable **reference** window, define a **current** window, run Evidently reports, and save HTML artifacts that can be attached to the submission.

**Use case:** `ForeRoute-BostonRisk`, a Boston crash-history classifier served through MLflow and used by the Next.js route planner.

**Reports produced**
- `reports/01_data_drift.html`
- `reports/02_data_quality.html`
- `reports/03_output_drift.html`
- `reports/04_classification_performance.html`
"""))

cells.append(md("""## 1. Setup

If you are running this outside the project environment, install the ML dependencies first:

```bash
pip install -r requirements.txt
```
"""))

cells.append(code("""from __future__ import annotations

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
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset

warnings.filterwarnings("ignore")

PROJECT = Path.cwd()
TRACKING_URI = f"sqlite:///{PROJECT / 'mlflow.db'}"
DATA = PROJECT / "data" / "boston_crash_dataset.csv"
REPORTS_DIR = PROJECT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "temperature", "precipitation_type", "precipitation_intensity",
    "wind_speed", "visibility", "humidity", "dew_point",
    "road_type", "segment_distance_m",
    "lat", "lon", "hour_of_day", "day_of_week", "month",
    "prior_year_crash_count",
]
NUMERIC = [c for c in FEATURE_COLS if c not in ("precipitation_type", "road_type")]
CATEGORICAL = ["precipitation_type", "road_type"]
CLASSIFICATION_THRESHOLD = 0.35
"""))

cells.append(md("""## 2. Load Production Model and Dataset

The monitored model is the MLflow production alias `ForeRoute-BostonRisk@production`. It returns a positive-class probability for each road segment.
"""))

cells.append(code("""mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()
model_version = client.get_model_version_by_alias("ForeRoute-BostonRisk", "production")
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{model_version.run_id}/model")
pipe = pyfunc_model.unwrap_python_model().pipe

df = pd.read_csv(DATA)
X = df[FEATURE_COLS]
y = df["label"].astype(int)

print("model version:", model_version.version)
print("run id:", model_version.run_id)
print("rows:", len(df), "positive rate:", round(y.mean(), 3))
"""))

cells.append(md("""## 3. Reference and Current Windows

For the assignment snapshot:

- **Reference** = stratified training split from the Boston crash dataset.
- **Current** = held-out stratified split, standing in for recent production traffic.

In production, `current` would be replaced by the last 7 days of logged `/api/routes` segment-scoring payloads.
"""))

cells.append(code("""X_ref, X_cur, y_ref, y_cur = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

ref_proba = pipe.predict_proba(X_ref)[:, 1]
cur_proba = pipe.predict_proba(X_cur)[:, 1]
ref_pred = (ref_proba >= CLASSIFICATION_THRESHOLD).astype(int)
cur_pred = (cur_proba >= CLASSIFICATION_THRESHOLD).astype(int)

reference = X_ref.copy()
reference["label"] = y_ref.values
reference["prediction_proba"] = ref_proba
reference["prediction"] = ref_pred

current = X_cur.copy()
current["label"] = y_cur.values
current["prediction_proba"] = cur_proba
current["prediction"] = cur_pred

print("reference rows:", len(reference), "current rows:", len(current))
print("reference pred-positive rate:", round(reference["prediction"].mean(), 3))
print("current pred-positive rate:", round(current["prediction"].mean(), 3))
"""))

cells.append(md("""## 4. Evidently Dataset Definitions

The typed `Dataset` definitions tell Evidently which columns are numerical features, categorical features, targets, prediction labels, and prediction probabilities.
"""))

cells.append(code("""def make_drift_dataset(frame: pd.DataFrame) -> Dataset:
    definition = DataDefinition(
        numerical_columns=[c for c in NUMERIC if c in frame.columns],
        categorical_columns=[c for c in CATEGORICAL if c in frame.columns],
    )
    return Dataset.from_pandas(frame, data_definition=definition)


def make_classification_dataset(frame: pd.DataFrame) -> Dataset:
    definition = DataDefinition(
        numerical_columns=[c for c in NUMERIC if c in frame.columns] + ["prediction_proba"],
        categorical_columns=[c for c in CATEGORICAL if c in frame.columns],
        classification=[
            BinaryClassification(
                target="label",
                prediction_labels="prediction",
                prediction_probas="prediction_proba",
                pos_label=1,
            )
        ],
    )
    return Dataset.from_pandas(frame, data_definition=definition)


ref_features = make_drift_dataset(reference[FEATURE_COLS])
cur_features = make_drift_dataset(current[FEATURE_COLS])
ref_scored = make_classification_dataset(reference)
cur_scored = make_classification_dataset(current)
"""))

cells.append(md("""## 5. Input Data Drift

This checks covariate shift: whether `P(X)` changed between the reference and current windows. For ForeRoute, this catches seasonal weather shifts, changed road-type mix, changed geographic coverage, and feature-pipeline bugs.
"""))

cells.append(code("""report = Report([DataDriftPreset(columns=FEATURE_COLS)])
snapshot = report.run(reference_data=ref_features, current_data=cur_features)
out = REPORTS_DIR / "01_data_drift.html"
snapshot.save_html(str(out))
print("wrote", out)
snapshot
"""))

cells.append(md("""## 6. Data Quality and Anomaly Checks

Evidently summarizes nulls, ranges, duplicate rows, and schema issues. The custom checks below model the production boundary checks we would add before sending features to MLflow.
"""))

cells.append(code("""report = Report([DataSummaryPreset()])
snapshot = report.run(reference_data=ref_features, current_data=cur_features)
out = REPORTS_DIR / "02_data_quality.html"
snapshot.save_html(str(out))
print("wrote", out)
"""))

cells.append(code("""def anomaly_checks(frame: pd.DataFrame) -> pd.DataFrame:
    issues = pd.DataFrame(index=frame.index)
    issues["temp_out_of_range"] = (frame["temperature"] < -50) | (frame["temperature"] > 60)
    issues["humidity_out_of_range"] = (frame["humidity"] < 0) | (frame["humidity"] > 100)
    issues["visibility_negative"] = frame["visibility"] < 0
    issues["distance_non_positive"] = frame["segment_distance_m"] <= 0
    issues["unknown_precipitation_type"] = ~frame["precipitation_type"].isin(
        ["none", "rain", "snow", "sleet", "freezingRain"]
    )
    issues["unknown_road_type"] = ~frame["road_type"].isin(
        ["highway", "arterial", "residential", "bridge", "tunnel", "mountain"]
    )
    issues["any"] = issues.any(axis=1)
    return issues


ref_anomalies = anomaly_checks(reference)
cur_anomalies = anomaly_checks(current)
summary = pd.DataFrame({
    "reference_count": ref_anomalies.drop(columns=["any"]).sum(),
    "current_count": cur_anomalies.drop(columns=["any"]).sum(),
})
print("reference anomaly rate:", round(ref_anomalies["any"].mean(), 4))
print("current anomaly rate:", round(cur_anomalies["any"].mean(), 4))
summary
"""))

cells.append(md("""## 7. Model Output Drift

This checks whether the model's probability distribution changed. Output drift can reveal label shift, seasonal risk changes, or feature plumbing regressions even before delayed crash labels arrive.
"""))

cells.append(code("""out_def = DataDefinition(numerical_columns=["prediction_proba"])
out_ref = Dataset.from_pandas(reference[["prediction_proba"]], data_definition=out_def)
out_cur = Dataset.from_pandas(current[["prediction_proba"]], data_definition=out_def)

report = Report([DataDriftPreset(columns=["prediction_proba"])])
snapshot = report.run(reference_data=out_ref, current_data=out_cur)
out = REPORTS_DIR / "03_output_drift.html"
snapshot.save_html(str(out))
print("wrote", out)
print("reference mean probability:", round(reference["prediction_proba"].mean(), 3))
print("current mean probability:", round(current["prediction_proba"].mean(), 3))
"""))

cells.append(md("""## 8. Classification Performance Drift

When labels are available, we monitor model outcome quality directly: confusion matrix, precision, recall, ROC-AUC, PR-AUC, and class-level performance.
"""))

cells.append(code("""report = Report([ClassificationPreset()])
snapshot = report.run(reference_data=ref_scored, current_data=cur_scored)
out = REPORTS_DIR / "04_classification_performance.html"
snapshot.save_html(str(out))
print("wrote", out)
snapshot
"""))

cells.append(md("""## 9. Operational Policy

Nightly production job:

1. Read the last 7 days of scored segment payloads from API logs.
2. Run the four reports in this notebook against the fixed training reference.
3. Save HTML reports and emit scalar summaries to MLflow.
4. Alert when feature or output PSI exceeds 0.25, or when backfilled PR-AUC/recall drops by at least 5 percentage points.

Retraining happens quarterly by default, or sooner after sustained drift or confirmed performance decay.
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(cells)} cells)")
