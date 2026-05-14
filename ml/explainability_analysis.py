"""
Generate global SHAP feature importance and a confusion matrix / ROC summary
for the registered ForeRoute-BostonRisk@production model. Outputs JSON used by
REPORT.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import shap
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{PROJECT / 'mlflow.db'}"
DATA = PROJECT / "data" / "boston_crash_dataset.csv"
OUT = PROJECT / "data" / "explainability_summary.json"

FEATURE_COLS = [
    "temperature", "precipitation_type", "precipitation_intensity",
    "wind_speed", "visibility", "humidity", "dew_point",
    "road_type", "segment_distance_m",
    "lat", "lon", "hour_of_day", "day_of_week", "month",
    "prior_year_crash_count",
]


def load_sklearn_pipeline():
    """Pull the production-aliased run, then load the underlying sklearn
    Pipeline (the pyfunc wrapper contains it as `self.pipe`)."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias("ForeRoute-BostonRisk", "production")
    run_id = mv.run_id
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    pipe = pyfunc_model.unwrap_python_model().pipe
    return pipe, run_id, mv.version


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    pipe, run_id, version = load_sklearn_pipeline()
    print(f"[model] loaded pipeline from run {run_id}, version v{version}")

    df = pd.read_csv(DATA)
    X = df[FEATURE_COLS]
    y = df["label"].astype(int)

    # Use same test split as train.py for honest reporting
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.285, random_state=42, stratify=y)
    print(f"[data] test set: n={len(X_test)} pos_rate={y_test.mean():.3f}")

    # ---- Predictions and standard metrics -------------------------------
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))
    print(f"[perf ] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"[cm   ] tn={tn} fp={fp} fn={fn} tp={tp}")

    # ROC curve points (downsample)
    fpr, tpr, _ = roc_curve(y_test, proba)
    idx = np.linspace(0, len(fpr) - 1, 50).astype(int)
    roc_points = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]

    # PR curve points (downsample)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    idx = np.linspace(0, len(prec) - 1, 50).astype(int)
    pr_points = [{"recall": float(rec[i]), "precision": float(prec[i])} for i in idx]

    # ---- SHAP global feature importance --------------------------------
    # GB classifier is in pipe.named_steps["clf"]; transform X_test through
    # the preprocessor first.
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_sample = X_test.sample(n=min(1500, len(X_test)), random_state=0)
    X_t = pre.transform(X_sample)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    feat_names = pre.get_feature_names_out().tolist()
    print(f"[shap ] explaining {len(X_sample)} rows over {len(feat_names)} engineered features")

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_t)
    # For binary GB sklearn, shap_values returns ndarray (n, m) for positive class
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    mean_abs = np.mean(np.abs(sv), axis=0)

    # Group one-hot expansions back to original feature names
    grouped: dict[str, float] = {}
    for name, val in zip(feat_names, mean_abs):
        # name is e.g. "num__temperature" or "cat__precipitation_type_snow"
        if "__" in name:
            _, rest = name.split("__", 1)
        else:
            rest = name
        original = rest
        for cat in ("precipitation_type", "road_type"):
            if rest.startswith(cat + "_"):
                original = cat
                break
        grouped[original] = grouped.get(original, 0.0) + float(val)
    ranked = sorted(grouped.items(), key=lambda kv: -kv[1])
    print("[shap ] top contributors (mean |SHAP|):")
    for name, val in ranked[:8]:
        print(f"          {name:32s} {val:.4f}")

    # Built-in feature_importances_ for cross-check
    fi = getattr(clf, "feature_importances_", None)
    if fi is not None:
        fi_grouped: dict[str, float] = {}
        for name, val in zip(feat_names, fi):
            if "__" in name:
                _, rest = name.split("__", 1)
            else:
                rest = name
            original = rest
            for cat in ("precipitation_type", "road_type"):
                if rest.startswith(cat + "_"):
                    original = cat
                    break
            fi_grouped[original] = fi_grouped.get(original, 0.0) + float(val)
        fi_ranked = sorted(fi_grouped.items(), key=lambda kv: -kv[1])
    else:
        fi_ranked = []

    payload = {
        "model_run_id": run_id,
        "model_version": version,
        "test_n": int(len(X_test)),
        "test_pos_rate": float(y_test.mean()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix_at_threshold_0_5": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
        "shap_top_features": [{"feature": k, "mean_abs_shap": v} for k, v in ranked],
        "feature_importances": [{"feature": k, "importance": v} for k, v in fi_ranked],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[out  ] wrote {OUT}")


if __name__ == "__main__":
    main()
