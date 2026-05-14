"""
Five-fold stratified cross-validation for ForeRoute-BostonRisk.

Mirrors the train.py recipe exactly inside each fold: rebalance training
negatives 1.5:1, train Gradient Boosting on rebalanced training, evaluate on
the un-rebalanced test fold (natural 0.25 positive rate). Reports mean and
standard deviation of every metric we report in REPORT.md.

This tells us whether the 0.65 ROC-AUC headline is a stable property of the
model or an artifact of a lucky single split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data" / "boston_crash_dataset_with_aadt.csv"
OUT = PROJECT / "data" / "cv_summary.json"

FEATURE_COLS = [
    "temperature", "precipitation_type", "precipitation_intensity",
    "wind_speed", "visibility", "humidity", "dew_point",
    "road_type", "segment_distance_m",
    "lat", "lon", "hour_of_day", "day_of_week", "month",
    "prior_year_crash_count",
    "traffic_volume",
]
NUMERIC = [c for c in FEATURE_COLS if c not in ("precipitation_type", "road_type")]
CATEGORICAL = ["precipitation_type", "road_type"]

N_SPLITS = 5
THRESHOLD = 0.35
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))])


def rebalance(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df = X.copy(); df["_y"] = y.values
    pos = df[df["_y"] == 1]; neg = df[df["_y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y
    neg_keep = min(len(neg), int(len(pos) * 1.5))
    out = pd.concat([pos, neg.sample(n=neg_keep, random_state=RANDOM_STATE)], axis=0)
    out = out.sample(frac=1.0, random_state=RANDOM_STATE)
    return out[X.columns].copy(), out["_y"].astype(int).copy()


def main() -> None:
    df = pd.read_csv(DATA)
    X = df[FEATURE_COLS]; y = df["label"].astype(int)
    print(f"[data ] n={len(df):,}  pos_rate={y.mean():.3f}")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    cols = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
    rows: list[dict] = []

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        print(f"\n[fold {i}/{N_SPLITS}]")
        X_tr_raw, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_raw, y_te = y.iloc[train_idx], y.iloc[test_idx]
        print(f"  train_raw={len(X_tr_raw):,} (pos={y_tr_raw.mean():.3f})  test={len(X_te):,} (pos={y_te.mean():.3f})")

        X_tr, y_tr = rebalance(X_tr_raw, y_tr_raw)
        print(f"  train_rebal={len(X_tr):,} (pos={y_tr.mean():.3f})")

        pipe = build_pipeline()
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        cm = confusion_matrix(y_te, pred).ravel()

        metrics = {
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "accuracy": float(accuracy_score(y_te, pred)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3]),
        }
        for c in cols:
            print(f"  {c:10s} {metrics[c]:.4f}")
        rows.append(metrics)

    # Aggregate
    print(f"\n[summary across {N_SPLITS} folds, threshold={THRESHOLD}]")
    print(f"{'metric':<12} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    summary = {}
    for c in cols:
        vals = np.array([r[c] for r in rows])
        m, s, lo, hi = float(vals.mean()), float(vals.std()), float(vals.min()), float(vals.max())
        summary[c] = {"mean": m, "std": s, "min": lo, "max": hi, "values": vals.tolist()}
        print(f"{c:<12} {m:>8.4f} {s:>8.4f} {lo:>8.4f} {hi:>8.4f}")

    OUT.write_text(json.dumps({
        "threshold": THRESHOLD,
        "n_splits": N_SPLITS,
        "per_fold": rows,
        "summary": summary,
    }, indent=2))
    print(f"\n[out  ] {OUT}")


if __name__ == "__main__":
    main()
