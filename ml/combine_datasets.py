"""
Concatenate the inter-city segment dataset and the Boston crash-labeled dataset
into a single training file with a unified `label` column.

  - fore_route_segments_large.csv has `segment_risk_binary` (synthetic label).
  - boston_crash_dataset.csv has `label` (real crash label).

We rename `segment_risk_binary` → `label`, tag each row with `dataset_source`
so train.py / downstream analysis can stratify if useful, and emit
data/combined_segments.csv. The output uses exactly the model's FEATURE_COLS
plus `label` and `dataset_source`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train import build_label as build_synthetic_label

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

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
]


def load_intercity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The `segment_risk_binary` column in this CSV is always 0; the real
    # synthetic hazard label is built dynamically in train.py from the
    # features. Rebuild it here so the row carries actual signal.
    label = build_synthetic_label(df).astype(int)
    out = df[FEATURE_COLS].copy()
    out["label"] = label.values
    out["dataset_source"] = "intercity_synth"
    return out


def load_boston(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        sys.exit(f"[fatal] {path.name} missing `label` column")
    out = df[FEATURE_COLS].copy()
    out["label"] = df["label"].astype(int)
    out["dataset_source"] = "boston_real"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intercity", type=Path, default=DATA_DIR / "fore_route_segments_large.csv")
    parser.add_argument("--boston", type=Path, default=DATA_DIR / "boston_crash_dataset.csv")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "combined_segments.csv")
    args = parser.parse_args()

    parts: list[pd.DataFrame] = []
    if args.intercity.exists():
        ic = load_intercity(args.intercity)
        print(f"[ic ] {len(ic):,} rows · pos_rate={ic.label.mean():.3f}")
        parts.append(ic)
    else:
        print(f"[ic ] skipping — {args.intercity} not found")

    if args.boston.exists():
        bo = load_boston(args.boston)
        print(f"[bos] {len(bo):,} rows · pos_rate={bo.label.mean():.3f}")
        parts.append(bo)
    else:
        sys.exit(
            f"[fatal] {args.boston} not found. Run `make boston-data` first to "
            f"build the Boston crash-labeled dataset."
        )

    if not parts:
        sys.exit("[fatal] no datasets to combine")

    combined = pd.concat(parts, axis=0, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    by_src = combined.groupby("dataset_source").agg(
        rows=("label", "size"), pos_rate=("label", "mean")
    )
    print(f"[out] {len(combined):,} rows total · pos_rate={combined.label.mean():.3f}")
    print(by_src)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
