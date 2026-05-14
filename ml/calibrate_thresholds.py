"""
Calibrate verdict thresholds from quantiles of the registered model's predicted
probability on a **representative sample of what users actually see**:
random in-bbox Boston points, paired with typical right-now conditions and the
runtime crash-density lookup. This matches the population the web app feeds
the model at inference time, NOT the training data (whose negatives are
anchored on crash locations and skew the model's "typical" prediction high).

Output: ml/data/verdict_thresholds.json, loaded by web/lib/mlVerdictServer.ts.

Bucketing:
  · p <  q25  →  lower        (bottom 25%)
  · q25 ≤ p < q75  →  typical   (middle 50%)
  · q75 ≤ p < q90  →  above     (top 25%)
  · p ≥ q90  →  muchHigher  (top 10%)

The median (q50) is also exported and used by the UI to render a ratio
caption ("1.4× a typical Boston road").
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{PROJECT / 'mlflow.db'}"
DATA = PROJECT / "data" / "boston_crash_dataset.csv"
OUT = PROJECT / "data" / "verdict_thresholds.json"
MODEL_URI = "models:/ForeRoute-BostonRisk@production"

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


def crashes_within_radius(
    lat: float, lon: float, crash_lats: np.ndarray, crash_lons: np.ndarray, radius_m: float = 100.0
) -> int:
    """Equirectangular-approximation point-in-radius count over recent crashes."""
    dy = (crash_lats - lat) * 111_111
    dx = (crash_lons - lon) * 111_111 * math.cos(math.radians(lat))
    return int(np.sum(dy * dy + dx * dx <= radius_m * radius_m))


def build_synthetic_samples(crashes_csv: Path, n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Generate `n` random in-bbox samples with typical right-now conditions.
    Mirrors the 15-feature payload the orchestrator sends per segment."""
    rng = np.random.default_rng(seed)
    crashes = pd.read_csv(crashes_csv)
    crashes["ts"] = pd.to_datetime(crashes["ts"], utc=True, format="mixed")
    cutoff = crashes["ts"].max() - pd.Timedelta(days=365)
    recent = crashes[(crashes["label"] == 1) & (crashes["ts"] >= cutoff)]
    crash_lats = recent["lat"].to_numpy()
    crash_lons = recent["lon"].to_numpy()
    print(f"[synth ] {len(recent)} recent crashes used for prior_year_crash_count lookup")

    # Boston bbox (must match TRAINING_BBOX in web/lib/mlVerdict.ts)
    lats = rng.uniform(42.20, 42.45, n)
    lons = rng.uniform(-71.20, -70.95, n)
    now = datetime.now(timezone.utc)
    # Spread hour/dow/month so the synthetic set covers temporal variation
    hours = rng.integers(0, 24, n)
    dows = rng.integers(0, 7, n)
    months = rng.integers(1, 13, n)
    road_choices = rng.choice(
        ["arterial", "residential", "highway"], size=n, p=[0.55, 0.40, 0.05]
    )

    rows = []
    for i in range(n):
        crashes_count = crashes_within_radius(lats[i], lons[i], crash_lats, crash_lons)
        rows.append({
            # "typical Boston May weather" — mild, clear; what the web app sees most of the year
            "temperature": float(rng.uniform(10.0, 22.0)),
            "precipitation_type": "none",
            "precipitation_intensity": 0.0,
            "wind_speed": float(rng.uniform(5.0, 15.0)),
            "visibility": 10.0,
            "humidity": float(rng.uniform(50.0, 80.0)),
            "dew_point": float(rng.uniform(5.0, 15.0)),
            "road_type": road_choices[i],
            "segment_distance_m": 1000.0,
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "hour_of_day": int(hours[i]),
            "day_of_week": int(dows[i]),
            "month": int(months[i]),
            "prior_year_crash_count": int(crashes_count),
        })
    return pd.DataFrame(rows)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"[mlflow] tracking_uri={TRACKING_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"[model ] loaded {MODEL_URI}")

    samples = build_synthetic_samples(DATA, n=500)
    X = samples[FEATURE_COLS]
    print(f"[synth ] {len(samples)} representative samples generated")

    probs = np.asarray(model.predict(X)).flatten().astype(float)
    print(f"[score ] probs: min={probs.min():.4f} mean={probs.mean():.4f} max={probs.max():.4f}")

    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    quantiles = {f"q{int(p*100)}": float(np.quantile(probs, p)) for p in qs}

    thresholds = {
        "lower_max": quantiles["q25"],
        "typical_max": quantiles["q75"],
        "above_max": quantiles["q90"],
        "median": quantiles["q50"],
    }

    payload = {
        "source": f"Synthetic Boston bbox samples with typical conditions (n={len(samples)})",
        "model_alias": MODEL_URI,
        "quantiles": quantiles,
        "thresholds": thresholds,
        "interpretation": {
            "lower": f"probability < {thresholds['lower_max']:.3f}",
            "typical": f"{thresholds['lower_max']:.3f} ≤ probability < {thresholds['typical_max']:.3f}",
            "above": f"{thresholds['typical_max']:.3f} ≤ probability < {thresholds['above_max']:.3f}",
            "muchHigher": f"probability ≥ {thresholds['above_max']:.3f}",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[out   ] wrote {OUT}")
    for k, v in quantiles.items():
        print(f"          {k} = {v:.4f}")


if __name__ == "__main__":
    main()
