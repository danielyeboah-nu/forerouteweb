"""
Build the `traffic_volume` feature by spatial-joining MassDOT (or Boston Open
Data) AADT counts onto the existing Boston crash dataset.

Why: traffic exposure is the strongest crash-prediction signal we don't have.
Adding AADT should push ROC-AUC from ~0.65 toward the 0.72–0.75 band.

Pipeline:
  1. User drops an AADT CSV at ml/data/massdot_aadt.csv.
     Common sources:
       · MassDOT IMPACT — mass.gov/info-details/massdot-impact-data (export CSV)
       · MassGIS MassDOT Roads — mass.gov/info-details/massgis-data-massdot-roads
       · data.boston.gov Boston Traffic Counts
     Expected columns (tolerant of variants):
       · lat / latitude / y_cord
       · lon / long / longitude / x_cord
       · aadt / AADT / volume / count / traffic_count
       · (optional) year, road_name, route_id
  2. We read the CSV, normalise columns, filter to most recent year per point.
  3. Build a BallTree (haversine) over AADT point locations.
  4. For each training sample, find the nearest AADT point within 200 m.
  5. Samples with no match get the road-type-conditional median AADT.
  6. Write ml/data/boston_crash_dataset_with_aadt.csv with one new column:
       · traffic_volume  (numeric, vehicles/day)
     plus an audit column `aadt_match` ∈ {direct, imputed}.

After running, retrain via `make train-boston` with the new dataset path and
update `FEATURE_COLS` in train.py and lib/mlflow.ts to include traffic_volume.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
AADT_CSV = DATA_DIR / "massdot_aadt.csv"
BOSTON_DATASET = DATA_DIR / "boston_crash_dataset.csv"
OUT_CSV = DATA_DIR / "boston_crash_dataset_with_aadt.csv"

# Match radius: 200 m gives reasonable coverage on most road segments without
# pulling in points from adjacent streets.
MATCH_RADIUS_M = 200.0
EARTH_M = 6_371_000.0

# Boston bbox (same as build_boston_dataset.py)
BBOX_LAT = (42.20, 42.45)
BBOX_LON = (-71.20, -70.95)

# Common column-name variants — picked tolerantly
LAT_KEYS = ("lat", "latitude", "y", "y_cord", "y_coord")
LON_KEYS = ("lon", "lng", "long", "longitude", "x", "x_cord", "x_coord")
AADT_KEYS = ("aadt", "AADT", "volume", "count", "traffic_count", "avg_daily_traffic", "adt")
YEAR_KEYS = ("year", "count_year", "data_year", "yr")


def pick(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def load_aadt(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"[fatal] Drop a MassDOT/Boston AADT CSV at {path}\n"
            f"        See module docstring for accepted sources and columns."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"[load ] {len(df):,} AADT rows from {path.name}")
    print(f"        columns: {list(df.columns)[:14]}{' ...' if len(df.columns) > 14 else ''}")

    lat_col = pick(df, LAT_KEYS)
    lon_col = pick(df, LON_KEYS)
    aadt_col = pick(df, AADT_KEYS)
    year_col = pick(df, YEAR_KEYS)
    if not (lat_col and lon_col and aadt_col):
        sys.exit(
            f"[fatal] Could not find lat/lon/aadt columns. "
            f"Got lat={lat_col} lon={lon_col} aadt={aadt_col}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        "aadt": pd.to_numeric(df[aadt_col], errors="coerce"),
    })
    if year_col:
        out["year"] = pd.to_numeric(df[year_col], errors="coerce")

    out = out.dropna(subset=["lat", "lon", "aadt"])
    out = out[
        out["lat"].between(*BBOX_LAT)
        & out["lon"].between(*BBOX_LON)
        & (out["aadt"] > 0)
    ].reset_index(drop=True)

    # If multiple counts exist for the same location, keep the most recent.
    if "year" in out.columns:
        out = out.sort_values("year").drop_duplicates(
            subset=[out["lat"].round(5), out["lon"].round(5)] if False else ["lat", "lon"],
            keep="last",
        ).reset_index(drop=True)

    print(f"[load ] {len(out):,} valid AADT points in Boston bbox")
    print(f"        AADT: min={out['aadt'].min():,.0f}  median={out['aadt'].median():,.0f}  "
          f"max={out['aadt'].max():,.0f}")
    return out


def attach_aadt(samples: pd.DataFrame, aadt: pd.DataFrame, radius_m: float = MATCH_RADIUS_M) -> pd.DataFrame:
    """For each sample, find the nearest AADT point within `radius_m`.
    Unmatched samples receive the median AADT for their `road_type`."""
    radius_rad = radius_m / EARTH_M

    aadt_rad = np.radians(aadt[["lat", "lon"]].to_numpy())
    sample_rad = np.radians(samples[["lat", "lon"]].to_numpy())

    print(f"[match] BallTree on {len(aadt):,} AADT points · radius {radius_m:.0f} m")
    tree = BallTree(aadt_rad, metric="haversine")
    dist, idx = tree.query(sample_rad, k=1)
    dist_m = dist[:, 0] * EARTH_M
    nearest_idx = idx[:, 0]

    matched = dist_m <= radius_m
    n_matched = int(matched.sum())
    print(f"[match] direct matches: {n_matched:,} / {len(samples):,} "
          f"({100 * n_matched / len(samples):.1f}%)  median dist {np.median(dist_m[matched]):.0f} m")

    traffic_volume = np.full(len(samples), np.nan)
    traffic_volume[matched] = aadt["aadt"].to_numpy()[nearest_idx[matched]]

    # Imputation by road_type median (computed only on matched rows so unmatched
    # samples don't contaminate the imputation distribution).
    if "road_type" in samples.columns:
        matched_df = samples.loc[matched, ["road_type"]].copy()
        matched_df["aadt"] = traffic_volume[matched]
        rt_median = matched_df.groupby("road_type")["aadt"].median().to_dict()
        global_median = float(np.nanmedian(traffic_volume[matched])) if matched.any() else 0.0
        for i in np.where(~matched)[0]:
            rt = samples["road_type"].iloc[i]
            traffic_volume[i] = rt_median.get(rt, global_median)
        print(f"[match] imputed {(~matched).sum():,} unmatched samples via road_type median")
        print(f"        per-road-type median AADT:")
        for rt, med in sorted(rt_median.items(), key=lambda kv: -kv[1]):
            print(f"          {rt:<14s} {med:>8,.0f}")
    else:
        global_median = float(np.nanmedian(traffic_volume[matched])) if matched.any() else 0.0
        traffic_volume[np.isnan(traffic_volume)] = global_median
        print(f"[match] no road_type column — imputed unmatched with global median {global_median:,.0f}")

    out = samples.copy()
    out["traffic_volume"] = traffic_volume.astype(float)
    out["aadt_match"] = np.where(matched, "direct", "imputed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aadt", type=Path, default=AADT_CSV)
    parser.add_argument("--samples", type=Path, default=BOSTON_DATASET)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--radius-m", type=float, default=MATCH_RADIUS_M)
    args = parser.parse_args()

    aadt = load_aadt(args.aadt)
    samples = pd.read_csv(args.samples)
    print(f"[load ] {len(samples):,} training samples from {args.samples.name}")

    if "traffic_volume" in samples.columns:
        print("[warn ] samples already have a `traffic_volume` column — it will be overwritten")
        samples = samples.drop(columns=["traffic_volume", "aadt_match"], errors="ignore")

    enriched = attach_aadt(samples, aadt, radius_m=args.radius_m)

    print(f"[out  ] traffic_volume distribution:")
    tv = enriched["traffic_volume"]
    print(f"          min     {tv.min():>10,.0f}")
    print(f"          p25     {tv.quantile(0.25):>10,.0f}")
    print(f"          median  {tv.median():>10,.0f}")
    print(f"          p75     {tv.quantile(0.75):>10,.0f}")
    print(f"          p95     {tv.quantile(0.95):>10,.0f}")
    print(f"          max     {tv.max():>10,.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out, index=False)
    print(f"[out  ] wrote {args.out}  ({len(enriched):,} rows, "
          f"{(enriched['aadt_match'] == 'direct').sum():,} directly matched)")


if __name__ == "__main__":
    main()
