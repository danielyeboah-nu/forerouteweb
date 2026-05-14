"""
Build a real-label training dataset from Boston Vision Zero crash records.

Steps:
  1. Load ml/data/boston_crashes.csv (user-provided).
  2. Normalize columns (tolerant of common Vision Zero schemas).
  3. Quantize space into a coarse grid; fetch hourly historical weather per
     unique cell from Open-Meteo's free archive API, cached.
  4. For each crash, look up road_type via Mapbox tilequery, with on-disk cache.
  5. Generate negative samples in the Boston bbox by spatial+temporal rejection
     sampling (must be >RADIUS m and >TIME_DELTA from any positive crash).
  6. Emit ml/data/boston_crash_dataset.csv with the same 9 model features as
     fore_route_segments_large.csv plus a real `label` column.

Run:
  cd ml && python3 build_boston_dataset.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree
from tqdm import tqdm

# ---- paths & constants ------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CRASHES_CSV = DATA_DIR / "boston_crashes.csv"
OUT_CSV = DATA_DIR / "boston_crash_dataset.csv"
DENSITY_JSON = DATA_DIR / "boston_crash_density.json"
RECENT_CRASHES_JSON = DATA_DIR / "boston_recent_crashes.json"
CACHE_DIR = PROJECT_ROOT / ".cache"
WEATHER_CACHE = CACHE_DIR / "openmeteo"
TILEQUERY_CACHE = CACHE_DIR / "tilequery.json"

# Prior-year crash density: count crashes within this radius and within the
# preceding 365 days of each sample's timestamp.
PRIOR_RADIUS_M = 100.0
PRIOR_WINDOW_SEC = 365 * 86400
# Crash-density JSON for inference: bucket lat/lon at 3 decimals (~110 m).
DENSITY_BUCKET_DECIMALS = 3

# Boston bbox (a generous metro envelope)
BBOX_LAT = (42.20, 42.45)
BBOX_LON = (-71.20, -70.95)
# Grid cell size for weather caching: ~0.025° ≈ 2.5 km
GRID_DEG = 0.025
# Negative-sampling rejection thresholds
NEG_RADIUS_M = 150.0
NEG_TIME_DELTA = timedelta(hours=2)
NEG_PER_POS = 3
RANDOM_SEED = 42

MAPBOX_TOKEN = os.environ.get("NEXT_PUBLIC_MAPBOX_TOKEN") or os.environ.get("MAPBOX_TOKEN")

# ---- crash CSV normalization -----------------------------------------------

# Common Vision Zero column name variants → canonical
LAT_KEYS = ("lat", "latitude", "y")
LON_KEYS = ("long", "lng", "lon", "longitude", "x")
TIME_KEYS = ("dispatch_ts", "crashdate", "crash_date", "crashdatetime", "crash_datetime", "incident_date", "datetime")


def pick(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    return None


def load_crashes(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"[fatal] Drop the Boston Vision Zero CSV at {path}\n"
            f"        (data.boston.gov → Vision Zero Crash Records → CSV)."
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"[load] {len(df)} crash rows from {path.name}")
    print(f"       columns: {list(df.columns)[:14]}{' ...' if len(df.columns) > 14 else ''}")

    lat_col = pick(df, LAT_KEYS)
    lon_col = pick(df, LON_KEYS)
    time_col = pick(df, TIME_KEYS)
    if not (lat_col and lon_col and time_col):
        sys.exit(
            f"[fatal] Could not find lat/lng/time columns. "
            f"Got lat={lat_col} lon={lon_col} time={time_col}. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "ts": pd.to_datetime(df[time_col], errors="coerce", utc=True),
        }
    ).dropna()
    out = out[
        out["lat"].between(*BBOX_LAT)
        & out["lon"].between(*BBOX_LON)
        & (out["ts"] >= pd.Timestamp("2018-01-01", tz="UTC"))
        & (out["ts"] <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7))
    ].reset_index(drop=True)
    print(f"[load] {len(out)} crashes after bbox+time filter ({lat_col}/{lon_col}/{time_col})")
    return out


# ---- open-meteo weather -----------------------------------------------------

OM_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_HOURLY = ",".join(
    [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "snowfall",
        "rain",
        "wind_speed_10m",
        "visibility",
        "weather_code",
    ]
)


def cell_key(lat: float, lon: float) -> tuple[float, float]:
    return (
        round(math.floor(lat / GRID_DEG) * GRID_DEG + GRID_DEG / 2, 6),
        round(math.floor(lon / GRID_DEG) * GRID_DEG + GRID_DEG / 2, 6),
    )


def weather_cache_path(lat: float, lon: float, start: str, end: str) -> Path:
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    return WEATHER_CACHE / f"{lat:.4f}_{lon:.4f}_{start}_{end}.json"


def fetch_weather_cell(
    lat: float, lon: float, start: str, end: str, max_attempts: int = 6
) -> pd.DataFrame:
    path = weather_cache_path(lat, lon, start, end)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": OM_HOURLY,
            "wind_speed_unit": "kmh",
            "timezone": "UTC",
        }
        last_err: str | None = None
        for attempt in range(max_attempts):
            try:
                r = requests.get(OM_URL, params=params, timeout=90)
                if r.status_code == 200:
                    data = r.json()
                    path.write_text(json.dumps(data))
                    break
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = repr(e)
            # exponential backoff with jitter — Open-Meteo throttles bursts hard
            time.sleep(2 ** attempt + 0.5 * attempt)
        else:
            raise RuntimeError(f"open-meteo failed for cell ({lat},{lon}): {last_err}")
    h = data.get("hourly") or {}
    if not h.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def precip_type_from(code: int, snow_mm: float, rain_mm: float) -> str:
    # WMO weather codes → our enum
    if code in (71, 73, 75, 77, 85, 86) or snow_mm > 0:
        return "snow"
    if code in (66, 67):
        return "freezingRain"
    if code in (56, 57):
        return "sleet"
    if (51 <= code <= 67) or (80 <= code <= 82) or rain_mm > 0:
        return "rain"
    return "none"


def attach_weather(
    samples: pd.DataFrame, weather_by_cell: dict[tuple[float, float], pd.DataFrame]
) -> pd.DataFrame:
    out = []
    for _, row in samples.iterrows():
        cell = cell_key(row["lat"], row["lon"])
        wdf = weather_by_cell.get(cell)
        if wdf is None or wdf.empty:
            continue
        # Snap to nearest hour
        ts_hour = row["ts"].floor("h")
        match = wdf[wdf["time"] == ts_hour]
        if match.empty:
            continue
        m = match.iloc[0]
        snow = float(m.get("snowfall") or 0.0) * 10.0  # cm → mm
        rain = float(m.get("rain") or 0.0)
        ptype = precip_type_from(int(m.get("weather_code") or 0), snow, rain)
        intensity = snow if ptype in ("snow", "sleet", "freezingRain") else rain
        vis = m.get("visibility")
        vis_km = (float(vis) / 1000.0) if pd.notna(vis) else 10.0
        out.append(
            {
                "lat": row["lat"],
                "lon": row["lon"],
                "ts": row["ts"],
                "temperature": float(m.get("temperature_2m")),
                "precipitation_type": ptype,
                "precipitation_intensity": float(intensity),
                "wind_speed": float(m.get("wind_speed_10m") or 0.0),
                "visibility": vis_km,
                "humidity": float(m.get("relative_humidity_2m") or 0.0),
                "dew_point": float(m.get("dew_point_2m") or 0.0),
                "label": int(row["label"]),
            }
        )
    return pd.DataFrame(out)


# ---- mapbox tilequery for road_type -----------------------------------------

TQ_URL = "https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/tilequery"


def _load_tilequery_cache() -> dict[str, str]:
    if TILEQUERY_CACHE.exists():
        return json.loads(TILEQUERY_CACHE.read_text())
    return {}


def _save_tilequery_cache(cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TILEQUERY_CACHE.write_text(json.dumps(cache))


def _road_class_to_type(props: dict) -> str:
    structure = (props.get("structure") or "").lower()
    if structure == "bridge":
        return "bridge"
    if structure == "tunnel":
        return "tunnel"
    cls = (props.get("class") or "").lower()
    if cls in ("motorway", "trunk"):
        return "highway"
    if cls in ("primary", "secondary"):
        return "arterial"
    if cls in ("tertiary", "street", "street_limited", "service", "residential"):
        return "residential"
    return "arterial"


def _tilequery_one(lat: float, lon: float, token: str) -> str:
    url = f"{TQ_URL}/{lon},{lat}.json"
    params = {
        "radius": 25,
        "limit": 5,
        "dedupe": "true",
        "layers": "road",
        "access_token": token,
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return "arterial"
    feats = r.json().get("features") or []
    if not feats:
        return "arterial"
    # Pick highest-class road in the result set
    rank = {"highway": 4, "arterial": 3, "residential": 2, "tunnel": 1, "bridge": 0}
    types = [_road_class_to_type(f.get("properties") or {}) for f in feats]
    return max(types, key=lambda t: rank.get(t, 0))


def classify_road_types(samples: pd.DataFrame, token: str | None) -> pd.Series:
    if not token:
        print("[road] no Mapbox token — defaulting all to 'arterial'")
        return pd.Series(["arterial"] * len(samples), index=samples.index)
    cache = _load_tilequery_cache()

    def key(lat: float, lon: float) -> str:
        return f"{round(lat, 4)},{round(lon, 4)}"

    # Dedupe unique snapped points
    keys = [key(row.lat, row.lon) for row in samples.itertuples()]
    unique = list({k for k in keys if k not in cache})
    print(f"[road] {len(samples)} samples · {len(unique)} unique points to fetch · {len(cache)} cached")

    def fetch_one(k: str) -> tuple[str, str]:
        lat_str, lon_str = k.split(",")
        return k, _tilequery_one(float(lat_str), float(lon_str), token)

    if unique:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_one, k) for k in unique]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="tilequery"):
                k, t = fut.result()
                cache[k] = t
        _save_tilequery_cache(cache)
    return pd.Series([cache[k] for k in keys], index=samples.index)


# ---- negative sampling ------------------------------------------------------


@dataclass
class CrashIndex:
    lats: np.ndarray
    lons: np.ndarray
    ts: np.ndarray  # int64 seconds

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "CrashIndex":
        return cls(
            df["lat"].to_numpy(),
            df["lon"].to_numpy(),
            df["ts"].astype("int64").to_numpy() // 1_000_000_000,
        )

    def collides(self, lat: float, lon: float, ts_sec: int) -> bool:
        # Approximate haversine in meters (Boston-scale OK)
        dlat = (self.lats - lat) * 111_111
        dlon = (self.lons - lon) * 111_111 * math.cos(math.radians(lat))
        dist = np.sqrt(dlat * dlat + dlon * dlon)
        within_space = dist < NEG_RADIUS_M
        within_time = np.abs(self.ts - ts_sec) < NEG_TIME_DELTA.total_seconds()
        return bool(np.any(within_space & within_time))


def generate_negatives(positives: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Anchor each negative on a randomly chosen positive's location (with small
    jitter so it isn't an exact dupe), then resample its timestamp to one that
    is *not* close to any actual crash. This keeps the road_type distribution
    of negatives close to positives' — preventing the model from learning a
    spurious 'residential roads = no crash' shortcut from uniform bbox samples.
    """
    rng = np.random.default_rng(seed)
    idx = CrashIndex.from_df(positives)
    t0 = positives["ts"].min().to_pydatetime().replace(tzinfo=timezone.utc)
    t1 = positives["ts"].max().to_pydatetime().replace(tzinfo=timezone.utc)
    span = (t1 - t0).total_seconds()
    pos_lat = positives["lat"].to_numpy()
    pos_lon = positives["lon"].to_numpy()

    rows: list[dict] = []
    rejected = 0
    pbar = tqdm(total=n, desc="negatives")
    while len(rows) < n:
        i = int(rng.integers(len(positives)))
        lat = float(pos_lat[i] + rng.normal(0, 0.0005))  # ~55 m σ
        lon = float(pos_lon[i] + rng.normal(0, 0.0005))
        if not (BBOX_LAT[0] <= lat <= BBOX_LAT[1] and BBOX_LON[0] <= lon <= BBOX_LON[1]):
            rejected += 1
            continue
        ts = t0 + timedelta(seconds=float(rng.uniform(0, span)))
        if idx.collides(lat, lon, int(ts.timestamp())):
            rejected += 1
            continue
        rows.append({"lat": lat, "lon": lon, "ts": pd.Timestamp(ts), "label": 0})
        pbar.update(1)
    pbar.close()
    print(f"[neg ] generated {n} negatives, rejected {rejected}")
    return pd.DataFrame(rows)


# ---- prior-year crash density ----------------------------------------------


def compute_prior_year_crash_count(
    samples: pd.DataFrame, crashes: pd.DataFrame
) -> np.ndarray:
    """For each sample, count `crashes` within PRIOR_RADIUS_M and within the
    PRIOR_WINDOW_SEC strictly *before* that sample's timestamp. Leakage-safe:
    a sample never sees its own time or later.
    """
    crash_rad = np.radians(crashes[["lat", "lon"]].to_numpy())
    crash_ts = crashes["ts"].astype("int64").to_numpy() // 1_000_000_000
    tree = BallTree(crash_rad, metric="haversine")
    radius_rad = PRIOR_RADIUS_M / 6_371_000.0

    sample_rad = np.radians(samples[["lat", "lon"]].to_numpy())
    sample_ts = samples["ts"].astype("int64").to_numpy() // 1_000_000_000

    neighbor_lists = tree.query_radius(sample_rad, r=radius_rad)
    counts = np.zeros(len(samples), dtype=np.int32)
    for i, idx in enumerate(neighbor_lists):
        if len(idx) == 0:
            continue
        ts_neighbors = crash_ts[idx]
        prior = (ts_neighbors < sample_ts[i]) & (ts_neighbors >= sample_ts[i] - PRIOR_WINDOW_SEC)
        counts[i] = int(prior.sum())
    return counts


def export_recent_crashes(crashes: pd.DataFrame, out_path: Path, window_days: int = 365) -> None:
    """Export crash (lat, lon) points from the last `window_days` of the dataset
    so the web side can do an exact haversine scan at inference time, matching
    the leakage-safe training feature's semantics. Cleaner than bucket lookup:
    no precision loss on coarse grids, no empty-cell misses, just point-in-radius.
    """
    cutoff = crashes["ts"].max() - pd.Timedelta(days=window_days)
    recent = crashes[crashes["ts"] >= cutoff].copy()
    pts = recent[["lat", "lon"]].round(5).values.tolist()
    payload = {
        "window_days": window_days,
        "ref_max_ts": str(crashes["ts"].max()),
        "n_points": len(pts),
        "points": pts,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    print(f"[recent] wrote {out_path} · {len(pts)} crashes in last {window_days} days")


def export_density_json(crashes: pd.DataFrame, out_path: Path) -> None:
    """Write a {bucket_key: per_year_crash_count} JSON used at inference for the
    `prior_year_crash_count` feature. Bucket = (round(lat, 3), round(lon, 3)) ~110 m.
    Counts are total crashes per bucket divided by years_spanned, so the field
    represents an approximate annual rate.
    """
    span_years = max(
        1.0, (crashes["ts"].max() - crashes["ts"].min()).total_seconds() / (365 * 86400)
    )
    fmt = f"{{:.{DENSITY_BUCKET_DECIMALS}f}}"
    lat_b = crashes["lat"].round(DENSITY_BUCKET_DECIMALS).map(fmt.format)
    lon_b = crashes["lon"].round(DENSITY_BUCKET_DECIMALS).map(fmt.format)
    bucket = lat_b + "," + lon_b
    counts = bucket.value_counts()
    density = (counts / span_years).round(3).to_dict()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {
            "bucket_decimals": DENSITY_BUCKET_DECIMALS,
            "span_years": round(span_years, 2),
            "counts_per_year": density,
        }
    ))
    print(f"[dens] wrote {out_path} · {len(density)} cells · span {span_years:.1f} years")


# ---- driver -----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crashes", type=Path, default=CRASHES_CSV)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--max-crashes", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    crashes = load_crashes(args.crashes)
    if args.max_crashes > 0:
        crashes = crashes.sample(n=min(args.max_crashes, len(crashes)), random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[load] subsampled to {len(crashes)} crashes")
    crashes["label"] = 1

    negatives = generate_negatives(crashes, n=NEG_PER_POS * len(crashes))
    samples = pd.concat([crashes[["lat", "lon", "ts", "label"]], negatives], axis=0, ignore_index=True)

    # ---- weather build (one Open-Meteo call per unique cell) ---
    unique_cells = sorted({cell_key(r.lat, r.lon) for r in samples.itertuples()})
    print(f"[wx  ] {len(unique_cells)} unique cells to fetch")
    start = (samples["ts"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (samples["ts"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    weather_by_cell: dict[tuple[float, float], pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(fetch_weather_cell, lat, lon, start, end): (lat, lon) for lat, lon in unique_cells}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="open-meteo"):
            cell = futures[fut]
            try:
                weather_by_cell[cell] = fut.result()
            except Exception as e:
                print(f"[wx  ] cell {cell} failed: {e}")
                weather_by_cell[cell] = pd.DataFrame()

    # Retry pass: try any still-empty cells sequentially with extra delay
    failed = [c for c, df in weather_by_cell.items() if df.empty]
    if failed:
        print(f"[wx  ] retrying {len(failed)} failed cells sequentially with longer backoff")
        for cell in failed:
            lat, lon = cell
            try:
                time.sleep(5)
                weather_by_cell[cell] = fetch_weather_cell(lat, lon, start, end, max_attempts=8)
                print(f"[wx  ] recovered cell {cell}")
            except Exception as e:
                print(f"[wx  ] cell {cell} still failing: {e}")

    with_wx = attach_weather(samples, weather_by_cell)
    print(f"[wx  ] {len(with_wx)} of {len(samples)} samples got weather rows")
    if len(with_wx) == 0:
        sys.exit("[fatal] no samples survived weather attachment — abort.")

    # ---- road_type --------------------------------------------
    with_wx["road_type"] = classify_road_types(with_wx[["lat", "lon"]], MAPBOX_TOKEN)

    # ---- prior-year crash density (leakage-safe) ---------------
    # Use ONLY positives (real crash events) as the crash population.
    positives_only = with_wx[with_wx["label"] == 1].reset_index(drop=True)
    print(f"[prior] computing prior-year crash count for {len(with_wx)} samples vs {len(positives_only)} crashes")
    with_wx = with_wx.reset_index(drop=True)
    with_wx["prior_year_crash_count"] = compute_prior_year_crash_count(with_wx, positives_only)
    print(
        f"[prior] mean={with_wx['prior_year_crash_count'].mean():.2f} "
        f"max={with_wx['prior_year_crash_count'].max()} "
        f"nonzero={(with_wx['prior_year_crash_count'] > 0).sum()}"
    )

    # ---- temporal features ------------------------------------
    ts = pd.to_datetime(with_wx["ts"], utc=True)
    with_wx["hour_of_day"] = ts.dt.hour.astype(int)
    with_wx["day_of_week"] = ts.dt.dayofweek.astype(int)
    with_wx["month"] = ts.dt.month.astype(int)

    # ---- finalize feature set ---------------------------------
    with_wx["segment_distance_m"] = 1000.0
    out_cols = [
        "lat",
        "lon",
        "ts",
        "temperature",
        "precipitation_type",
        "precipitation_intensity",
        "wind_speed",
        "visibility",
        "humidity",
        "dew_point",
        "road_type",
        "segment_distance_m",
        "hour_of_day",
        "day_of_week",
        "month",
        "prior_year_crash_count",
        "label",
    ]
    final = with_wx[out_cols].copy()
    pos = int((final["label"] == 1).sum())
    neg = int((final["label"] == 0).sum())
    print(f"[out ] {len(final)} rows · positives={pos} negatives={neg} pos_rate={pos / len(final):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.out, index=False)
    print(f"[out ] wrote {args.out}")

    # ---- inference artifacts ----------------------------------
    export_density_json(positives_only, DENSITY_JSON)
    export_recent_crashes(positives_only, RECENT_CRASHES_JSON, window_days=365)


if __name__ == "__main__":
    main()
