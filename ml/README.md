# ForeRouteWeb — ML pipeline

Trains three classifiers on the ForeRoute segment dataset, logs runs to MLflow, registers the best as `ForeRoute-SegmentRisk`, and serves it for the Next.js app.

## Layout

```
ml/
├── data/                          # CSVs copied from the iOS project
│   ├── fore_route_segments.csv
│   └── fore_route_segments_large.csv
├── train.py                       # entry point
├── requirements.txt
├── Makefile
├── mlflow.db                      # (created) SQLite tracking backend
└── mlruns/, mlartifacts/          # (created) MLflow artifacts
```

## Commands

```bash
make install     # pip install -r requirements.txt
make train       # train + log + register best as ForeRoute-SegmentRisk@production
make mlflow-ui   # open MLflow UI on http://localhost:5000
make serve       # mlflow models serve on http://localhost:5001
make smoke       # curl the /invocations endpoint with a sample payload
```

## Feature contract

The served model expects a `dataframe_split` body with these columns:

| column                    | type   | example   |
|---------------------------|--------|-----------|
| `temperature`             | float  | -3.0      |
| `precipitation_type`      | string | snow      |
| `precipitation_intensity` | float  | 4.2       |
| `wind_speed`              | float  | 28.0      |
| `visibility`              | float  | 0.6       |
| `humidity`                | float  | 90.0      |
| `dew_point`               | float  | -4.0      |
| `road_type`               | string | bridge    |
| `segment_distance_m`      | float  | 5000.0    |

Response: `[0|1]` per row (predicted hazardous flag).

## How the label is built

The training label `risky_condition_label` is sampled from a hazard probability that combines snow/sleet, precipitation intensity, low visibility, near-freezing temperature, high winds, and humidity, with ~10% label noise. This matches the methodology in the XAI notebook and avoids deterministic leakage of a hand-coded rule into the features.
