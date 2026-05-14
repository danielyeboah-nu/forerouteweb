# Model Card — ForeRoute-BostonRisk

| Field | Value |
|---|---|
| **Name** | `ForeRoute-BostonRisk` |
| **Version** | v3 (production alias) |
| **Algorithm** | Gradient Boosting Classifier (scikit-learn 1.5.2), served as an MLflow pyfunc that returns the positive-class probability |
| **Tracking** | MLflow 2.18 — SQLite store at `ml/mlflow.db`, model artifact in `ml/mlruns/` |
| **Geographic scope** | Boston metro bbox (42.20–42.45 N, −71.20 to −70.95 W) |
| **Temporal scope** | Crashes 2018-01-01 to 2025-12-31 (8 years) |
| **Last trained** | 2026-05-13 |

## Intended use

Per-segment safety advisory for a driver choosing between alternative routes in the Boston metro. The web product surfaces two complementary scores per route:

- **Weather right now** (rule-based, 0–100) — answers *"are right-now conditions known-adverse?"*. Triggers on snow, ice, hydroplaning rain, low visibility, wind, and tricky road types (bridge / tunnel / mountain).
- **Crash history** (this model, plain-language verdict) — answers *"does this stretch of road have a worse crash history than typical for Boston?"*.

Both are displayed together. Disagreement between them is explicitly flagged to the user.

## Out of scope

- **Not a probability of crashing on this trip.** The model emits a class probability over a training distribution with a 25% base rate. Probabilities are mapped to plain-language verdicts (`Fewer crashes than usual` / `About average` / `More crashes than usual` / `Crashy stretch`) using calibrated quantile thresholds — never shown to the user as a raw "X% chance".
- **Not safe by itself for navigation decisions.** Always paired with the rule-based weather scorer, which runs even if the model is unavailable.
- **Not generalisable outside the Boston bbox.** Out-of-region segments are tagged `Not enough data` in the UI; the displayed number is treated as an extrapolation and de-emphasised.
- **Not a substitute for live traffic, road-closure, or incident data.**

## Data sources

| Layer | Source | Volume after filtering |
|---|---|---|
| Crash records | Boston Open Data — Vision Zero Crash Records (`boston_crashes.csv`) | 32,001 raw → **22,260** after bbox + 2018+ + valid-coordinate filters |
| Historical weather | Open-Meteo Archive API (free, no key) | Hourly: temp, precip type, intensity, wind, visibility, humidity, dew point. Fetched per ~2.5 km grid cell. |
| Road classification | Mapbox Tilequery (`mapbox.mapbox-streets-v8`) | Per-point class mapped to {highway, arterial, residential, bridge, tunnel, mountain} |
| Negatives | Synthesised | 3× positives, anchored on a randomly chosen positive's location with ~55 m σ jitter, timestamp resampled outside any crash's ±2 h window |

## Features (15)

| Feature | Type | Notes |
|---|---|---|
| `temperature` (°C) | numeric | Open-Meteo at sample's hour |
| `precipitation_type` | categorical | WMO weather code mapping → none / rain / snow / sleet / freezingRain |
| `precipitation_intensity` (mm/h) | numeric | Snowfall × 10 or rain rate |
| `wind_speed` (km/h) | numeric | Open-Meteo |
| `visibility` (km) | numeric | Open-Meteo |
| `humidity` (%) | numeric | Open-Meteo |
| `dew_point` (°C) | numeric | Open-Meteo |
| `road_type` | categorical | **Mapbox tilequery, at training and inference** |
| `segment_distance_m` | numeric | 1000 m for crash-point samples |
| `lat`, `lon` | numeric | Sample location |
| `hour_of_day` (0–23) | numeric | Timestamp (UTC) |
| `day_of_week` (0=Mon … 6=Sun) | numeric | Timestamp |
| `month` (1–12) | numeric | Timestamp |
| `prior_year_crash_count` | numeric | **Leakage-safe.** BallTree query: count of *other* crashes within 100 m AND with timestamp strictly < this sample's timestamp AND within 365 days prior. Training mean 2.88, max 32. |

## Label

Binary `label ∈ {0, 1}`. Positives are real Vision Zero crashes; negatives are spatially anchored, temporally resampled non-crash points (see Data sources above).

## Training

- **Split** — 71.5% train / 28.5% test, stratified on `label`. Pre-rebalance positive rate 0.250 in both folds.
- **Rebalance** — under-sample training negatives to 1.5× positives (training positive rate ≈ 0.40). The test fold keeps the natural 0.250 rate.
- **Compared families** — Decision Tree (`max_depth=6, min_samples_leaf=20`), Gradient Boosting (sklearn defaults), MLP (`hidden_layer_sizes=(64, 32)`).
- **Selection** — highest test ROC-AUC. Registered automatically into the MLflow Model Registry as `ForeRoute-BostonRisk` with alias `production`.

## Held-out performance

Test set, n = 17,808.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Decision Tree | 0.6355 | 0.707 | 0.373 | 0.253 | 0.302 |
| **Gradient Boosting (registered)** | **0.6506** | 0.715 | 0.389 | 0.247 | 0.302 |
| MLP | 0.6098 | 0.639 | 0.321 | 0.400 | 0.356 |

### Context

ROC-AUC 0.65 sits at the lower end of the 0.65–0.75 range published in crash-prediction-from-weather work. We are at the bottom of that range **on purpose**:

- The strongest crash predictors — real-time traffic volume, driver behaviour, intersection-specific geometry, signal timing — are not in our feature set.
- Most crashes happen in mild weather (because most driving happens in mild weather), so weather alone has a low predictive ceiling.
- The model honestly stays near base rate on calm urban segments instead of being falsely confident.

The rule-based `Weather right now` score complements this by encoding driving-physics thresholds the ML cannot reliably learn from a noisy crash label.

## Verdict thresholds (calibrated)

The web UI never shows raw probabilities. Each probability is mapped to a plain-language verdict using thresholds calibrated from quantiles of the model's predictions on a representative sample of in-bbox Boston points with typical conditions (n=500, see `ml/calibrate_thresholds.py`).

| Verdict | UI label | Probability threshold | Quantile bucket |
|---|---|---|---|
| `lower` | Fewer crashes than usual | < 0.228 | bottom 25% |
| `typical` | About average | 0.228 – 0.453 | middle 50% |
| `above` | More crashes than usual | 0.453 – 0.532 | 75–90th percentile |
| `muchHigher` | Crashy stretch | ≥ 0.532 | top 10% |
| `outOfRegion` | Not enough data | n/a | outside bbox |

The median (q50 = 0.322) is exported alongside and used to render an explanatory ratio caption in the UI (e.g., "1.4× a typical Boston road").

## Known limitations

- **Boston-only.** Out-of-region predictions are tagged in the UI; the number is treated as extrapolation.
- **Sparse rare-condition coverage.** Boston gets few hours per year of extreme snow / ice / freezing rain. Hazard-weather predictions are partly extrapolative; the rule-based score is the dominant signal in those cases.
- **Mild compression around the base rate.** This is why the UI reframes the probability as a plain-language verdict with a tooltip rather than a raw percent.
- **No demographic / equity audit shipped.** Crash density per cell is shaped by enforcement patterns, infrastructure investment, and reporting bias — all of which correlate with neighbourhood demographics. A per-neighbourhood breakdown is planned.
- **Static crash-density lookup.** The web app loads a precomputed list of crashes from the last 365 days of the dataset; not updated in real time.

## Maintenance

- **Retrain trigger** — feature PSI > 0.25 against the training reference for ≥ 3 consecutive days, OR a ≥ 5 pp drop in recall on backfilled ground truth.
- **Cadence** — quarterly retrain regardless of drift signal.
- **Rollback** — previous version stays in the registry; flip the `production` alias to a prior version.

## Reproducibility

Pipeline lives entirely under `ml/`:

- `train.py` — CLI: `--data`, `--label-col`, `--registry-name`. Trains the three families, evaluates on a held-out split, registers the best.
- `build_boston_dataset.py` — raw crashes → cleaned + labelled CSV + density / recent-crashes JSON for runtime lookup.
- `calibrate_thresholds.py` — re-computes the verdict thresholds from quantiles when the model is retrained.
- `Makefile` — `boston-data` · `train-boston` · `serve-boston` · `calibrate-thresholds`.

## Contact

ForeRoute group — Northeastern University, 2026.
