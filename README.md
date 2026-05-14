# ForeRouteWeb

Web version of [ForeRoute](https://github.com/) — weather-aware safe driving routes — built with **Next.js** for the UI/API and **MLflow** for ML experiment tracking, model registry, and serving.

## Architecture

```
┌─────────────────┐        ┌──────────────────┐        ┌────────────────────┐
│  Next.js (web)  │ ──────▶│  Mapbox / OWM    │        │  MLflow Tracking   │
│  - UI           │        │  external APIs   │        │  - mlruns/         │
│  - /api/routes  │ ──┐    └──────────────────┘        │  - SQLite backend  │
└─────────────────┘   │                                 │  - Model Registry  │
                      │    ┌──────────────────────┐    └────────────────────┘
                      └───▶│  MLflow Model Server │             ▲
                           │  (mlflow models serve)│  trains    │
                           │   /invocations :5001  │◀───────────┘
                           └──────────────────────┘
```

- **`web/`** — Next.js 15 (App Router, TS, Tailwind). UI + `/api/routes` orchestrator that calls Mapbox, OpenWeather, and the MLflow inference endpoint.
- **`ml/`** — Python training pipeline. Trains a Decision Tree, Gradient Boosting, and MLP on the segment-level risk dataset, logs everything to MLflow, registers the best model as `ForeRoute-SegmentRisk`, and serves it.

## Quick start

```bash
# 1. Train models, log to MLflow, register best
cd ml
make install         # pip install -r requirements.txt
make train           # writes mlruns/ + mlflow.db, registers ForeRoute-SegmentRisk

# 2. Start the MLflow UI (optional) and model server
make mlflow-ui       # http://localhost:5000
make serve           # http://localhost:5001/invocations

# 3. Start the web app
cd ../web
cp .env.local.example .env.local      # fill in MAPBOX + OWM keys
npm install
npm run dev          # http://localhost:3000
```

## Risk model

Two paths run side-by-side for every route segment:

1. **Rule-based** (port of the iOS `RiskCalculator.swift`): deterministic 0–100 score with itemized factors. Always available; used as the safety fallback.
2. **ML prediction** (MLflow-served model): probability that the segment is hazardous, served from the registered `ForeRoute-SegmentRisk` model. Used to flag predictions that disagree with the rule.

Both signals are surfaced in the UI so the driver — and any reviewer — can see *why* a route was rated the way it was.
