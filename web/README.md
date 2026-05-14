# ForeRouteWeb — Next.js frontend

The web UI and API orchestrator for ForeRoute. Calls Mapbox (Directions + Geocoding), OpenWeatherMap, and the MLflow model server (`mlflow models serve`) — no FastAPI in the loop.

## Setup

```bash
cp .env.local.example .env.local
# Fill in:
#   NEXT_PUBLIC_MAPBOX_TOKEN
#   OPENWEATHER_API_KEY
#   MLFLOW_SERVE_URL (default: http://localhost:5001/invocations)

npm install
npm run dev
# http://localhost:3000
```

The MLflow model server must already be running (`cd ../ml && make serve`). If `MLFLOW_SERVE_URL` is unset or the server is down, the app silently falls back to rule-based scoring only.

## Routes

- `POST /api/routes` — body `{ from, to }` (city names or `fromLngLat`/`toLngLat`). Returns ranked-by-risk alternatives with per-segment weather, rule-based risk score, and (when MLflow is reachable) ML hazard probability.

## File map

```
web/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                  # client home (search + map + cards + detail)
│   ├── globals.css
│   └── api/routes/route.ts       # orchestrator: Mapbox → Weather → MLflow → score
├── components/
│   ├── SearchForm.tsx
│   ├── MapView.tsx               # mapbox-gl polylines colored by risk level
│   ├── RouteCard.tsx
│   ├── RouteDetail.tsx           # factor breakdown + per-segment table
│   └── RiskBadge.tsx
├── lib/
│   ├── types.ts                  # Route, RouteSegment, WeatherCondition, RiskScore
│   ├── risk.ts                   # port of iOS RiskCalculator.swift
│   ├── mapbox.ts                 # Directions + Geocoding + polyline segmentation
│   ├── weather.ts                # OpenWeather adapter + Magnus dew point
│   └── mlflow.ts                 # POST /invocations (dataframe_split)
└── ...
```
