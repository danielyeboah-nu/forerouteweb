import { NextResponse } from "next/server";
import {
  fetchAlternatives,
  geocode,
  segmentPolyline,
  tileQueryRoadType,
  type MapboxRoute,
} from "@/lib/mapbox";
import { fetchWeather, mockWeatherFor } from "@/lib/weather";
import { scoreBatch, type MlInput } from "@/lib/mlflow";
import { scoreRoute, scoreSegment } from "@/lib/risk";
import { crashesNear } from "@/lib/crashDensity";
import { annotateMlVerdicts } from "@/lib/mlVerdictServer";
import type {
  LngLat,
  Route,
  RouteSegment,
  RoutesResponse,
  WeatherCondition,
} from "@/lib/types";

export const runtime = "nodejs";

interface RequestBody {
  from?: string;
  to?: string;
  fromLngLat?: LngLat;
  toLngLat?: LngLat;
}

function randomId(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `id-${Math.random().toString(36).slice(2, 10)}`
  );
}

function mockRoute(start: LngLat, end: LngLat, name: string, idx: number): MapboxRoute {
  const dx = end.lng - start.lng;
  const dy = end.lat - start.lat;
  const wiggle = idx === 0 ? 0 : 0.2;
  const mid: [number, number] = [
    start.lng + dx * 0.5 + wiggle * dy,
    start.lat + dy * 0.5 - wiggle * dx,
  ];
  const distance = haversine(start, end) * (1 + 0.05 * idx);
  return {
    name,
    distance,
    duration: distance / 18, // ~65 km/h
    geometry: [
      [start.lng, start.lat],
      mid,
      [end.lng, end.lat],
    ],
  };
}

function haversine(a: LngLat, b: LngLat): number {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const R = 6371000;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const x =
    Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(x));
}

async function resolveEndpoint(
  query: string | undefined,
  lngLat: LngLat | undefined,
  mapboxToken: string | undefined
): Promise<LngLat | null> {
  if (lngLat) return lngLat;
  if (!query) return null;
  if (mapboxToken) {
    try {
      return await geocode(query, mapboxToken);
    } catch (err) {
      console.warn("[geocode] falling back to mock:", err);
    }
  }
  // Mock geocoding: a couple of known cities, else center on Boston
  const lookup: Record<string, LngLat> = {
    boston: { lng: -71.0589, lat: 42.3601, name: "Boston, MA" },
    "new york": { lng: -74.006, lat: 40.7128, name: "New York, NY" },
    nyc: { lng: -74.006, lat: 40.7128, name: "New York, NY" },
    chicago: { lng: -87.6298, lat: 41.8781, name: "Chicago, IL" },
    denver: { lng: -104.9903, lat: 39.7392, name: "Denver, CO" },
    seattle: { lng: -122.3321, lat: 47.6062, name: "Seattle, WA" },
    portland: { lng: -122.6784, lat: 45.5152, name: "Portland, OR" },
  };
  return lookup[query.toLowerCase()] ?? { lng: -71.0589, lat: 42.3601, name: query };
}

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as RequestBody;

  const mapboxToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
  const weatherKey = process.env.OPENWEATHER_API_KEY;
  const mlflowUrl = process.env.MLFLOW_SERVE_URL;

  const start = await resolveEndpoint(body.from, body.fromLngLat, mapboxToken);
  const end = await resolveEndpoint(body.to, body.toLngLat, mapboxToken);
  if (!start || !end) {
    return NextResponse.json({ error: "Could not resolve from/to" }, { status: 400 });
  }

  const source: RoutesResponse["source"] = {
    mapbox: mapboxToken ? "live" : "mock",
    weather: weatherKey ? "live" : "mock",
    mlflow: mlflowUrl ? "live" : "off",
  };

  // 1. Fetch alternatives (real or mock)
  let alternatives: MapboxRoute[];
  if (mapboxToken) {
    try {
      alternatives = await fetchAlternatives(start, end, mapboxToken);
    } catch (err) {
      console.warn("[mapbox] falling back to mock:", err);
      source.mapbox = "mock";
      alternatives = [
        mockRoute(start, end, "Fastest Route", 0),
        mockRoute(start, end, "Alternate 1", 1),
      ];
    }
  } else {
    alternatives = [
      mockRoute(start, end, "Fastest Route", 0),
      mockRoute(start, end, "Alternate 1", 1),
    ];
  }

  // 2. For each alternative, build segments, fetch weather, score
  const routes: Route[] = [];
  for (const alt of alternatives) {
    const segs = segmentPolyline(alt.geometry, alt.distance, 5);

    // Override the length-based road_type heuristic with real Mapbox tilequery
    // per segment midpoint so the model sees the same classification it was
    // trained on. Parallelised + in-memory cached.
    if (mapboxToken) {
      await Promise.all(
        segs.map(async (s, i) => {
          const midLat = (s.start.lat + s.end.lat) / 2;
          const midLng = (s.start.lng + s.end.lng) / 2;
          segs[i].roadType = await tileQueryRoadType(midLat, midLng, mapboxToken);
        })
      );
    }

    const segmentRecords: RouteSegment[] = [];

    for (const s of segs) {
      const midpoint: LngLat = {
        lng: (s.start.lng + s.end.lng) / 2,
        lat: (s.start.lat + s.end.lat) / 2,
      };
      let conditions: WeatherCondition;
      if (weatherKey) {
        try {
          conditions = await fetchWeather(midpoint, weatherKey);
        } catch (err) {
          console.warn("[weather] falling back to mock:", err);
          source.weather = "mock";
          conditions = mockWeatherFor(midpoint);
        }
      } else {
        conditions = mockWeatherFor(midpoint);
      }
      const risk = scoreSegment(conditions, s.roadType);
      segmentRecords.push({
        id: randomId(),
        start: s.start,
        end: s.end,
        distance: s.distance,
        roadType: s.roadType,
        conditions,
        risk,
      });
    }

    // 3. ML scoring (single batched call per route)
    if (mlflowUrl && segmentRecords.length > 0) {
      try {
        const now = new Date();
        const jsDow = now.getUTCDay(); // 0=Sun..6=Sat
        const pandasDow = (jsDow + 6) % 7; // 0=Mon..6=Sun (match training)
        const hourOfDay = now.getUTCHours();
        const month = now.getUTCMonth() + 1;
        const inputs: MlInput[] = segmentRecords.map((s) => {
          const midLat = (s.start.lat + s.end.lat) / 2;
          const midLng = (s.start.lng + s.end.lng) / 2;
          return {
            conditions: s.conditions,
            roadType: s.roadType,
            distance: s.distance,
            lat: midLat,
            lon: midLng,
            hourOfDay,
            dayOfWeek: pandasDow,
            month,
            priorYearCrashCount: crashesNear(midLat, midLng),
          };
        });
        const results = await scoreBatch(mlflowUrl, inputs);
        results.forEach((r, i) => {
          const seg = segmentRecords[i];
          if (seg) {
            seg.risk.mlPrediction = r.prediction;
            seg.risk.mlProbability = r.probability;
          }
        });
      } catch (err) {
        console.warn("[mlflow] scoring failed, continuing without ML:", err);
        source.mlflow = "off";
      }
    }

    const routeRisk = scoreRoute(segmentRecords);
    // Bubble up ML score at the route level as the max-segment probability
    const mlProbs = segmentRecords
      .map((s) => s.risk.mlProbability)
      .filter((p): p is number => typeof p === "number");
    if (mlProbs.length > 0) {
      routeRisk.mlProbability = Math.max(...mlProbs);
      routeRisk.mlPrediction = routeRisk.mlProbability >= 0.5 ? 1 : 0;
    }

    routes.push({
      id: randomId(),
      name: alt.name,
      start,
      end,
      segments: segmentRecords,
      geometry: alt.geometry,
      totalDistance: alt.distance,
      estimatedDuration: alt.duration,
      risk: routeRisk,
    });
  }

  routes.sort((a, b) => a.risk.value - b.risk.value);

  annotateMlVerdicts(routes);

  const payload: RoutesResponse = { routes, source };
  return NextResponse.json(payload);
}
