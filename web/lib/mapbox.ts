import type { LngLat, RoadType } from "./types";

export interface MapboxRoute {
  name: string;
  distance: number; // meters
  duration: number; // seconds
  geometry: [number, number][]; // [lng, lat]
}

const DIRECTIONS_URL =
  "https://api.mapbox.com/directions/v5/mapbox/driving";

export async function fetchAlternatives(
  start: LngLat,
  end: LngLat,
  token: string
): Promise<MapboxRoute[]> {
  const coords = `${start.lng},${start.lat};${end.lng},${end.lat}`;
  const url =
    `${DIRECTIONS_URL}/${coords}` +
    `?alternatives=true&geometries=geojson&overview=full&steps=false` +
    `&access_token=${encodeURIComponent(token)}`;

  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Mapbox Directions ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as {
    routes: Array<{
      distance: number;
      duration: number;
      geometry: { coordinates: [number, number][] };
    }>;
  };
  return data.routes.map((r, i) => ({
    name: i === 0 ? "Fastest Route" : `Alternate ${i}`,
    distance: r.distance,
    duration: r.duration,
    geometry: r.geometry.coordinates,
  }));
}

export async function geocode(
  query: string,
  token: string
): Promise<LngLat | null> {
  const url =
    `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json` +
    `?limit=1&access_token=${encodeURIComponent(token)}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Mapbox Geocoding ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as {
    features: Array<{ center: [number, number]; place_name: string }>;
  };
  const f = data.features[0];
  if (!f) return null;
  return { lng: f.center[0], lat: f.center[1], name: f.place_name };
}

export interface PlaceSuggestion extends LngLat {
  id: string;
  name: string;
  placeName: string;
}

export async function searchSuggestions(
  query: string,
  token: string,
  opts: { proximity?: LngLat; limit?: number; signal?: AbortSignal } = {}
): Promise<PlaceSuggestion[]> {
  const q = query.trim();
  if (!q) return [];
  const params = new URLSearchParams({
    autocomplete: "true",
    limit: String(opts.limit ?? 5),
    access_token: token,
  });
  if (opts.proximity) {
    params.set("proximity", `${opts.proximity.lng},${opts.proximity.lat}`);
  }
  const url =
    `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(q)}.json?` +
    params.toString();
  const res = await fetch(url, { cache: "no-store", signal: opts.signal });
  if (!res.ok) {
    throw new Error(`Mapbox Geocoding ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as {
    features: Array<{
      id: string;
      text: string;
      place_name: string;
      center: [number, number];
    }>;
  };
  return data.features.map((f) => ({
    id: f.id,
    name: f.text,
    placeName: f.place_name,
    lng: f.center[0],
    lat: f.center[1],
  }));
}

export async function reverseGeocode(
  point: LngLat,
  token: string
): Promise<LngLat | null> {
  const url =
    `https://api.mapbox.com/geocoding/v5/mapbox.places/${point.lng},${point.lat}.json` +
    `?limit=1&types=place,locality,neighborhood,address&access_token=${encodeURIComponent(token)}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Mapbox Reverse Geocoding ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as {
    features: Array<{ center: [number, number]; place_name: string; text: string }>;
  };
  const f = data.features[0];
  if (!f) return null;
  return { lng: f.center[0], lat: f.center[1], name: f.text };
}

/**
 * Resample a polyline into N evenly-spaced midpoints so we can sample weather
 * and assign a road type per segment.
 */
export function segmentPolyline(
  geometry: [number, number][],
  totalDistance: number,
  maxSegments = 6
): Array<{ start: LngLat; end: LngLat; distance: number; roadType: RoadType }> {
  if (geometry.length < 2) return [];
  const n = Math.max(1, Math.min(maxSegments, Math.floor(geometry.length / 4) || 1));
  const step = Math.floor(geometry.length / n);
  const segs: Array<{
    start: LngLat;
    end: LngLat;
    distance: number;
    roadType: RoadType;
  }> = [];
  for (let i = 0; i < n; i++) {
    const aIdx = i * step;
    const bIdx = i === n - 1 ? geometry.length - 1 : (i + 1) * step;
    const a = geometry[aIdx];
    const b = geometry[bIdx];
    if (!a || !b) continue;
    segs.push({
      start: { lng: a[0], lat: a[1] },
      end: { lng: b[0], lat: b[1] },
      distance: totalDistance / n,
      roadType: inferRoadType(totalDistance, n),
    });
  }
  return segs;
}

function inferRoadType(totalDistance: number, segments: number): RoadType {
  const segMean = totalDistance / segments;
  if (segMean > 50000) return "highway";
  if (segMean > 5000) return "arterial";
  return "residential";
}

// ---- tilequery road-type classification (matches training pipeline) --------

const roadTypeCache = new Map<string, RoadType>();

function roadCacheKey(lat: number, lng: number): string {
  return `${lat.toFixed(4)},${lng.toFixed(4)}`;
}

// Map a Mapbox tilequery feature's properties → our RoadType enum.
// Mirrors ml/build_boston_dataset.py `_road_class_to_type`.
function roadClassToType(props: Record<string, unknown>): RoadType {
  const structure = String(props.structure ?? "").toLowerCase();
  if (structure === "bridge") return "bridge";
  if (structure === "tunnel") return "tunnel";
  const cls = String(props.class ?? "").toLowerCase();
  if (cls === "motorway" || cls === "trunk") return "highway";
  if (cls === "primary" || cls === "secondary") return "arterial";
  if (
    cls === "tertiary" ||
    cls === "street" ||
    cls === "street_limited" ||
    cls === "service" ||
    cls === "residential"
  ) {
    return "residential";
  }
  return "arterial";
}

/**
 * Classify the road at (lat, lng) via Mapbox tilequery, matching the training
 * pipeline's classification so the model sees the same feature it learned on.
 * Cached in-memory by 4-decimal lat/lng (~11 m). Falls back to "arterial" on
 * any error / no features so callers never have to handle a null.
 */
export async function tileQueryRoadType(
  lat: number,
  lng: number,
  token: string
): Promise<RoadType> {
  const key = roadCacheKey(lat, lng);
  const cached = roadTypeCache.get(key);
  if (cached) return cached;
  const url =
    `https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/tilequery/${lng},${lat}.json` +
    `?radius=25&limit=5&dedupe=true&layers=road&access_token=${encodeURIComponent(token)}`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      roadTypeCache.set(key, "arterial");
      return "arterial";
    }
    const data = (await res.json()) as {
      features?: Array<{ properties?: Record<string, unknown> }>;
    };
    const features = data.features ?? [];
    if (features.length === 0) {
      roadTypeCache.set(key, "arterial");
      return "arterial";
    }
    // Match training: pick the highest-class road if multiple features overlap.
    const rank: Record<RoadType, number> = {
      highway: 4,
      arterial: 3,
      residential: 2,
      tunnel: 1,
      bridge: 0,
      mountain: 0,
    };
    const types = features.map((f) => roadClassToType(f.properties ?? {}));
    const best = types.reduce((a, b) => (rank[b] > rank[a] ? b : a));
    roadTypeCache.set(key, best);
    return best;
  } catch {
    return "arterial";
  }
}
