import fs from "fs";
import path from "path";

interface RecentCrashes {
  window_days: number;
  ref_max_ts: string;
  n_points: number;
  points: [number, number][]; // [lat, lon]
}

const RADIUS_M = 100; // match training: PRIOR_RADIUS_M
const EARTH_M = 6_371_000;

let cached: RecentCrashes | null | undefined;

function load(): RecentCrashes | null {
  if (cached !== undefined) return cached;
  const candidates = [
    path.resolve(process.cwd(), "../ml/data/boston_recent_crashes.json"),
    path.resolve(process.cwd(), "ml/data/boston_recent_crashes.json"),
    path.resolve(process.cwd(), "lib/data/boston_recent_crashes.json"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) {
        cached = JSON.parse(fs.readFileSync(p, "utf-8")) as RecentCrashes;
        console.log(
          `[crashDensity] loaded ${cached.n_points} crash points (window=${cached.window_days}d, ref=${cached.ref_max_ts})`
        );
        return cached;
      }
    } catch {
      // fall through
    }
  }
  console.warn("[crashDensity] no boston_recent_crashes.json found — returning 0 for all lookups");
  cached = null;
  return null;
}

/**
 * Count crashes within RADIUS_M meters of (lat, lon) using a brute-force
 * haversine scan over the last-365-day crash population. Matches the training
 * `prior_year_crash_count` feature semantics (radius + 1-year window).
 */
export function crashesNear(lat: number, lon: number): number {
  const d = load();
  if (!d) return 0;
  const radiusRad = RADIUS_M / EARTH_M;
  const radiusRadSq = radiusRad * radiusRad;
  const latRad = (lat * Math.PI) / 180;
  const cosLat = Math.cos(latRad);
  // Pre-filter by bounding box to skip far points fast.
  const dLatMax = radiusRad * (180 / Math.PI);
  const dLonMax = dLatMax / Math.max(cosLat, 1e-6);
  let count = 0;
  for (const [pLat, pLon] of d.points) {
    if (Math.abs(pLat - lat) > dLatMax) continue;
    if (Math.abs(pLon - lon) > dLonMax) continue;
    // Equirectangular approximation; tight enough at 100 m
    const dy = ((pLat - lat) * Math.PI) / 180;
    const dx = (((pLon - lon) * Math.PI) / 180) * cosLat;
    if (dy * dy + dx * dx <= radiusRadSq) count++;
  }
  return count;
}
