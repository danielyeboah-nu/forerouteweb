import type { LngLat, PrecipitationType, WeatherCondition } from "./types";

const BASE = "https://api.openweathermap.org/data/2.5/weather";

interface OWMResponse {
  main: { temp: number; humidity: number };
  wind: { speed: number };
  visibility?: number;
  weather: Array<{ main: string; id: number }>;
  rain?: { "1h"?: number; "3h"?: number };
  snow?: { "1h"?: number; "3h"?: number };
}

function mapPrecipitationType(weather: OWMResponse["weather"]): PrecipitationType {
  const id = weather[0]?.id ?? 800;
  if (id >= 600 && id < 700) {
    if (id === 611 || id === 612 || id === 613) return "sleet";
    return "snow";
  }
  if (id >= 500 && id < 600) {
    if (id === 511) return "freezingRain";
    return "rain";
  }
  if (id >= 300 && id < 500) return "rain";
  return "none";
}

function dewPoint(tempC: number, humidity: number): number {
  // Magnus formula approximation
  const a = 17.625;
  const b = 243.04;
  const h = Math.max(1, humidity);
  const gamma = Math.log(h / 100) + (a * tempC) / (b + tempC);
  return (b * gamma) / (a - gamma);
}

export async function fetchWeather(
  point: LngLat,
  apiKey: string
): Promise<WeatherCondition> {
  const url =
    `${BASE}?lat=${point.lat}&lon=${point.lng}` +
    `&units=metric&appid=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`OpenWeather ${res.status}: ${await res.text()}`);
  }
  const data = (await res.json()) as OWMResponse;

  const ptype = mapPrecipitationType(data.weather);
  const rainMm = data.rain?.["1h"] ?? (data.rain?.["3h"] ?? 0) / 3;
  const snowMm = data.snow?.["1h"] ?? (data.snow?.["3h"] ?? 0) / 3;
  const precipitationIntensity = ptype === "snow" || ptype === "sleet" ? snowMm : rainMm;

  return {
    temperature: data.main.temp,
    precipitationType: ptype,
    precipitationIntensity,
    windSpeed: (data.wind.speed ?? 0) * 3.6, // m/s → km/h
    visibility: (data.visibility ?? 10000) / 1000, // m → km
    humidity: data.main.humidity,
    dewPoint: dewPoint(data.main.temp, data.main.humidity),
  };
}

export function mockWeatherFor(point: LngLat): WeatherCondition {
  // Deterministic-ish mock derived from coordinates so the UI feels stable
  const seed = Math.abs(Math.sin(point.lat * 17 + point.lng * 31));
  const t = -5 + seed * 25;
  const ptypes: PrecipitationType[] = ["none", "rain", "snow", "sleet"];
  const ptype = ptypes[Math.floor(seed * ptypes.length) % ptypes.length];
  const intensity = ptype === "none" ? 0 : seed * 6;
  return {
    temperature: t,
    precipitationType: ptype,
    precipitationIntensity: intensity,
    windSpeed: 10 + seed * 40,
    visibility: Math.max(0.3, 10 - seed * 9.5),
    humidity: 50 + seed * 45,
    dewPoint: t - 3,
  };
}
