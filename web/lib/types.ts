export type PrecipitationType = "none" | "rain" | "snow" | "sleet" | "freezingRain";

export type RoadType =
  | "highway"
  | "arterial"
  | "residential"
  | "bridge"
  | "tunnel"
  | "mountain";

export type RiskLevel = "safe" | "caution" | "risky" | "hazardous";

export type MlVerdict =
  | "outOfRegion"
  | "lower"
  | "typical"
  | "above"
  | "muchHigher";

export type RiskFactorType =
  | "snow"
  | "rain"
  | "ice"
  | "lowVisibility"
  | "wind"
  | "temperature"
  | "roadType";

export interface WeatherCondition {
  temperature: number;
  precipitationType: PrecipitationType;
  precipitationIntensity: number;
  windSpeed: number;
  visibility: number;
  humidity: number;
  dewPoint: number;
}

export interface RiskFactor {
  type: RiskFactorType;
  severity: number;
  description: string;
}

export interface RiskScore {
  value: number;
  level: RiskLevel;
  factors: RiskFactor[];
  /** Probability returned by the MLflow-served model, if available. */
  mlProbability?: number;
  /** Predicted hazardous flag from the MLflow-served model, if available. */
  mlPrediction?: 0 | 1;
  /** Probability divided by the average across all segments in the response. */
  mlRatio?: number;
  /** Plain-language bucket derived from `mlRatio` and region check. */
  mlVerdict?: MlVerdict;
}

export interface LngLat {
  lng: number;
  lat: number;
  name?: string;
}

export interface RouteSegment {
  id: string;
  start: LngLat;
  end: LngLat;
  distance: number;
  roadType: RoadType;
  conditions: WeatherCondition;
  risk: RiskScore;
}

export interface Route {
  id: string;
  name: string;
  start: LngLat;
  end: LngLat;
  segments: RouteSegment[];
  geometry: [number, number][];
  totalDistance: number;
  estimatedDuration: number;
  risk: RiskScore;
}

export interface RoutesResponse {
  routes: Route[];
  source: {
    mapbox: "live" | "mock";
    weather: "live" | "mock";
    mlflow: "live" | "off";
  };
}
