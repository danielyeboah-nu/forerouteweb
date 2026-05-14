"use client";

import { useEffect, useRef } from "react";
import mapboxgl, { type Map as MapboxMap } from "mapbox-gl";
import type { Route } from "@/lib/types";

const LEVEL_COLOR: Record<string, string> = {
  safe: "#22c55e",
  caution: "#eab308",
  risky: "#f97316",
  hazardous: "#ef4444",
};

export interface MapViewProps {
  routes: Route[];
  selectedId?: string | null;
}

export function MapView({ routes, selectedId }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapboxMap | null>(null);
  const markersRef = useRef<mapboxgl.Marker[]>([]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
    if (token) mapboxgl.accessToken = token;

    mapRef.current = new mapboxgl.Map({
      container: containerRef.current,
      style: token ? "mapbox://styles/mapbox/dark-v11" : "mapbox://styles/mapbox/dark-v11",
      center: [-73, 42.5],
      zoom: 4,
      attributionControl: false,
    });
    mapRef.current.addControl(new mapboxgl.NavigationControl({ visualizePitch: false }), "top-right");
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const draw = () => {
      // Remove previously-added layers/sources
      const existing = map.getStyle()?.layers ?? [];
      for (const layer of existing) {
        if (layer.id.startsWith("fr-route-")) map.removeLayer(layer.id);
      }
      const sources = Object.keys((map.getStyle() as any)?.sources ?? {});
      for (const s of sources) {
        if (s.startsWith("fr-route-")) map.removeSource(s);
      }
      // Remove previously-added markers (and their popups). Without this,
      // every new search stacks more pins on the map.
      for (const m of markersRef.current) m.remove();
      markersRef.current = [];

      if (routes.length === 0) return;

      const bounds = new mapboxgl.LngLatBounds();
      routes.forEach((route) => {
        const id = `fr-route-${route.id}`;
        map.addSource(id, {
          type: "geojson",
          data: {
            type: "Feature",
            properties: {},
            geometry: { type: "LineString", coordinates: route.geometry },
          },
        });
        const isSelected = selectedId ? selectedId === route.id : false;
        map.addLayer({
          id,
          type: "line",
          source: id,
          layout: { "line-join": "round", "line-cap": "round" },
          paint: {
            "line-color": LEVEL_COLOR[route.risk.level] ?? "#38bdf8",
            "line-width": isSelected ? 6 : 4,
            "line-opacity": isSelected ? 1 : 0.6,
          },
        });
        route.geometry.forEach(([lng, lat]) => bounds.extend([lng, lat]));
      });

      // Endpoint markers from the first route, with labels
      const first = routes[0];
      if (first) {
        const startName = first.start.name ?? "Origin";
        const endName = first.end.name ?? "Destination";
        const startMarker = new mapboxgl.Marker({ color: "#38bdf8" })
          .setLngLat([first.start.lng, first.start.lat])
          .setPopup(
            new mapboxgl.Popup({ offset: 18, closeButton: false }).setHTML(
              `<div style="font-size:11px;color:#0f172a"><strong>From</strong><br/>${escapeHtml(startName)}</div>`
            )
          )
          .addTo(map)
          .togglePopup();
        const endMarker = new mapboxgl.Marker({ color: "#f43f5e" })
          .setLngLat([first.end.lng, first.end.lat])
          .setPopup(
            new mapboxgl.Popup({ offset: 18, closeButton: false }).setHTML(
              `<div style="font-size:11px;color:#0f172a"><strong>To</strong><br/>${escapeHtml(endName)}</div>`
            )
          )
          .addTo(map)
          .togglePopup();
        markersRef.current.push(startMarker, endMarker);
      }

      if (!bounds.isEmpty()) {
        map.fitBounds(bounds, { padding: 60, duration: 600 });
      }
    };

    if (map.isStyleLoaded()) draw();
    else map.once("load", draw);
  }, [routes, selectedId]);

  return <div ref={containerRef} className="h-full w-full rounded-lg overflow-hidden ring-1 ring-slate-800" />;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
