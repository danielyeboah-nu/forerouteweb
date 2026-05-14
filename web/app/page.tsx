"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { SearchForm, type SearchPayload } from "@/components/SearchForm";
import { RouteCard } from "@/components/RouteCard";
import { RouteDetail } from "@/components/RouteDetail";
import type { RoutesResponse } from "@/lib/types";

const MapView = dynamic(
  () => import("@/components/MapView").then((m) => m.MapView),
  { ssr: false }
);

export default function HomePage() {
  const [data, setData] = useState<RoutesResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function search(payload: SearchPayload) {
    setLoading(true);
    setError(null);
    setSelectedId(null);
    try {
      const res = await fetch("/api/routes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `Request failed (${res.status})`);
      }
      const result = (await res.json()) as RoutesResponse;
      setData(result);
      setSelectedId(result.routes[0]?.id ?? null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const selected = data?.routes.find((r) => r.id === selectedId) ?? null;

  return (
    <main className="mx-auto max-w-7xl px-4 py-6 sm:py-10">
      <header className="mb-6 flex flex-col gap-1">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
          ForeRoute <span className="text-slate-500">— know the road before you go</span>
        </h1>
        <p className="text-slate-400 text-sm">
          Weather-aware routing. Rule-based risk scoring + MLflow-served hazard
          model, in your browser.
        </p>
      </header>

      <div className="mb-6 rounded-lg bg-slate-900/60 ring-1 ring-slate-800 p-4">
        <SearchForm onSubmit={search} loading={loading} />
        {data ? (
          <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-400">
            <SourceBadge label="Mapbox" value={data.source.mapbox} />
            <SourceBadge label="Weather" value={data.source.weather} />
            <SourceBadge label="MLflow" value={data.source.mlflow} />
          </div>
        ) : null}
      </div>

      {error ? (
        <div className="mb-4 rounded-md bg-red-950/40 ring-1 ring-red-900 p-3 text-red-300 text-sm">
          {error}
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div className="h-[55vh] lg:h-[70vh]">
          <MapView routes={data?.routes ?? []} selectedId={selectedId} />
        </div>

        <aside className="space-y-3">
          {!data && !loading ? (
            <p className="text-slate-400 text-sm">
              Enter an origin and destination to see routes ranked by risk.
            </p>
          ) : null}
          {data?.routes.map((r) => (
            <RouteCard
              key={r.id}
              route={r}
              selected={r.id === selectedId}
              onSelect={() => setSelectedId(r.id)}
            />
          ))}
        </aside>
      </div>

      {selected ? (
        <section className="mt-8 rounded-lg bg-slate-900/60 ring-1 ring-slate-800 p-5">
          <RouteDetail route={selected} />
        </section>
      ) : null}
    </main>
  );
}

function SourceBadge({
  label,
  value,
}: {
  label: string;
  value: "live" | "mock" | "off";
}) {
  const color =
    value === "live"
      ? "bg-emerald-500/15 text-emerald-300 ring-emerald-700"
      : value === "mock"
      ? "bg-amber-500/15 text-amber-300 ring-amber-700"
      : "bg-slate-500/15 text-slate-400 ring-slate-700";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 ring-1 ${color}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}: {value}
    </span>
  );
}
