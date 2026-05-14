"use client";

import { useEffect, useRef, useState } from "react";
import {
  searchSuggestions,
  reverseGeocode,
  type PlaceSuggestion,
} from "@/lib/mapbox";
import type { LngLat } from "@/lib/types";

export interface SearchPayload {
  from: string;
  to: string;
  fromLngLat?: LngLat;
  toLngLat?: LngLat;
}

export interface SearchFormProps {
  onSubmit: (payload: SearchPayload) => void;
  loading?: boolean;
}

const TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

export function SearchForm({ onSubmit, loading }: SearchFormProps) {
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [fromCoord, setFromCoord] = useState<LngLat | null>(null);
  const [toCoord, setToCoord] = useState<LngLat | null>(null);
  const [userLoc, setUserLoc] = useState<LngLat | null>(null);
  const [fromSugs, setFromSugs] = useState<PlaceSuggestion[]>([]);
  const [toSugs, setToSugs] = useState<PlaceSuggestion[]>([]);
  const [focused, setFocused] = useState<"from" | "to" | null>(null);
  const [geoLoading, setGeoLoading] = useState(false);

  // Auto-fill "From" with the user's location on mount
  useEffect(() => {
    if (!navigator.geolocation) return;
    setGeoLoading(true);
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const loc: LngLat = {
          lng: pos.coords.longitude,
          lat: pos.coords.latitude,
        };
        setUserLoc(loc);
        if (TOKEN) {
          try {
            const place = await reverseGeocode(loc, TOKEN);
            const named = place ?? { ...loc, name: "Current location" };
            setFromCoord(named);
            setFrom(named.name ?? "Current location");
          } catch {
            setFromCoord({ ...loc, name: "Current location" });
            setFrom("Current location");
          }
        } else {
          setFromCoord({ ...loc, name: "Current location" });
          setFrom("Current location");
        }
        setGeoLoading(false);
      },
      () => setGeoLoading(false),
      { enableHighAccuracy: false, timeout: 6000, maximumAge: 5 * 60_000 }
    );
  }, []);

  // Debounced typeahead for "From"
  const fromReq = useRef(0);
  useEffect(() => {
    if (!TOKEN) return;
    if (focused !== "from") return;
    if (fromCoord && from === fromCoord.name) {
      setFromSugs([]);
      return;
    }
    if (from.trim().length < 2) {
      setFromSugs([]);
      return;
    }
    const reqId = ++fromReq.current;
    const handle = setTimeout(async () => {
      try {
        const sugs = await searchSuggestions(from, TOKEN!, {
          proximity: userLoc ?? undefined,
        });
        if (reqId === fromReq.current) setFromSugs(sugs);
      } catch {
        /* swallow */
      }
    }, 250);
    return () => clearTimeout(handle);
  }, [from, focused, fromCoord, userLoc]);

  // Debounced typeahead for "To" — biased toward fromCoord or userLoc
  const toReq = useRef(0);
  useEffect(() => {
    if (!TOKEN) return;
    if (focused !== "to") return;
    if (toCoord && to === toCoord.name) {
      setToSugs([]);
      return;
    }
    if (to.trim().length < 2) {
      setToSugs([]);
      return;
    }
    const reqId = ++toReq.current;
    const handle = setTimeout(async () => {
      try {
        const sugs = await searchSuggestions(to, TOKEN!, {
          proximity: fromCoord ?? userLoc ?? undefined,
        });
        if (reqId === toReq.current) setToSugs(sugs);
      } catch {
        /* swallow */
      }
    }, 250);
    return () => clearTimeout(handle);
  }, [to, focused, toCoord, fromCoord, userLoc]);

  function pickFrom(s: PlaceSuggestion) {
    setFromCoord({ lng: s.lng, lat: s.lat, name: s.placeName });
    setFrom(s.placeName);
    setFromSugs([]);
    setFocused(null);
  }
  function pickTo(s: PlaceSuggestion) {
    setToCoord({ lng: s.lng, lat: s.lat, name: s.placeName });
    setTo(s.placeName);
    setToSugs([]);
    setFocused(null);
  }

  function submit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      from: from.trim(),
      to: to.trim(),
      fromLngLat: fromCoord && from === fromCoord.name ? fromCoord : undefined,
      toLngLat: toCoord && to === toCoord.name ? toCoord : undefined,
    });
  }

  return (
    <form
      onSubmit={submit}
      className="flex flex-col gap-3 sm:flex-row sm:items-end"
    >
      <SuggestField
        label="From"
        value={from}
        placeholder={geoLoading ? "Locating…" : "Origin"}
        onChange={(v) => {
          setFrom(v);
          setFromCoord(null);
        }}
        onFocus={() => setFocused("from")}
        onBlur={() => setTimeout(() => setFocused((f) => (f === "from" ? null : f)), 120)}
        suggestions={focused === "from" ? fromSugs : []}
        onPick={pickFrom}
      />
      <SuggestField
        label="To"
        value={to}
        placeholder="Destination"
        onChange={(v) => {
          setTo(v);
          setToCoord(null);
        }}
        onFocus={() => setFocused("to")}
        onBlur={() => setTimeout(() => setFocused((f) => (f === "to" ? null : f)), 120)}
        suggestions={focused === "to" ? toSugs : []}
        onPick={pickTo}
      />
      <button
        type="submit"
        disabled={loading || !from.trim() || !to.trim()}
        className="rounded-md bg-sky-600 hover:bg-sky-500 disabled:opacity-50 px-5 py-2 font-medium"
      >
        {loading ? "Routing…" : "Find safer routes"}
      </button>
    </form>
  );
}

interface SuggestFieldProps {
  label: string;
  value: string;
  placeholder: string;
  onChange: (v: string) => void;
  onFocus: () => void;
  onBlur: () => void;
  suggestions: PlaceSuggestion[];
  onPick: (s: PlaceSuggestion) => void;
}

function SuggestField({
  label,
  value,
  placeholder,
  onChange,
  onFocus,
  onBlur,
  suggestions,
  onPick,
}: SuggestFieldProps) {
  return (
    <label className="relative flex flex-1 flex-col gap-1 text-sm">
      <span className="text-slate-400">{label}</span>
      <input
        className="rounded-md bg-slate-900 border border-slate-700 px-3 py-2 outline-none focus:border-sky-500"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={onFocus}
        onBlur={onBlur}
        placeholder={placeholder}
        autoComplete="off"
        required
      />
      {suggestions.length > 0 ? (
        <ul className="absolute left-0 right-0 top-full z-20 mt-1 max-h-72 overflow-auto rounded-md border border-slate-700 bg-slate-900 shadow-lg">
          {suggestions.map((s) => (
            <li key={s.id}>
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  onPick(s);
                }}
                className="block w-full px-3 py-2 text-left hover:bg-slate-800"
              >
                <div className="text-sm">{s.name}</div>
                <div className="text-xs text-slate-400 truncate">{s.placeName}</div>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </label>
  );
}
