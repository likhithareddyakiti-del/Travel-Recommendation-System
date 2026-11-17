"""
import_real_data.py
-------------------
Turns the REAL Kaggle file "Top Indian Places to Visit.csv" into data/destinations.csv.

Source (verified to exist when this project was written, file NOT bundled):
  https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places
  (325 attractions; ratings are Google review averages. Check the dataset page for its licence.)

What the Kaggle file HAS : State, City, Name, Type, time needed (hrs), Google rating,
                           entrance fee (INR), significance ...
What it does NOT have    : latitude/longitude, travel cost, daily cost, my interest categories.
So this script ADDS them, and every added field is labelled:
  * latitude/longitude : geocoded from "City, State, India" with OpenStreetMap Nominatim
                         (checked to lie inside India's bounding box; SPOT-CHECK a few!)
  * interests          : derived from the 'Type'/'Significance' text with TYPE_TO_INTEREST below
  * travel_cost        : ASSUMPTION = 2 x straight-line km x 1.3 detour x Rs/km from a reference origin
  * daily_cost         : ASSUMPTION = base daily cost + entrance fees / days
  * recommended_days   : ASSUMPTION = ceil(total visit hours / 6), between 1 and 5
  The three cost/duration items are NOT verified prices. Change the constants or edit the CSV.

The Kaggle data is per ATTRACTION; this project recommends per DESTINATION (city), so
attractions are aggregated by (State, City): mean rating, union of interests, summed fees/hours.

Usage:
  python import_real_data.py --input "data/raw/Top Indian Places to Visit.csv" --origin "Chennai, India"
Needs internet + osmnx for geocoding (about 1 request/second; results cached in data/geocode_cache.csv).
"""
import argparse
import math
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from src.data_preprocessing import REQUIRED_DEST_COLUMNS, haversine_km

# ---- ASSUMPTIONS (not verified prices - edit freely) ----------------------------------
RUPEES_PER_KM = 2.5          # rough public-transport style cost per km
DETOUR_FACTOR = 1.3          # road distance is longer than straight line
BASE_DAILY_COST = 1800       # food + local transport + modest stay, per day
HOURS_PER_DAY = 6
MIN_TRAVEL_COST = 300
INDIA_BBOX = (6.0, 37.5, 68.0, 98.0)      # lat_min, lat_max, lon_min, lon_max

# keyword (lower-case, searched inside Type + Significance) -> project interest
TYPE_TO_INTEREST = {
    "beach": "beaches", "coast": "beaches",
    "temple": "history", "monument": "history", "fort": "history", "palace": "history",
    "museum": "history", "memorial": "history", "historical": "history", "heritage": "history",
    "tomb": "history", "mosque": "history", "church": "history", "religious": "history",
    "cave": "history", "stupa": "history", "gurudwara": "history", "shrine": "history",
    "park": "nature", "wildlife": "nature", "sanctuary": "nature", "lake": "nature",
    "waterfall": "nature", "garden": "nature", "nature": "nature", "zoo": "nature",
    "backwater": "nature", "forest": "nature", "river": "nature", "dam": "nature",
    "hill": "hills", "mountain": "hills", "valley": "hills", "peak": "hills",
    "adventure": "adventure", "trek": "adventure", "sport": "adventure",
    "rafting": "adventure", "ski": "adventure", "camp": "adventure",
}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_columns(df):
    """Locate needed columns by keyword so small header spelling differences don't break us."""
    norm = {_norm(c): c for c in df.columns}

    def pick(*keywords, exclude=()):
        for n, original in norm.items():
            if all(k in n for k in keywords) and not any(e in n for e in exclude):
                return original
        return None

    cols = {
        "state": pick("state"), "city": pick("city"), "name": pick("name"),
        "type": pick("type"), "rating": pick("rating"),
        "hours": pick("time", "hrs") or pick("time", "visit"),
        "fee": pick("fee"), "significance": pick("significance"),
    }
    missing = [k for k in ("state", "city", "type", "rating") if cols[k] is None]
    if missing:
        raise ValueError(f"Could not find columns for {missing}. Columns in your file: {list(df.columns)}")
    return cols


def map_interests(*texts):
    """Interest set from free text; empty set when nothing matches."""
    blob = " ".join(str(t).lower() for t in texts if isinstance(t, str))
    return {interest for kw, interest in TYPE_TO_INTEREST.items() if kw in blob}


def aggregate_cities(raw):
    """Attraction rows -> one row per (state, city)."""
    c = find_columns(raw)
    d = pd.DataFrame({
        "state": raw[c["state"]].astype(str).str.strip(),
        "city": raw[c["city"]].astype(str).str.strip(),
        "rating": pd.to_numeric(raw[c["rating"]], errors="coerce"),
        "hours": pd.to_numeric(raw[c["hours"]], errors="coerce") if c["hours"] else 1.0,
        "fee": pd.to_numeric(raw[c["fee"]], errors="coerce").fillna(0) if c["fee"] else 0.0,
    })
    d["interests"] = [map_interests(t, s) for t, s in zip(
        raw[c["type"]], raw[c["significance"]] if c["significance"] else [""] * len(raw))]
    unmapped = sorted(set(raw[c["type"]].astype(str)[d["interests"].apply(len) == 0]))

    d["hours"] = d["hours"].fillna(1.0)
    d = d.dropna(subset=["rating"])
    d = d[(d["interests"].apply(len) > 0) & (d["city"] != "") & (d["city"].str.lower() != "nan")]
    g = d.groupby(["state", "city"], as_index=False).agg(
        rating=("rating", "mean"), hours=("hours", "sum"), fee=("fee", "sum"),
        n_attractions=("rating", "size"),
        interests=("interests", lambda sets: "|".join(sorted(set().union(*sets)))))
    return g, unmapped


def inside_india(lat, lon):
    return INDIA_BBOX[0] <= lat <= INDIA_BBOX[1] and INDIA_BBOX[2] <= lon <= INDIA_BBOX[3]


def build_destinations(raw, geocode, origin):
    """geocode: function('City, State, India') -> (lat, lon) or raises. origin: (lat, lon)."""
    g, unmapped = aggregate_cities(raw)
    lats, lons, bad = [], [], []
    for _, r in g.iterrows():
        try:
            lat, lon = geocode(f"{r['city']}, {r['state']}, India")
            if not inside_india(lat, lon):
                raise ValueError("outside India")
        except Exception as exc:
            lat, lon = float("nan"), float("nan")
            bad.append(f"{r['city']}, {r['state']} ({exc})")
        lats.append(lat)
        lons.append(lon)
    g["latitude"], g["longitude"] = lats, lons
    g = g.dropna(subset=["latitude", "longitude"]).copy()

    g["recommended_days"] = g["hours"].apply(lambda h: int(min(5, max(1, math.ceil(h / HOURS_PER_DAY)))))
    straight = haversine_km(origin[0], origin[1], g["latitude"], g["longitude"])
    g["travel_cost"] = (2 * straight * DETOUR_FACTOR * RUPEES_PER_KM / 100).round().astype(int) * 100
    g["travel_cost"] = g["travel_cost"].clip(lower=MIN_TRAVEL_COST)
    g["daily_cost"] = ((BASE_DAILY_COST + g["fee"] / g["recommended_days"]) / 50).round().astype(int) * 50
    g["rating"] = g["rating"].round(2)

    g = g.sort_values(["state", "city"]).reset_index(drop=True)
    g["destination_id"] = range(1, len(g) + 1)
    g["destination_name"] = g["city"]
    g["data_source"] = "Kaggle saketk511/travel-dataset-guide-to-indias-must-see-places (ratings, fees, hours)"
    g["cost_source"] = "ASSUMPTION_NOT_VERIFIED"
    out = g[REQUIRED_DEST_COLUMNS + ["n_attractions", "data_source", "cost_source"]]
    return out, unmapped, bad


def cached_geocoder(cache_path="data/geocode_cache.csv", pause=1.1):
    """Real geocoder (OSMnx/Nominatim) with an on-disk cache and a polite delay."""
    from src.route_network import geocode_place
    cache = {}
    p = Path(cache_path)
    if p.exists():
        cache = {r.query: (r.lat, r.lon) for r in pd.read_csv(p).itertuples()}

    def geocode(query):
        if query in cache:
            return cache[query]
        time.sleep(pause)                                    # Nominatim: max ~1 request/second
        cache[query] = geocode_place(query)
        pd.DataFrame([(q, a, b) for q, (a, b) in cache.items()],
                     columns=["query", "lat", "lon"]).to_csv(p, index=False)
        return cache[query]
    return geocode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to the Kaggle 'Top Indian Places to Visit.csv'")
    ap.add_argument("--origin", default="Chennai, India", help="reference origin for travel_cost")
    ap.add_argument("--output", default="data/destinations.csv")
    a = ap.parse_args()

    raw = pd.read_csv(a.input)
    print(f"Read {len(raw)} attractions. Geocoding (slow the first time, cached afterwards)...")
    geocode = cached_geocoder()
    origin = geocode(a.origin)
    out, unmapped, bad = build_destinations(raw, geocode, origin)

    out_path = Path(a.output)
    if out_path.exists():                                    # never destroy the sample file silently
        backup = out_path.with_name("destinations_sample_backup.csv")
        if not backup.exists():
            shutil.copy(out_path, backup)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} destinations -> {out_path}")
    if unmapped:
        print(f"Types with no interest mapping (edit TYPE_TO_INTEREST): {unmapped}")
    if bad:
        print(f"{len(bad)} places could not be geocoded / were outside India and were dropped:")
        for b in bad[:20]:
            print("   -", b)
    print("NEXT: spot-check coordinates, then run  python generate_sample_data.py --preferences-only")


if __name__ == "__main__":
    sys.exit(main())
