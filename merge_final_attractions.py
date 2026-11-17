"""
merge_final_attractions.py

Creates data/final_attractions.csv by combining:
1. data/raw/Top Indian Places to Visit.csv  (325 attraction records)
2. data/destinations.csv                   (city-level coordinates and estimates)

Run from the project root:
    python merge_final_attractions.py
"""

from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "Top Indian Places to Visit.csv"
DEST_FILE = ROOT / "data" / "destinations.csv"
OUTPUT_FILE = ROOT / "data" / "final_attractions.csv"


def clean_text(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_state(value):
    key = clean_text(value)
    aliases = {
        "maharastra": "maharashtra",
        "orissa": "odisha",
        "pondicherry": "puducherry",
        "uttaranchal": "uttarakhand",
    }
    return aliases.get(key, key)


def clean_city(value):
    key = clean_text(value)
    aliases = {
        "bangalore": "bengaluru",
        "bengaluru": "bengaluru",
        "prayagraj": "allahabad",
        "allahabad": "allahabad",
        "dalhousie": "dalhousie",
    }
    return aliases.get(key, key)


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Original Kaggle CSV not found: {RAW_FILE}")
    if not DEST_FILE.exists():
        raise FileNotFoundError(f"Destination CSV not found: {DEST_FILE}")

    attractions = pd.read_csv(RAW_FILE, encoding="utf-8-sig")
    destinations = pd.read_csv(DEST_FILE, encoding="utf-8-sig")

    required_raw = {"City", "State", "Name"}
    required_dest = {
        "destination_name", "state", "latitude", "longitude",
        "travel_cost", "daily_cost", "recommended_days",
        "n_attractions", "rating", "interests", "cost_source"
    }
    missing_raw = required_raw - set(attractions.columns)
    missing_dest = required_dest - set(destinations.columns)
    if missing_raw:
        raise ValueError(f"Missing columns in Kaggle file: {sorted(missing_raw)}")
    if missing_dest:
        raise ValueError(f"Missing columns in destinations.csv: {sorted(missing_dest)}")

    # Standardize city/state names for matching without changing attraction names.
    attractions["city_clean"] = attractions["City"].map(clean_city)
    attractions["state_clean"] = attractions["State"].map(clean_state)
    destinations["city_clean"] = destinations["destination_name"].map(clean_city)
    destinations["state_clean"] = destinations["state"].map(clean_state)

    # Use one row per normalized city/state. For duplicate aliases such as
    # Bangalore/Bengaluru, combine conservatively: keep the first coordinate,
    # maximum daily estimate, maximum travel estimate, and maximum recommended days.
    destinations = destinations.sort_values("destination_id" if "destination_id" in destinations.columns else "destination_name")
    dest_lookup = (
        destinations.groupby(["city_clean", "state_clean"], as_index=False)
        .agg(
            matched_destination=("destination_name", "first"),
            destination_state=("state", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            destination_interests=("interests", lambda s: "|".join(sorted({
                item.strip() for value in s.dropna()
                for item in str(value).split("|") if item.strip()
            }))),
            destination_rating=("rating", "mean"),
            estimated_travel_cost_inr=("travel_cost", "max"),
            estimated_daily_cost_inr=("daily_cost", "max"),
            recommended_days=("recommended_days", "max"),
            destination_attraction_count=("n_attractions", "max"),
            cost_source=("cost_source", "first"),
        )
    )

    merged = attractions.merge(
        dest_lookup,
        on=["city_clean", "state_clean"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    merged["destination_match_status"] = merged["_merge"].map({
        "both": "matched_city_level",
        "left_only": "unmatched_city_or_state",
        "right_only": "destination_only",
    })
    merged.drop(columns=["_merge"], inplace=True)

    # A combined planning estimate; not a verified quote.
    merged["estimated_trip_budget_inr"] = (
        merged["estimated_travel_cost_inr"]
        + merged["estimated_daily_cost_inr"] * merged["recommended_days"]
    )
    merged["budget_note"] = merged["destination_match_status"].map(
        lambda status: (
            "Estimated only; not verified. Travel estimate is based on the "
            "project's Chennai-origin assumptions; daily cost is a planning estimate."
            if status == "matched_city_level"
            else "No matching city/state budget record; estimate not available."
        )
    )
    merged["coordinate_level"] = merged["latitude"].apply(
        lambda value: "city_approximate" if pd.notna(value) else "not_available"
    )

    # Keep original Kaggle columns first, followed by the added project fields.
    added_columns = [
        "matched_destination", "destination_state",
        "latitude", "longitude", "coordinate_level",
        "destination_interests", "destination_rating",
        "estimated_travel_cost_inr", "estimated_daily_cost_inr",
        "recommended_days", "estimated_trip_budget_inr",
        "destination_attraction_count", "cost_source",
        "budget_note", "destination_match_status",
    ]
    original_columns = list(attractions.columns)
    original_columns = [c for c in original_columns if c not in {"city_clean", "state_clean"}]
    output_columns = original_columns + added_columns
    merged[output_columns].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    matched = int((merged["destination_match_status"] == "matched_city_level").sum())
    unmatched = len(merged) - matched
    print(f"Original attraction rows: {len(attractions)}")
    print(f"Final attraction rows:    {len(merged)}")
    print(f"Matched city records:     {matched}")
    print(f"Unmatched records:        {unmatched}")
    print(f"Saved: {OUTPUT_FILE}")

    if len(merged) != len(attractions):
        raise RuntimeError("Row count changed unexpectedly; check the merge.")
    if unmatched:
        print("\nUnmatched City/State pairs (review these):")
        print(
            merged.loc[merged["destination_match_status"] != "matched_city_level",
                       ["City", "State"]]
            .drop_duplicates()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
