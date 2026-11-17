"""
src/data_preprocessing.py
-------------------------
Loading/cleaning destinations.csv and building the numeric features
that the Machine Learning model needs.

Estimated trip cost (used everywhere in this project):
    estimated_total_cost = travel_cost + daily_cost * recommended_days
  travel_cost : estimated round-trip transport cost (illustrative figure)
  daily_cost  : estimated food + local transport + stay + activities per day
Excluded: shopping, insurance, visa/permit fees, emergencies, tips.
"""
import numpy as np
import pandas as pd

INTEREST_OPTIONS = ["beaches", "history", "nature", "hills", "adventure"]

REQUIRED_DEST_COLUMNS = [
    "destination_id", "destination_name", "state", "latitude", "longitude",
    "interests", "rating", "travel_cost", "daily_cost", "recommended_days",
]

# Columns the ML model is trained on (order matters, keep it fixed).
FEATURE_COLUMNS = [
    "user_budget", "interest_match", "rating", "estimated_total_cost",
    "distance_km", "available_days", "recommended_days",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line (great-circle) distance in km. NOT road distance!
    Works with plain numbers or NumPy arrays / pandas Series."""
    r = 6371.0088  # mean Earth radius in km
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def parse_interests(text):
    """'beaches|nature' -> {'beaches', 'nature'}"""
    return {t.strip().lower() for t in str(text).split("|") if t.strip()}


def interest_vector(interests):
    """Multi-hot encoding: {'beaches','hills'} -> [1,0,0,1,0] (order of INTEREST_OPTIONS)."""
    return np.array([1 if i in interests else 0 for i in INTEREST_OPTIONS])


def interest_match(user_interests, dest_interests):
    """Fraction of the user's interests that the destination offers (0.0 - 1.0)."""
    u = interest_vector(set(user_interests))
    d = interest_vector(parse_interests(dest_interests) if isinstance(dest_interests, str) else set(dest_interests))
    if u.sum() == 0:
        return 0.0
    return float((u * d).sum() / u.sum())


def load_destinations(path="data/destinations.csv"):
    """Read + clean + validate destinations.csv and add estimated_total_cost."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_DEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"destinations.csv is missing columns: {missing}")

    df = df[REQUIRED_DEST_COLUMNS].copy()
    for col in ["destination_name", "state", "interests"]:
        df[col] = df[col].astype(str).str.strip()
    for col in ["latitude", "longitude", "rating", "travel_cost", "daily_cost", "recommended_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates(subset="destination_id").reset_index(drop=True)

    # Range checks -> fail loudly instead of silently training on nonsense
    if not df["latitude"].between(-90, 90).all() or not df["longitude"].between(-180, 180).all():
        raise ValueError("Latitude/longitude out of range in destinations.csv")
    if not df["rating"].between(0, 5).all():
        raise ValueError("Ratings must be between 0 and 5")
    if (df[["travel_cost", "daily_cost"]] < 0).any().any() or (df["recommended_days"] < 1).any():
        raise ValueError("Costs must be >= 0 and recommended_days >= 1")
    for text in df["interests"]:
        unknown = parse_interests(text) - set(INTEREST_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown interest(s) {unknown}; allowed: {INTEREST_OPTIONS}")

    df["recommended_days"] = df["recommended_days"].astype(int)
    df["estimated_total_cost"] = df["travel_cost"] + df["daily_cost"] * df["recommended_days"]
    return df


def build_feature_frame(dest_df, user_budget, user_interests, available_days, start_lat, start_lon):
    """One row per destination for ONE user request. Returns dest_df + feature columns."""
    out = dest_df.copy()
    out["user_budget"] = float(user_budget)
    out["available_days"] = int(available_days)
    out["interest_match"] = out["interests"].apply(lambda s: interest_match(user_interests, s))
    out["distance_km"] = haversine_km(start_lat, start_lon, out["latitude"], out["longitude"])
    return out
