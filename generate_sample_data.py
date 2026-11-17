
"""
generate_sample_data.py
-----------------------
Generates synthetic preference training data using destination records.

IMPORTANT:
- Normal run: creates 27 sample destinations and synthetic preferences.
- --preferences-only: preserves imported destinations and generates preferences.
- Synthetic labels are not real tourist feedback.
"""

import sys
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    INTEREST_OPTIONS,
    build_feature_frame,
    load_destinations,
)

SEED = 42

# Sample destinations (used only when --preferences-only is NOT specified)
DESTINATIONS = [
    (1, "Calangute Beach", "Goa", 15.5439, 73.7553, "beaches|adventure", 4.3, 6000, 2500, 3),
    (2, "Munnar", "Kerala", 10.0889, 77.0595, "nature|hills", 4.5, 3500, 2200, 3),
    (3, "Ooty", "Tamil Nadu", 11.4102, 76.6950, "hills|nature", 4.2, 2500, 2000, 2),
    (4, "Kodaikanal", "Tamil Nadu", 10.2381, 77.4892, "hills|nature", 4.3, 2500, 2000, 2),
    (5, "Mahabalipuram", "Tamil Nadu", 12.6208, 80.1923, "history|beaches", 4.1, 800, 1500, 1),
    (6, "Hampi", "Karnataka", 15.3350, 76.4600, "history|adventure", 4.6, 3500, 1800, 3),
    (7, "Mysuru", "Karnataka", 12.2958, 76.6394, "history", 4.4, 2200, 1800, 2),
    (8, "Puducherry", "Puducherry", 11.9416, 79.8083, "beaches|history", 4.2, 1200, 1900, 2),
    (9, "Madurai", "Tamil Nadu", 9.9252, 78.1198, "history", 4.3, 2000, 1500, 2),
    (10, "Rameswaram", "Tamil Nadu", 9.2876, 79.3129, "history|beaches", 4.2, 2800, 1500, 2),
    (11, "Kanyakumari", "Tamil Nadu", 8.0883, 77.5385, "beaches|history", 4.1, 3000, 1600, 2),
    (12, "Madikeri (Coorg)", "Karnataka", 12.4244, 75.7382, "nature|hills|adventure", 4.4, 3500, 2200, 3),
    (13, "Gokarna", "Karnataka", 14.5479, 74.3188, "beaches|nature", 4.4, 4500, 1800, 3),
    (14, "Varkala", "Kerala", 8.7379, 76.7163, "beaches", 4.3, 3200, 1900, 2),
    (15, "Alappuzha", "Kerala", 9.4981, 76.3388, "nature", 4.4, 3000, 2300, 2),
    (16, "Thekkady", "Kerala", 9.6000, 77.1667, "nature|adventure", 4.3, 3000, 2000, 2),
    (17, "Yercaud", "Tamil Nadu", 11.7753, 78.2090, "hills|nature", 4.0, 1500, 1700, 2),
    (18, "Tirupati", "Andhra Pradesh", 13.6288, 79.4192, "history", 4.5, 800, 1200, 1),
    (19, "Lonavala", "Maharashtra", 18.7546, 73.4062, "hills|nature", 4.0, 7000, 2300, 2),
    (20, "Jaipur", "Rajasthan", 26.9124, 75.7873, "history", 4.5, 11000, 2600, 3),
    (21, "Udaipur", "Rajasthan", 24.5854, 73.7125, "history|nature", 4.6, 11500, 2600, 3),
    (22, "Manali", "Himachal Pradesh", 32.2396, 77.1887, "hills|adventure|nature", 4.5, 14000, 2800, 4),
    (23, "Rishikesh", "Uttarakhand", 30.0869, 78.2676, "adventure|nature", 4.4, 12000, 2200, 3),
    (24, "Varanasi", "Uttar Pradesh", 25.3176, 82.9739, "history", 4.4, 9500, 1800, 2),
    (25, "Darjeeling", "West Bengal", 27.0360, 88.2627, "hills|nature", 4.4, 14000, 2400, 3),
    (26, "Havelock Island", "Andaman and Nicobar Islands", 11.9760, 92.9870, "beaches|adventure", 4.6, 16000, 3500, 4),
    (27, "Jaisalmer", "Rajasthan", 26.9157, 70.9083, "history|adventure", 4.4, 12500, 2300, 3),
]

COLUMNS = [
    "destination_id", "destination_name", "state",
    "latitude", "longitude", "interests", "rating",
    "travel_cost", "daily_cost", "recommended_days",
]

START_POINTS = [
    (13.08, 80.27),
    (12.97, 77.59),
    (17.38, 78.48),
    (19.07, 72.88),
    (28.61, 77.21),
    (22.57, 88.36),
    (9.93, 76.27),
    (18.52, 73.86),
]

# Columns used in the suitability calculation must be numeric.
NUMERIC_LABEL_COLUMNS = [
    "interest_match",
    "rating",
    "distance_km",
    "estimated_total_cost",
    "user_budget",
    "recommended_days",
    "available_days",
]


def rule_label(df):
    """Apply the synthetic suitability rule."""

    fit = (
        0.5 * df["interest_match"]
        + 0.3 * df["rating"] / 5
        + 0.2 * (
            1 - np.minimum(df["distance_km"], 2500) / 2500
        )
    )

    suitable = (
        (df["estimated_total_cost"] <= df["user_budget"])
        & (df["recommended_days"] <= df["available_days"])
        & (df["interest_match"] > 0)
        & (fit >= 0.5)
    )

    return suitable.astype(int)


def main():
    rng = np.random.default_rng(SEED)

    # Preserve imported destinations when --preferences-only is used.
    if "--preferences-only" not in sys.argv:
        pd.DataFrame(
            DESTINATIONS, columns=COLUMNS
        ).to_csv("data/destinations.csv", index=False)

    dest = load_destinations("data/destinations.csv")

    if dest.empty:
        raise ValueError("No destinations found in data/destinations.csv")

    frames = []

    # Generate synthetic user preference records.
    for user_id in range(1, 501):
        k = int(rng.integers(1, 4))

        interests = list(
            rng.choice(INTEREST_OPTIONS, size=k, replace=False)
        )

        budget = int(rng.integers(30, 301)) * 100
        days = int(rng.integers(1, 8))

        lat, lon = START_POINTS[
            int(rng.integers(len(START_POINTS)))
        ]

        features = build_feature_frame(
            dest, budget, interests, days, lat, lon
        )

        features["user_id"] = user_id
        frames.append(features)

    data = pd.concat(frames, ignore_index=True)

    # Convert label-related features to numeric values.
    for column in NUMERIC_LABEL_COLUMNS:
        if column not in data.columns:
            raise ValueError(
                f"Required feature column is missing: {column}"
            )

        original_values = data[column].copy()

        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

        if data[column].isna().any():
            bad_values = original_values[
                data[column].isna()
            ].head(5).tolist()

            raise ValueError(
                f"Column '{column}' contains non-numeric values. "
                f"Examples: {bad_values}"
            )

    # Check that all expected model features are present.
    missing_features = [
        col for col in FEATURE_COLUMNS if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model feature columns: {missing_features}"
        )

    print("\nFeature data types after conversion:")
    print(data[FEATURE_COLUMNS].dtypes)

    # Generate synthetic suitability labels.
    data["suitable"] = rule_label(data)

    # Add 3% random label noise (only once).
    flip = rng.random(len(data)) < 0.03
    data.loc[flip, "suitable"] = (
        1 - data.loc[flip, "suitable"]
    )

    data["label_source"] = "synthetic_rule_based"

    # Save the generated preference dataset.
    output_columns = (
        ["user_id"] + FEATURE_COLUMNS
        + ["suitable", "label_source"]
    )

    output = data[output_columns].round(4)

    output.to_csv(
        "data/preference_data.csv", index=False
    )

    print("\nDATA GENERATION SUCCESSFUL")
    print(f"Destinations: {len(dest)}")
    print(f"Preference records: {len(output)}")
    print(
        f"Labelled suitable: {output['suitable'].mean():.1%}"
    )
    print("Label source: synthetic rule-based")
    print("Saved: data/preference_data.csv")


if __name__ == "__main__":
    main()
