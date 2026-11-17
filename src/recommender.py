"""
src/recommender.py
------------------
Glue between the ML scores and the Knapsack selection (no Streamlit code here,
so it can be tested). Routing is done separately in route_network.py.
"""
from src.data_preprocessing import INTEREST_OPTIONS, build_feature_frame
from src.knapsack import knapsack_select
from src.ml_recommendation import score_destinations

MIN_SUITABILITY = 0.5     # options scored below this are not offered


def validate_inputs(place, budget, interests, days):
    """Return a list of human-readable error messages (empty list = inputs are fine)."""
    errors = []
    if not str(place).strip():
        errors.append("Please enter a starting location (or switch to coordinates).")
    if budget is None or budget < 1000:
        errors.append("Budget must be at least Rs 1,000.")
    if not interests:
        errors.append("Select at least one interest.")
    elif not set(interests) <= set(INTEREST_OPTIONS):
        errors.append(f"Interests must be from: {', '.join(INTEREST_OPTIONS)}.")
    if days is None or days < 1:
        errors.append("Available days must be at least 1.")
    return errors


def recommend(dest_df, model, budget, interests, days, start_lat, start_lon,
              min_score=MIN_SUITABILITY, unit=100):
    """
    1. build features for every destination   2. ML suitability score
    3. hard filters: interest overlap, recommended_days <= days, cost <= budget, score >= min
    4. 0/1 knapsack over the remaining options
    Returns dict with: scored (all), candidates (after filters), selected, knapsack result.
    """
    scored = build_feature_frame(dest_df, budget, interests, days, start_lat, start_lon)
    scored["suitability"] = score_destinations(model, scored)

    mask = ((scored["interest_match"] > 0)
            & (scored["recommended_days"] <= days)
            & (scored["estimated_total_cost"] <= budget)
            & (scored["suitability"] >= min_score))
    candidates = scored[mask].sort_values("suitability", ascending=False).reset_index(drop=True)

    result = knapsack_select(candidates["estimated_total_cost"].tolist(),
                             candidates["suitability"].tolist(), budget, unit)
    selected = candidates.iloc[result["selected_indices"]].reset_index(drop=True)
    return {"scored": scored, "candidates": candidates, "selected": selected, "knapsack": result}
