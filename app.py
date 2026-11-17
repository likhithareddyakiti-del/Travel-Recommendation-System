
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    INTEREST_OPTIONS,
    haversine_km,
    load_destinations
)

from src.live_data import (
    LiveDataError,
    get_current_weather
)

from src.ml_recommendation import (
    MODEL_PATH,
    METRICS_PATH,
    load_model,
    train_model
)

from src.recommender import (
    recommend,
    validate_inputs
)

from src.route_network import (
    RouteDataError,
    geocode_place
)

# City-based 2D graph (no OSM required)
from src.route_graph import draw_city_2d_graph


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Travel Recommendation System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM UI DESIGN
# ---------------------------------------------------------

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.main-title {
    font-size: 36px;
    font-weight: 750;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

.subtitle {
    color: #8993A4;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.25);
    padding: 15px;
    border-radius: 12px;
    background: rgba(128,128,128,0.035);
}

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    border-radius: 9px;
    font-weight: 650;
}

div[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.2);
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

@st.cache_data
def get_destinations():

    return load_destinations(
        "data/destinations.csv"
    )


@st.cache_data
def get_attractions():

    path = Path("data/final_attractions.csv")

    if not path.exists():

        raise FileNotFoundError(
            "data/final_attractions.csv not found."
        )

    return pd.read_csv(path)


@st.cache_resource
def get_model():

    if not Path(MODEL_PATH).exists():

        with st.spinner("Training recommendation model..."):
            train_model()

    return load_model()


@st.cache_data(show_spinner=False)
def cached_geocode(query):

    return geocode_place(query)


# ---------------------------------------------------------
# APPLICATION HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '✈️ Travel Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover destinations based on your interests, '
    'budget and available time.'
    '</div>',
    unsafe_allow_html=True
)

with st.expander("About this project"):

    st.write("""
    This application combines:

    - Machine Learning for destination suitability prediction.
    - Knapsack optimization for budget-based destination selection.
    - Dijkstra algorithm for shortest paths in a city network.
    - 2D city network visualization using city names.
    - Straight-line distance calculations using coordinates.
    - Tourist attraction information.
    - Live weather information.
    """)



# ---------------------------------------------------------
# SIDEBAR - USER INPUTS
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## 🧳 Plan Your Trip")

    st.caption(
        "Enter your travel preferences below."
    )

    with st.form("inputs"):

        mode = st.radio(
            "Starting location",
            [
                "Place name (needs internet)",
                "Latitude / longitude"
            ]
        )

        place = st.text_input(
            "Place name",
            value="Chennai, India"
        )

        st.caption("Or enter coordinates:")

        c1, c2 = st.columns(2)

        manual_lat = c1.number_input(
            "Latitude",
            value=13.0827,
            format="%.4f",
            min_value=-90.0,
            max_value=90.0
        )

        manual_lon = c2.number_input(
            "Longitude",
            value=80.2707,
            format="%.4f",
            min_value=-180.0,
            max_value=180.0
        )

        st.divider()

        budget = st.number_input(
            "Total budget (₹)",
            min_value=0,
            value=25000,
            step=500
        )

        interests = st.multiselect(
            "Your interests",
            INTEREST_OPTIONS,
            default=["beaches", "history"]
        )

        days = st.number_input(
            "Available days",
            min_value=0,
            value=4,
            step=1
        )

        st.divider()

        st.markdown("### 🏙️ City Network Graph")

        st.caption(
            "A 2D graph of city names and intermediate cities. "
            "No geographical map or OSM download is required."
        )

        show_graph = st.checkbox(
            "Generate 2D city graph",
            value=True
        )

        st.divider()

        show_weather = st.checkbox(
            "Show live weather",
            value=True
        )

        go = st.form_submit_button(
            "✈️ Find Recommendations",
            use_container_width=True
        )


# ---------------------------------------------------------
# BEFORE RECOMMENDATION
# ---------------------------------------------------------

if not go:

    st.markdown("### Welcome!")

    st.write(
        "Start by entering your trip details in the sidebar. "
        "Your recommendations will be generated using your preferences."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "🎯 **Personalized**\n\n"
            "Recommendations based on your interests."
        )

    with c2:
        st.info(
            "💰 **Budget-aware**\n\n"
            "Knapsack optimization selects affordable options."
        )

    with c3:
        st.info(
            "🏙️ **City network**\n\n"
            "2D graph with city names and intermediate cities."
        )

    if Path(METRICS_PATH).exists():

        with st.expander("Model Evaluation"):

            st.json(
                Path(METRICS_PATH).read_text(
                    encoding="utf-8"
                )
            )

    st.stop()


# ---------------------------------------------------------
# VALIDATE INPUTS
# ---------------------------------------------------------

name_for_check = (
    place if mode.startswith("Place") else "coordinates"
)

errors = validate_inputs(
    name_for_check,
    budget,
    interests,
    days
)

if errors:

    for error in errors:
        st.error(error)

    st.stop()


# ---------------------------------------------------------
# STARTING LOCATION
# ---------------------------------------------------------

if mode.startswith("Place"):

    try:

        start = cached_geocode(
            place.strip()
        )

        source_name = place.strip().split(",")[0].strip()

    except RouteDataError as exc:

        st.error(
            f"{exc}\n\n"
            "Try using latitude and longitude instead."
        )

        st.stop()

else:

    start = (
        float(manual_lat),
        float(manual_lon)
    )

    source_name = "Starting Location"


st.success(
    f"Starting location: {start[0]:.4f}, {start[1]:.4f}"
)


# ---------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------

try:

    destinations = get_destinations()

    attractions = get_attractions()

except Exception as exc:

    st.error(
        f"Dataset loading error: {exc}"
    )

    st.stop()


# ---------------------------------------------------------
# ML RECOMMENDATION + KNAPSACK
# ---------------------------------------------------------

try:

    out = recommend(
        destinations,
        get_model(),
        float(budget),
        interests,
        int(days),
        start[0],
        start[1]
    )

except Exception as exc:

    st.error(
        f"Recommendation error: {exc}"
    )

    st.stop()


selected = out["selected"]

cand = out["candidates"]

ks = out["knapsack"]


if selected.empty:

    st.error(
        "No destination fits your current budget, days and interests. "
        "Try increasing your budget or available days."
    )

    st.stop()


# ---------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Your Trip Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Destinations Selected",
    len(selected)
)

c2.metric(
    "Estimated Total Cost",
    f"₹{ks['total_cost']:,.0f}"
)

c3.metric(
    "Remaining Budget",
    f"₹{ks['remaining_budget']:,.0f}"
)

c4.metric(
    "Total Suitability",
    f"{ks['total_value']:.2f}"
)

st.caption(
    "These are independent destination options, not one continuous journey."
)


# ---------------------------------------------------------
# DESTINATION CARDS
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    '🌍 Recommended Destinations'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Destinations selected by the ML recommendation model "
    "and Knapsack optimization."
)

for _, r in selected.iterrows():

    name = r["destination_name"]

    distance = haversine_km(
        start[0],
        start[1],
        r["latitude"],
        r["longitude"]
    )

    with st.container(border=True):

        left, right = st.columns([3, 1])

        with left:

            st.markdown(
                f"### 📍 {name}"
            )

            st.caption(
                str(r["state"])
            )

            st.write(
                f"**Interests:** "
                f"{str(r['interests']).replace('|', ', ')}"
            )

        with right:

            st.metric(
                "Suitability",
                f"{float(r['suitability']):.3f}"
            )

        a, b, c = st.columns(3)

        a.metric(
            "Estimated Cost",
            f"₹{float(r['estimated_total_cost']):,.0f}"
        )

        b.metric(
            "Duration",
            f"{int(r['recommended_days'])} days"
        )

        c.metric(
            "Straight-line Distance",
            f"{distance:.1f} km"
        )


# ---------------------------------------------------------
# RECOMMENDATION TABLE
# ---------------------------------------------------------

rows = []

for _, r in selected.iterrows():

    distance = haversine_km(
        start[0],
        start[1],
        r["latitude"],
        r["longitude"]
    )

    rows.append({

        "Destination": r["destination_name"],

        "State": r["state"],

        "Interests": str(
            r["interests"]
        ).replace("|", ", "),

        "Suitability": round(
            float(r["suitability"]),
            3
        ),

        "Estimated Cost (₹)": int(
            r["estimated_total_cost"]
        ),

        "Duration (days)": int(
            r["recommended_days"]
        ),

        "Straight-line Distance (km)": round(
            float(distance),
            1
        )

    })


with st.expander("View Recommendation Comparison Table"):

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# 2D CITY NETWORK GRAPH
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    '🏙️ 2D City Network Graph'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "This is a normal 2D city network graph, not a geographical map. "
    "Nodes represent city names. Edges represent connections with "
    "straight-line distances in kilometres."
)

if show_graph:

    for _, r in selected.iterrows():

        destination = str(
            r["destination_name"]
        )

        with st.expander(
            f"{source_name} → {destination}",
            expanded=True
        ):

            try:

                fig = draw_city_2d_graph(
                    destinations=destinations,
                    start=start,
                    destination_lat=float(r["latitude"]),
                    destination_lon=float(r["longitude"]),
                    source_name=source_name,
                    destination_name=destination
                )

                if fig is not None:

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    plt.close(fig)

                else:

                    st.warning(
                        "Could not generate the city graph."
                    )

            except Exception as exc:

                st.error(
                    f"City graph error for {destination}: {exc}"
                )

else:

    st.info(
        "Enable Generate 2D city graph in the sidebar."
    )


# ---------------------------------------------------------
# TOURIST ATTRACTIONS
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    '🏛️ Tourist Attractions'
    '</div>',
    unsafe_allow_html=True
)

if "City" not in attractions.columns:

    st.warning(
        "The final attraction dataset does not contain a City column."
    )

else:

    attractions["_city_key"] = (
        attractions["City"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    for _, r in selected.iterrows():

        destination = str(
            r["destination_name"]
        ).strip()

        city_attractions = attractions[
            attractions["_city_key"] == destination.casefold()
        ]

        with st.expander(
            f"Attractions in {destination}"
        ):

            if city_attractions.empty:

                st.info(
                    "No matching attraction records found."
                )

            else:

                display_columns = [

                    c for c in [
                        "Name",
                        "Type",
                        "Google review rating",
                        "Entrance Fee in INR",
                        "Best Time to visit",
                        "time needed to visit in hrs"
                    ]

                    if c in city_attractions.columns

                ]

                st.write(
                    f"Found {len(city_attractions)} attractions."
                )

                st.dataframe(
                    city_attractions[display_columns],
                    use_container_width=True,
                    hide_index=True
                )


# ---------------------------------------------------------
# LIVE WEATHER
# ---------------------------------------------------------

if show_weather:

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🌤️ Live Weather'
        '</div>',
        unsafe_allow_html=True
    )

    wrows = []

    for _, r in selected.iterrows():

        try:

            w = get_current_weather(
                r["latitude"],
                r["longitude"]
            )

            wrows.append({

                "Destination": r["destination_name"],

                "Temperature (°C)": w["temperature_c"],

                "Conditions": w["description"],

                "Rain (mm)": w["precipitation_mm"],

                "Wind (km/h)": w["wind_kmh"],

                "Observed At": w["time"]

            })

        except LiveDataError as exc:

            wrows.append({

                "Destination": r["destination_name"],

                "Conditions": f"Unavailable: {exc}"

            })

    st.dataframe(
        pd.DataFrame(wrows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Weather data from Open-Meteo (CC BY 4.0). "
        "Weather is informational and is not used by the ML model."
    )


# ---------------------------------------------------------
# ALL CANDIDATES
# ---------------------------------------------------------

with st.expander(
    "View All ML Candidates Before Knapsack"
):

    st.dataframe(

        cand[
            [
                "destination_name",
                "interests",
                "suitability",
                "estimated_total_cost",
                "recommended_days"
            ]
        ],

        hide_index=True,

        use_container_width=True

    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    "<div style='text-align:center;color:#8993A4;font-size:13px;'>"
    "Travel Recommendation System | Machine Learning • "
    "Knapsack • Dijkstra • 2D City Network"
    "</div>",
    unsafe_allow_html=True
)
