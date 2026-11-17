
"""
route_graph.py
--------------
2D City Network Graph

Displays:
1. Starting city
2. Intermediate city names
3. Recommended destination
4. Straight-line distances between connected cities

Uses:
- City coordinates from destinations.csv
- Haversine distance
- NetworkX Dijkstra algorithm

Does not use OSM or geographical maps.
"""

import math

import networkx as nx
import matplotlib.pyplot as plt

from src.data_preprocessing import haversine_km


def draw_city_2d_graph(
    destinations,
    start,
    destination_lat,
    destination_lon,
    source_name,
    destination_name
):

    required_columns = {
        "destination_name",
        "latitude",
        "longitude"
    }

    if not required_columns.issubset(destinations.columns):
        raise ValueError(
            "Dataset must contain destination_name, latitude and longitude."
        )

    # -----------------------------------------------------
    # START AND DESTINATION
    # -----------------------------------------------------

    start_lat = float(start[0])
    start_lon = float(start[1])

    end_lat = float(destination_lat)
    end_lon = float(destination_lon)

    source_name = str(source_name).strip()
    destination_name = str(destination_name).strip()

    direct_distance = haversine_km(
        start_lat,
        start_lon,
        end_lat,
        end_lon
    )

    if direct_distance <= 0:
        raise ValueError("Starting city and destination are identical.")

    # -----------------------------------------------------
    # CLEAN CITY DATA
    # -----------------------------------------------------

    cities = destinations[
        ["destination_name", "latitude", "longitude"]
    ].copy()

    cities["destination_name"] = (
        cities["destination_name"].astype(str).str.strip()
    )

    cities["latitude"] = pd_to_numeric(cities["latitude"])
    cities["longitude"] = pd_to_numeric(cities["longitude"])

    cities = cities.dropna()

    cities = cities[
        cities["latitude"].between(-90, 90)
        & cities["longitude"].between(-180, 180)
    ]

    cities = cities.drop_duplicates(
        subset=["destination_name"]
    )

    excluded = {
        source_name.casefold(),
        destination_name.casefold()
    }

    cities = cities[
        ~cities["destination_name"].str.casefold().isin(excluded)
    ]

    # -----------------------------------------------------
    # FIND CITIES BETWEEN START AND DESTINATION
    # -----------------------------------------------------

    # Convert coordinates to approximate local 2D coordinates.
    mean_lat = math.radians((start_lat + end_lat) / 2)

    def xy(lat, lon):

        x = lon * math.cos(mean_lat)
        y = lat

        return x, y

    sx, sy = xy(start_lat, start_lon)
    ex, ey = xy(end_lat, end_lon)

    dx = ex - sx
    dy = ey - sy

    line_length_squared = dx * dx + dy * dy

    candidates = []

    # Maximum distance from the direct line.
    corridor_width_km = max(
        40,
        direct_distance * 0.20
    )

    for _, row in cities.iterrows():

        name = row["destination_name"]

        lat = float(row["latitude"])
        lon = float(row["longitude"])

        x, y = xy(lat, lon)

        # Position along the start-destination line.
        t = (
            ((x - sx) * dx + (y - sy) * dy)
            / line_length_squared
        )

        # Only cities between start and destination.
        if not 0.08 < t < 0.92:
            continue

        # Perpendicular distance from the direct line.
        projected_x = sx + t * dx
        projected_y = sy + t * dy

        perpendicular_degrees = math.sqrt(
            (x - projected_x) ** 2
            + (y - projected_y) ** 2
        )

        perpendicular_km = perpendicular_degrees * 111.2

        if perpendicular_km > corridor_width_km:
            continue

        candidates.append({
            "name": name,
            "latitude": lat,
            "longitude": lon,
            "position": t,
            "perpendicular_km": perpendicular_km
        })

    # Sort cities in travel direction.
    candidates.sort(
        key=lambda item: item["position"]
    )

    # Select up to five well-spaced intermediate cities.
    intermediate_cities = []

    for city in candidates:

        if len(intermediate_cities) >= 5:
            break

        if not intermediate_cities:
            intermediate_cities.append(city)
            continue

        previous_position = intermediate_cities[-1]["position"]

        if city["position"] - previous_position >= 0.10:
            intermediate_cities.append(city)

    # -----------------------------------------------------
    # CREATE CITY GRAPH
    # -----------------------------------------------------

    graph = nx.Graph()

    SOURCE = "START"
    DESTINATION = "END"

    graph.add_node(
        SOURCE,
        label=source_name,
        category="source",
        latitude=start_lat,
        longitude=start_lon
    )

    route_city_ids = []

    for index, city in enumerate(intermediate_cities):

        city_id = f"CITY_{index}"

        graph.add_node(
            city_id,
            label=city["name"],
            category="intermediate",
            latitude=city["latitude"],
            longitude=city["longitude"]
        )

        route_city_ids.append(city_id)

    graph.add_node(
        DESTINATION,
        label=destination_name,
        category="destination",
        latitude=end_lat,
        longitude=end_lon
    )

    # -----------------------------------------------------
    # DISTANCE CALCULATION
    # -----------------------------------------------------

    def city_distance(node1, node2):

        a = graph.nodes[node1]
        b = graph.nodes[node2]

        return float(
            haversine_km(
                a["latitude"],
                a["longitude"],
                b["latitude"],
                b["longitude"]
            )
        )

    # -----------------------------------------------------
    # CONNECT CITIES IN ORDER
    # -----------------------------------------------------

    ordered_nodes = (
        [SOURCE]
        + route_city_ids
        + [DESTINATION]
    )

    for u, v in zip(
        ordered_nodes[:-1],
        ordered_nodes[1:]
    ):

        distance = city_distance(u, v)

        graph.add_edge(
            u,
            v,
            weight=distance
        )

    # -----------------------------------------------------
    # DIJKSTRA SHORTEST PATH
    # -----------------------------------------------------

    route_nodes = nx.dijkstra_path(
        graph,
        SOURCE,
        DESTINATION,
        weight="weight"
    )

    route_graph = graph.subgraph(
        route_nodes
    ).copy()

    total_distance = nx.dijkstra_path_length(
        graph,
        SOURCE,
        DESTINATION,
        weight="weight"
    )

    # -----------------------------------------------------
    # NORMAL 2D GRAPH LAYOUT
    # -----------------------------------------------------

    positions = nx.spring_layout(
        route_graph,
        seed=42,
        k=1.5,
        iterations=150
    )

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    intermediate_nodes = [
        node for node in route_nodes
        if node not in [SOURCE, DESTINATION]
    ]

    # Intermediate cities.
    if intermediate_nodes:

        nx.draw_networkx_nodes(
            route_graph,
            positions,
            nodelist=intermediate_nodes,
            node_color="orange",
            node_size=1400,
            edgecolors="black",
            ax=ax
        )

    # Starting city.
    nx.draw_networkx_nodes(
        route_graph,
        positions,
        nodelist=[SOURCE],
        node_color="green",
        node_size=1900,
        edgecolors="black",
        ax=ax
    )

    # Destination.
    nx.draw_networkx_nodes(
        route_graph,
        positions,
        nodelist=[DESTINATION],
        node_color="red",
        node_size=1900,
        edgecolors="black",
        ax=ax
    )

    # Connections.
    nx.draw_networkx_edges(
        route_graph,
        positions,
        width=2.5,
        edge_color="steelblue",
        ax=ax
    )

    # City names only.
    labels = nx.get_node_attributes(
        route_graph,
        "label"
    )

    nx.draw_networkx_labels(
        route_graph,
        positions,
        labels=labels,
        font_size=9,
        font_weight="bold",
        ax=ax
    )

    # Distances between connected cities.
    edge_labels = {
        (u, v): f"{data['weight']:.1f} km"
        for u, v, data in route_graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        route_graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax
    )

    ax.set_title(
        f"2D City Network: {source_name} to {destination_name}\n"
        f"Intermediate Cities: {len(intermediate_nodes)} | "
        f"Total Straight-line Distance: {total_distance:.2f} km",
        fontsize=14,
        fontweight="bold"
    )

    ax.axis("off")

    plt.tight_layout()

    return fig


def pd_to_numeric(series):

    import pandas as pd

    return pd.to_numeric(
        series,
        errors="coerce"
    )
