"""
src/route_network.py
--------------------
Road-network side of the project (OpenStreetMap data via OSMnx).

IMPORTANT: straight-line (haversine) distance != road distance. Roads bend, climb and
detour around rivers/mountains, so road distance is almost always longer.

What this module does
  1. geocode_place()      : place name -> (lat, lon) using Nominatim (needs internet)
  2. load_road_graph()    : downloads ONE graph whose bounding box covers BOTH the start
                            and the destination (+ padding), then caches it on disk
  3. nearest_node()       : snaps a coordinate to the nearest road-network node
  4. compute_route()      : runs OUR Dijkstra (src/dijkstra.py) on that graph and
                            cross-checks the distance with NetworkX

Steps 3-4 only need NetworkX/NumPy, so they are unit-tested without internet.
OpenStreetMap data (c) OpenStreetMap contributors, ODbL: https://www.openstreetmap.org/copyright
Be polite to public services: results are cached; do not hammer Nominatim/Overpass.
"""
import hashlib
import math
from pathlib import Path

import networkx as nx
import numpy as np

from src.data_preprocessing import haversine_km
from src.dijkstra import NoPathError, graph_from_networkx, shortest_path

CACHE_DIR = "outputs/graph_cache"
# For longer trips we download only major roads (much smaller download)
MAJOR_ROADS_FILTER = '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link"]'
FULL_DETAIL_BELOW_KM = 40        # below this straight-line distance use the full drivable network
MAX_SNAP_KM = 25                 # refuse if the nearest road node is farther than this


class RouteDataError(Exception):
    """Road data could not be downloaded / is unusable (offline, island, too large...)."""


def _import_osmnx():
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RouteDataError("The 'osmnx' package is not installed. Run: pip install osmnx") from exc
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 300
    return ox


def geocode_place(query):
    """'Chennai, India' -> (lat, lon). Raises RouteDataError on failure."""
    ox = _import_osmnx()
    try:
        lat, lon = ox.geocode(query)
        return float(lat), float(lon)
    except Exception as exc:
        raise RouteDataError(f"Could not find '{query}' (offline, or the name was not recognised): {exc}") from exc


def bbox_for_points(p1, p2, padding_km=15):
    """Bounding box (west, south, east, north) covering BOTH points plus padding."""
    lats, lons = [p1[0], p2[0]], [p1[1], p2[1]]
    dlat = padding_km / 111.0
    dlon = padding_km / (111.0 * max(math.cos(math.radians(sum(lats) / 2)), 0.2))
    return (min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat)


def load_road_graph(p1, p2, padding_km=15, cache_dir=CACHE_DIR):
    """Download (or load cached) road graph covering both endpoints."""
    ox = _import_osmnx()
    straight = float(haversine_km(p1[0], p1[1], p2[0], p2[1]))
    major_only = straight > FULL_DETAIL_BELOW_KM
    bbox = bbox_for_points(p1, p2, padding_km)

    key = hashlib.md5(f"{[round(b, 3) for b in bbox]}-{major_only}".encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"graph_{key}.graphml"
    if path.exists():
        return ox.load_graphml(path)

    try:
        if major_only:
            G = ox.graph_from_bbox(bbox, custom_filter=MAJOR_ROADS_FILTER, simplify=True, retain_all=False)
        else:
            G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True, retain_all=False)
    except Exception as exc:
        raise RouteDataError(f"Could not download road data from OpenStreetMap "
                             f"(offline, no roads in the area, or query too large): {exc}") from exc
    if G.number_of_edges() == 0:
        raise RouteDataError("The downloaded area contains no roads.")
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)
    return G


def nearest_node(G, lat, lon):
    """Nearest graph node to (lat, lon) -> (node_id, snap_distance_km)."""
    ids = list(G.nodes)
    ys = np.array([G.nodes[n]["y"] for n in ids], dtype=float)
    xs = np.array([G.nodes[n]["x"] for n in ids], dtype=float)
    d = haversine_km(lat, lon, ys, xs)
    i = int(np.argmin(d))
    return ids[i], float(d[i])


def compute_route(G, start, dest):
    """Shortest road route between two (lat, lon) points on graph G using OUR Dijkstra.

    Returns dict: distance_km, coords [(lat, lon)...], snap_start_km, snap_dest_km,
    networkx_check_km, matches_networkx (bool).
    """
    u, snap_u = nearest_node(G, *start)
    v, snap_v = nearest_node(G, *dest)
    if snap_u > MAX_SNAP_KM or snap_v > MAX_SNAP_KM:
        raise RouteDataError(f"Nearest road is {max(snap_u, snap_v):.0f} km from a point "
                             f"(limit {MAX_SNAP_KM} km). Is it an island or outside the downloaded area?")

    adjacency = graph_from_networkx(G, weight="length")          # lengths are metres
    dist_m, nodes = shortest_path(adjacency, u, v)               # raises NoPathError

    try:
        check_m = nx.shortest_path_length(G, u, v, weight="length")   # validation only
    except nx.NetworkXNoPath:
        check_m = float("nan")
    return {
        "distance_km": dist_m / 1000.0,
        "coords": [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in nodes],
        "snap_start_km": snap_u,
        "snap_dest_km": snap_v,
        "networkx_check_km": check_m / 1000.0,
        "matches_networkx": abs(check_m - dist_m) < 1e-6 * max(1.0, dist_m),
    }


def get_route(start, dest, padding_km=15, max_straight_km=150):
    """Full pipeline: guard -> download graph -> Dijkstra. Retries once with wider padding."""
    straight = float(haversine_km(start[0], start[1], dest[0], dest[1]))
    if straight > max_straight_km:
        raise RouteDataError(f"Straight-line distance {straight:.0f} km exceeds the limit of "
                             f"{max_straight_km} km set for road downloads (public Overpass servers "
                             f"are not meant for huge downloads). Raise the limit at your own risk.")
    try:
        return compute_route(load_road_graph(start, dest, padding_km), start, dest)
    except NoPathError:
        return compute_route(load_road_graph(start, dest, padding_km * 2), start, dest)
