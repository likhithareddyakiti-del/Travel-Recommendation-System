import unittest

import networkx as nx

from src.data_preprocessing import haversine_km
from src.dijkstra import NoPathError
from src.route_network import RouteDataError, bbox_for_points, compute_route, nearest_node


def grid_graph(rows=4, cols=4, step=0.01):
    """A small fake 'road network': a grid of two-way streets near Chennai."""
    G = nx.MultiDiGraph()
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c), y=13.0 + r * step, x=80.0 + c * step)
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 < rows and c2 < cols:
                    a, b = G.nodes[(r, c)], G.nodes[(r2, c2)]
                    w = float(haversine_km(a["y"], a["x"], b["y"], b["x"])) * 1000
                    G.add_edge((r, c), (r2, c2), length=w)
                    G.add_edge((r2, c2), (r, c), length=w)
    return G


class TestRouting(unittest.TestCase):
    def test_route_matches_networkx_and_is_not_shorter_than_straight_line(self):
        G = grid_graph()
        start, dest = (13.0, 80.0), (13.03, 80.03)
        r = compute_route(G, start, dest)
        self.assertTrue(r["matches_networkx"])
        straight = float(haversine_km(*start, *dest))
        self.assertGreaterEqual(r["distance_km"] + 1e-9, straight)     # road >= straight line
        self.assertEqual(r["coords"][0], (13.0, 80.0))
        self.assertEqual(r["coords"][-1], (13.03, 80.03))

    def test_snapping(self):
        node, snap = nearest_node(grid_graph(), 13.0101, 80.0099)
        self.assertEqual(node, (1, 1))
        self.assertLess(snap, 0.2)

    def test_disconnected_graph_gives_clear_error(self):
        G = grid_graph(2, 2)
        G.add_node("island", y=13.005, x=80.0305)       # no edges
        with self.assertRaises(NoPathError):
            compute_route(G, (13.0, 80.0), (13.005, 80.0305))

    def test_point_far_from_any_road_is_rejected(self):
        with self.assertRaises(RouteDataError):
            compute_route(grid_graph(), (13.0, 80.0), (11.97, 92.98))   # far away (islands)

    def test_bbox_covers_both_endpoints(self):
        west, south, east, north = bbox_for_points((13.08, 80.27), (9.93, 78.12), padding_km=10)
        for lat, lon in [(13.08, 80.27), (9.93, 78.12)]:
            self.assertTrue(south < lat < north and west < lon < east)


if __name__ == "__main__":
    unittest.main()
