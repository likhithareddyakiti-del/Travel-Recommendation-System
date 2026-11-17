import unittest

from src.dijkstra import NoPathError, graph_from_networkx, shortest_path

# Hand-checkable graph:
#   A --7-- B --3-- C         shortest A->E:  A-C-D-E = 9+2+... see below
#   A --9-- C, A --14-- F, B --10-- D, C --11-- D, C --2-- F, D --6-- E, F --9-- E
GRAPH_UNDIRECTED_EDGES = [("A", "B", 7), ("A", "C", 9), ("A", "F", 14), ("B", "C", 10),
                          ("B", "D", 15), ("C", "D", 11), ("C", "F", 2), ("D", "E", 6),
                          ("E", "F", 9)]


def build(edges):
    g = {}
    for u, v, w in edges:
        g.setdefault(u, []).append((v, w))
        g.setdefault(v, []).append((u, w))
    return g


class TestDijkstra(unittest.TestCase):
    def test_known_shortest_paths(self):
        g = build(GRAPH_UNDIRECTED_EDGES)
        self.assertEqual(shortest_path(g, "A", "E"), (20, ["A", "C", "F", "E"]))   # 9+2+9
        self.assertEqual(shortest_path(g, "A", "D"), (20, ["A", "C", "D"]))       # 9+11
        self.assertEqual(shortest_path(g, "A", "F"), (11, ["A", "C", "F"]))       # 9+2

    def test_same_node(self):
        self.assertEqual(shortest_path(build(GRAPH_UNDIRECTED_EDGES), "A", "A"), (0.0, ["A"]))

    def test_unreachable_raises_clear_error(self):
        g = {"A": [("B", 1)], "B": [("A", 1)], "X": [("Y", 1)], "Y": [("X", 1)]}
        with self.assertRaises(NoPathError):
            shortest_path(g, "A", "X")

    def test_one_way_edges_respected(self):
        g = {"A": [("B", 1)], "B": []}
        self.assertEqual(shortest_path(g, "A", "B")[0], 1)
        with self.assertRaises(NoPathError):
            shortest_path(g, "B", "A")

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            shortest_path({"A": [("B", -1)]}, "A", "B")

    def test_missing_nodes(self):
        with self.assertRaises(KeyError):
            shortest_path({"A": [("B", 1)]}, "Z", "B")

    def test_agrees_with_networkx(self):
        import networkx as nx
        G = nx.MultiDiGraph()
        for u, v, w in GRAPH_UNDIRECTED_EDGES:
            G.add_edge(u, v, length=w)
            G.add_edge(v, u, length=w)
        G.add_edge("A", "B", length=3)                      # parallel, shorter edge
        ours = shortest_path(graph_from_networkx(G), "A", "E")[0]
        self.assertEqual(ours, nx.shortest_path_length(G, "A", "E", weight="length"))


if __name__ == "__main__":
    unittest.main()
