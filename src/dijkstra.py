"""
src/dijkstra.py
---------------
Dijkstra's shortest-path algorithm written from scratch.

Graph format (adjacency list):   {node: [(neighbour, weight), ...], ...}
Data structure: a binary min-heap (heapq) as the priority queue.

Idea: always settle the unvisited node with the smallest known distance. Because all
weights are >= 0, that distance can never be improved later.

Complexity: O((V + E) log V) with a binary heap. Negative weights are NOT allowed.
Dijkstra finds the shortest path between two points. It does not decide the order
of visiting several destinations.
"""
import heapq
import math


class NoPathError(Exception):
    """Raised when the target cannot be reached from the source."""


def dijkstra(graph, source, target=None):
    """Return (dist, prev). Stops early when `target` is settled."""
    nodes = set(graph)
    for u in graph:
        for v, w in graph[u]:
            if w < 0:
                raise ValueError(f"Negative weight on edge {u}->{v}: Dijkstra needs weights >= 0")
            nodes.add(v)
    if source not in nodes:
        raise KeyError(f"Source node {source!r} is not in the graph")

    dist = {n: math.inf for n in nodes}
    prev = {}
    dist[source] = 0.0
    heap = [(0.0, source)]                                   # (distance, node)
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:                                     # stale heap entry
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist[v]:                                 # shorter route found
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def shortest_path(graph, source, target):
    """Return (total_distance, [source, ..., target]) or raise NoPathError."""
    dist, prev = dijkstra(graph, source, target)
    if target not in dist:
        raise KeyError(f"Target node {target!r} is not in the graph")
    if math.isinf(dist[target]):
        raise NoPathError(f"No road path exists from {source!r} to {target!r} in this graph.")
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return dist[target], path


def graph_from_networkx(G, weight="length"):
    """Convert a NetworkX/OSMnx (Multi)DiGraph into our adjacency list.
    Parallel edges: keep the shortest. Edge direction (one-way streets) is respected."""
    adj = {n: {} for n in G.nodes}
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight, 0.0))
        if v not in adj[u] or w < adj[u][v]:
            adj[u][v] = w
    return {u: list(nbrs.items()) for u, nbrs in adj.items()}
