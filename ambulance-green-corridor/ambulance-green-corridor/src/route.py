"""
route.py — Component D: Route Representation

A Route is just an ordered sequence of intersections + the segments that
connect them, plus a total distance. RoutePlanner computes a baseline
route using Dijkstra's algorithm, weighted by a caller-supplied cost
function (defaults to plain segment length so "baseline" really means
"shortest path with normal traffic signal behavior", per requirement #6 —
congestion-aware weighting is layered on via the weight_fn hook and used
later by the ETA/AI components without changing this module).
"""

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from network import RoadNetwork, RoadSegment


@dataclass
class Route:
    """An ordered path through the network from origin to destination."""
    node_ids: List[str]
    segment_ids: List[str]
    total_distance_km: float

    @property
    def origin(self) -> str:
        return self.node_ids[0]

    @property
    def destination(self) -> str:
        return self.node_ids[-1]

    def is_trivial(self) -> bool:
        return len(self.node_ids) <= 1

    def __len__(self) -> int:
        return len(self.segment_ids)


# A weight function takes a RoadSegment and returns a non-negative cost (e.g. time or distance)
WeightFn = Callable[[RoadSegment], float]


def default_distance_weight(segment: RoadSegment) -> float:
    """Baseline weight: plain physical distance (km), ignoring traffic."""
    return segment.length_km


class RoutePlanner:
    """Computes routes through a RoadNetwork using Dijkstra's algorithm."""

    def __init__(self, network: RoadNetwork) -> None:
        self.network = network

    def shortest_path(self, start_id: str, end_id: str,
                       weight_fn: Optional[WeightFn] = None) -> Route:
        """
        Compute the lowest-cost route from start_id to end_id.
        Raises ValueError if no path exists.
        """
        if start_id not in self.network.intersections:
            raise ValueError(f"Unknown start intersection '{start_id}'")
        if end_id not in self.network.intersections:
            raise ValueError(f"Unknown end intersection '{end_id}'")

        weight_fn = weight_fn or default_distance_weight

        if start_id == end_id:
            return Route(node_ids=[start_id], segment_ids=[], total_distance_km=0.0)

        # Dijkstra
        dist: Dict[str, float] = {start_id: 0.0}
        prev_node: Dict[str, str] = {}
        prev_segment: Dict[str, str] = {}
        visited = set()
        heap = [(0.0, start_id)]

        while heap:
            d, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            if node == end_id:
                break

            for neighbor_id, segment in self.network.neighbors(node):
                if neighbor_id in visited:
                    continue
                cost = weight_fn(segment)
                if cost < 0:
                    raise ValueError(f"Negative weight on segment '{segment.id}'")
                new_dist = d + cost
                if new_dist < dist.get(neighbor_id, float("inf")):
                    dist[neighbor_id] = new_dist
                    prev_node[neighbor_id] = node
                    prev_segment[neighbor_id] = segment.id
                    heapq.heappush(heap, (new_dist, neighbor_id))

        if end_id not in dist:
            raise ValueError(f"No path found from '{start_id}' to '{end_id}'")

        # Reconstruct path
        node_path = [end_id]
        segment_path = []
        cur = end_id
        while cur != start_id:
            segment_path.append(prev_segment[cur])
            cur = prev_node[cur]
            node_path.append(cur)
        node_path.reverse()
        segment_path.reverse()

        total_distance = sum(self.network.get_segment(sid).length_km for sid in segment_path)

        return Route(node_ids=node_path, segment_ids=segment_path, total_distance_km=total_distance)


def generate_route_candidates(planner: "RoutePlanner", start_id: str, end_id: str,
                               weight_fn: WeightFn, k: int = 2) -> List[Route]:
    """
    Up to k candidate routes under the same weight function: the shortest
    path, plus (if k > 1 and a genuine alternative exists) a detour that
    avoids the primary route's segments entirely. Shared by any caller
    that needs "at least two possible routes" (tools.py, route_selection.py)
    so the candidate-generation algorithm has one definition.
    """
    primary = planner.shortest_path(start_id, end_id, weight_fn=weight_fn)
    candidates = [primary]

    if k > 1:
        penalized = set(primary.segment_ids)

        def detour_weight(segment: RoadSegment) -> float:
            base = weight_fn(segment)
            return base * 1000.0 if segment.id in penalized else base

        try:
            alternate = planner.shortest_path(start_id, end_id, weight_fn=detour_weight)
            if alternate.node_ids != primary.node_ids:
                candidates.append(alternate)
        except ValueError:
            pass  # no genuine alternative exists; primary-only is fine

    return candidates[:k]
