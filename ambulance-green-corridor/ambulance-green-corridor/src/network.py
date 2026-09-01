"""
network.py — Component A: Road / Intersection Data Model

Defines the static structure of the road network: intersections (nodes)
and road segments (edges) connecting them. No traffic or time-dependent
behavior lives here — this is pure topology + geometry.

Deliberately dependency-free (no networkx / no external graph library)
so the whole simulation stays lightweight, transparent, and easy to test.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class Intersection:
    """A single intersection / node in the road network."""
    id: str
    x: float  # arbitrary planar coordinate (km), used for distance + map display
    y: float
    has_signal: bool = True  # whether this intersection has a traffic light

    def distance_to(self, other: "Intersection") -> float:
        """Euclidean distance in km to another intersection."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class RoadSegment:
    """
    A directed road segment between two intersections.
    Bidirectional roads are represented as two RoadSegment objects
    (one per direction) so each direction can carry independent traffic.
    """
    id: str
    from_id: str
    to_id: str
    length_km: float
    lanes: int = 1
    speed_limit_kmh: float = 50.0
    capacity_vehicles_per_lane: int = 40  # rough capacity used for density calc

    @property
    def capacity(self) -> int:
        return max(1, self.lanes * self.capacity_vehicles_per_lane)

    def free_flow_time_hr(self) -> float:
        """Travel time in hours with zero congestion, at the speed limit."""
        return self.length_km / self.speed_limit_kmh


class RoadNetwork:
    """
    Holds all intersections and road segments, and exposes adjacency
    lookups used by routing, traffic simulation, and ETA calculation.
    """

    def __init__(self) -> None:
        self.intersections: Dict[str, Intersection] = {}
        self.segments: Dict[str, RoadSegment] = {}
        # adjacency: intersection_id -> list of segment ids leaving it
        self._outgoing: Dict[str, List[str]] = {}

    # ---- construction ----

    def add_intersection(self, intersection: Intersection) -> None:
        if intersection.id in self.intersections:
            raise ValueError(f"Intersection '{intersection.id}' already exists")
        self.intersections[intersection.id] = intersection
        self._outgoing.setdefault(intersection.id, [])

    def add_segment(self, segment: RoadSegment, bidirectional: bool = False) -> None:
        """
        Add a directed segment. If bidirectional=True, also creates the
        reverse segment automatically (with id suffixed '_r').
        """
        self._validate_segment_endpoints(segment)
        if segment.id in self.segments:
            raise ValueError(f"Segment '{segment.id}' already exists")

        self.segments[segment.id] = segment
        self._outgoing.setdefault(segment.from_id, []).append(segment.id)

        if bidirectional:
            reverse = RoadSegment(
                id=f"{segment.id}_r",
                from_id=segment.to_id,
                to_id=segment.from_id,
                length_km=segment.length_km,
                lanes=segment.lanes,
                speed_limit_kmh=segment.speed_limit_kmh,
                capacity_vehicles_per_lane=segment.capacity_vehicles_per_lane,
            )
            self.segments[reverse.id] = reverse
            self._outgoing.setdefault(reverse.from_id, []).append(reverse.id)

    def _validate_segment_endpoints(self, segment: RoadSegment) -> None:
        if segment.from_id not in self.intersections:
            raise ValueError(f"Unknown intersection '{segment.from_id}'")
        if segment.to_id not in self.intersections:
            raise ValueError(f"Unknown intersection '{segment.to_id}'")

    # ---- lookups ----

    def get_intersection(self, intersection_id: str) -> Intersection:
        return self.intersections[intersection_id]

    def get_segment(self, segment_id: str) -> RoadSegment:
        return self.segments[segment_id]

    def find_segment(self, from_id: str, to_id: str) -> Optional[RoadSegment]:
        """Find the segment (if any) directly connecting from_id -> to_id."""
        for seg_id in self._outgoing.get(from_id, []):
            seg = self.segments[seg_id]
            if seg.to_id == to_id:
                return seg
        return None

    def neighbors(self, intersection_id: str) -> List[Tuple[str, RoadSegment]]:
        """List of (neighbor_intersection_id, segment) reachable directly from this node."""
        result = []
        for seg_id in self._outgoing.get(intersection_id, []):
            seg = self.segments[seg_id]
            result.append((seg.to_id, seg))
        return result

    def outgoing_segments(self, intersection_id: str) -> List[RoadSegment]:
        return [self.segments[sid] for sid in self._outgoing.get(intersection_id, [])]

    def all_intersection_ids(self) -> List[str]:
        return list(self.intersections.keys())

    def all_segment_ids(self) -> List[str]:
        return list(self.segments.keys())

    def validate(self) -> None:
        """Raise if the network references any dangling intersection ids."""
        for seg in self.segments.values():
            self._validate_segment_endpoints(seg)
