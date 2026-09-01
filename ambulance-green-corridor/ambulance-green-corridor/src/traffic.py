"""
traffic.py — Component B: Traffic Simulation

Covers core requirements #2 and #3:
  - traffic density / vehicle queues at each intersection
  - traffic signal behavior

Design notes:
  - SignalController is purely time-based (deterministic): given an
    elapsed simulation time, it always returns the same phase. No
    randomness involved, so it behaves identically in demo mode or live.
  - TrafficSimulator owns per-segment congestion density and per-
    intersection vehicle queues. It uses a seeded RNG so that when a
    seed is provided (demo mode), evolution is fully reproducible.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, Optional

from network import RoadNetwork, RoadSegment


# ---------------------------------------------------------------------------
# Traffic signals
# ---------------------------------------------------------------------------

@dataclass
class SignalTiming:
    """Fixed-time signal plan for one intersection."""
    intersection_id: str
    cycle_length_s: float = 60.0     # total red+green cycle duration
    green_duration_s: float = 30.0   # portion of the cycle that is green
    offset_s: float = 0.0            # phase offset so intersections aren't all in sync


class SignalController:
    """
    Deterministic traffic-signal simulator. Every intersection with a
    signal cycles between green and red on a fixed schedule.

    Also supports temporary "forced green" overrides, which is the
    mechanism the AI decision engine uses to pre-clear or hold open a
    junction for the ambulance (requirement #7/#8). Overrides are the
    *only* way signal behavior deviates from the fixed schedule, and
    they can only be set through `force_green` / `clear_override` —
    i.e. through the tool interface, never by reaching into internal
    state directly.
    """

    def __init__(self, network: RoadNetwork, default_cycle_s: float = 60.0,
                 default_green_s: float = 30.0, seed: Optional[int] = None) -> None:
        self.network = network
        self.timings: Dict[str, SignalTiming] = {}
        self._forced_green_until: Dict[str, float] = {}

        # Offsets are deterministic per-seed so demo mode is reproducible,
        # but different intersections still desync from one another.
        rng = random.Random(seed if seed is not None else 0)
        for node_id, node in network.intersections.items():
            if node.has_signal:
                offset = rng.uniform(0, default_cycle_s)
                self.timings[node_id] = SignalTiming(
                    intersection_id=node_id,
                    cycle_length_s=default_cycle_s,
                    green_duration_s=default_green_s,
                    offset_s=offset,
                )

    def force_green(self, intersection_id: str, until_time_s: float) -> None:
        """Override the signal to green until the given simulation time."""
        if intersection_id not in self.network.intersections:
            raise ValueError(f"Unknown intersection '{intersection_id}'")
        self._forced_green_until[intersection_id] = until_time_s

    def clear_override(self, intersection_id: str) -> None:
        """Remove any forced-green override, reverting to the normal schedule."""
        self._forced_green_until.pop(intersection_id, None)

    def has_override(self, intersection_id: str, at_time_s: float) -> bool:
        return self._forced_green_until.get(intersection_id, -1.0) >= at_time_s

    def is_green(self, intersection_id: str, at_time_s: float) -> bool:
        """Whether the signal is green at the given simulation time (seconds)."""
        if self.has_override(intersection_id, at_time_s):
            return True
        timing = self.timings.get(intersection_id)
        if timing is None:
            return True  # no signal at this intersection => always passable
        phase = (at_time_s + timing.offset_s) % timing.cycle_length_s
        return phase < timing.green_duration_s

    def time_until_green(self, intersection_id: str, at_time_s: float) -> float:
        """Seconds to wait (from at_time_s) until the signal turns green. 0 if already green."""
        if self.has_override(intersection_id, at_time_s):
            return 0.0
        timing = self.timings.get(intersection_id)
        if timing is None:
            return 0.0
        phase = (at_time_s + timing.offset_s) % timing.cycle_length_s
        if phase < timing.green_duration_s:
            return 0.0
        return timing.cycle_length_s - phase


# ---------------------------------------------------------------------------
# Traffic density & queues
# ---------------------------------------------------------------------------

@dataclass
class SegmentTraffic:
    """Live traffic state for one road segment."""
    segment_id: str
    density: float = 0.2  # fraction of capacity in use, 0.0 (empty) - 1.0+ (jammed)

    def congestion_factor(self) -> float:
        """
        Multiplier applied to free-flow travel time.
        1.0 = no slowdown. Grows steeply past density 0.8 to model jams.
        """
        d = max(0.0, self.density)
        if d <= 0.8:
            return 1.0 + d  # up to 1.8x at density 0.8
        # Beyond 0.8 congestion gets much worse per unit density (near-gridlock)
        return 1.8 + (d - 0.8) * 6.0


AVG_DISCHARGE_S_PER_VEHICLE = 2.0  # avg seconds for one queued vehicle to clear an intersection


@dataclass
class IntersectionQueue:
    """Vehicles currently queued at an intersection waiting on a red signal."""
    intersection_id: str
    queue_length: int = 0

    def clearance_time_s(self) -> float:
        """Extra seconds needed to clear the queue once the light turns green."""
        return self.queue_length * AVG_DISCHARGE_S_PER_VEHICLE


class TrafficSimulator:
    """
    Owns per-segment density and per-intersection queue state, and evolves
    it over time. Seeded RNG => deterministic in demo mode.
    """

    def __init__(self, network: RoadNetwork, seed: Optional[int] = None,
                 volatility: float = 0.03) -> None:
        self.network = network
        self.rng = random.Random(seed)
        self.volatility = volatility
        self.elapsed_s: float = 0.0

        self.segment_traffic: Dict[str, SegmentTraffic] = {
            seg_id: SegmentTraffic(segment_id=seg_id, density=self.rng.uniform(0.1, 0.4))
            for seg_id in network.all_segment_ids()
        }
        self.queues: Dict[str, IntersectionQueue] = {
            node_id: IntersectionQueue(intersection_id=node_id, queue_length=0)
            for node_id in network.all_intersection_ids()
        }

    def get_density(self, segment_id: str) -> float:
        return self.segment_traffic[segment_id].density

    def get_congestion_factor(self, segment_id: str) -> float:
        return self.segment_traffic[segment_id].congestion_factor()

    def get_queue_length(self, intersection_id: str) -> int:
        return self.queues[intersection_id].queue_length

    def set_density(self, segment_id: str, density: float) -> None:
        """Directly set density (e.g. to script deterministic test/demo scenarios)."""
        self.segment_traffic[segment_id].density = max(0.0, density)

    def step(self, dt_s: float) -> None:
        """
        Advance traffic state by dt_s seconds:
          - each segment's density does a small bounded random walk
          - queues grow slightly with density and shrink over time
        Deterministic given the same seed and sequence of calls.
        """
        self.elapsed_s += dt_s
        minutes = dt_s / 60.0

        for traffic in self.segment_traffic.values():
            delta = self.rng.uniform(-self.volatility, self.volatility) * minutes * 10
            traffic.density = max(0.0, min(1.5, traffic.density + delta))

        for queue in self.queues.values():
            seg_traffic_here = self._avg_incoming_density(queue.intersection_id)
            growth = seg_traffic_here * dt_s * 0.05
            drain = (dt_s / AVG_DISCHARGE_S_PER_VEHICLE) * 0.3
            queue.queue_length = max(0, round(queue.queue_length + growth - drain))

    def _avg_incoming_density(self, intersection_id: str) -> float:
        incoming = [s for s in self.segment_traffic.values()
                    if self.network.segments[s.segment_id].to_id == intersection_id]
        if not incoming:
            return 0.0
        return sum(s.density for s in incoming) / len(incoming)
