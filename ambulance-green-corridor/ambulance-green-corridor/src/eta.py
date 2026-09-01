"""
eta.py — Component E: ETA Calculation

Computes travel time estimates for a Route given the current traffic
state (congestion + signals + queues). Produces both a total ETA and a
per-segment breakdown so results are explainable later by the AI layer
(requirement #11) without needing to touch this module.
"""

from dataclasses import dataclass
from typing import List, Optional

from network import RoadNetwork
from route import Route
from traffic import TrafficSimulator, SignalController


@dataclass
class SegmentETA:
    """ETA contribution of a single segment in a route."""
    segment_id: str
    from_id: str
    to_id: str
    free_flow_time_s: float
    congestion_factor: float
    signal_wait_s: float
    queue_clearance_s: float

    @property
    def travel_time_s(self) -> float:
        return self.free_flow_time_s * self.congestion_factor + self.signal_wait_s + self.queue_clearance_s


@dataclass
class ETAResult:
    """Full ETA breakdown for a route, evaluated at a specific start time."""
    route: Route
    start_time_s: float
    segment_etas: List[SegmentETA]

    @property
    def total_time_s(self) -> float:
        return sum(seg.travel_time_s for seg in self.segment_etas)

    @property
    def total_time_min(self) -> float:
        return self.total_time_s / 60.0

    @property
    def arrival_time_s(self) -> float:
        return self.start_time_s + self.total_time_s


class ETACalculator:
    """
    Calculates ETA for a Route using live traffic + signal state.

    By default this represents the BASELINE scenario (requirement #6):
    traffic signals behave normally and the ambulance may have to wait
    at red lights and behind queues, same as any other vehicle. Emergency
    green-corridor behavior overrides this later without changing this
    class's public interface.
    """

    def __init__(self, network: RoadNetwork, traffic_sim: TrafficSimulator,
                 signal_controller: SignalController) -> None:
        self.network = network
        self.traffic_sim = traffic_sim
        self.signals = signal_controller

    def calculate(self, route: Route, start_time_s: float = 0.0,
                   obey_signals: bool = True) -> ETAResult:
        """
        Walk the route segment by segment, accumulating travel time.
        `obey_signals=False` simulates a fully cleared green corridor
        (used later by the emergency-mode component).
        """
        segment_etas: List[SegmentETA] = []
        clock = start_time_s

        for seg_id in route.segment_ids:
            segment = self.network.get_segment(seg_id)
            free_flow_s = segment.free_flow_time_hr() * 3600.0
            congestion_factor = self.traffic_sim.get_congestion_factor(seg_id)

            signal_wait_s = 0.0
            queue_clearance_s = 0.0
            arrival_at_next = clock + free_flow_s * congestion_factor

            if obey_signals:
                signal_wait_s = self.signals.time_until_green(segment.to_id, arrival_at_next)
                queue_clearance_s = self.traffic_sim.get_queue_length(segment.to_id) * 2.0

            seta = SegmentETA(
                segment_id=seg_id,
                from_id=segment.from_id,
                to_id=segment.to_id,
                free_flow_time_s=free_flow_s,
                congestion_factor=congestion_factor,
                signal_wait_s=signal_wait_s,
                queue_clearance_s=queue_clearance_s,
            )
            segment_etas.append(seta)
            clock += seta.travel_time_s

        return ETAResult(route=route, start_time_s=start_time_s, segment_etas=segment_etas)
