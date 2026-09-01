"""
ambulance.py — Component C: Ambulance Movement

Simulates an ambulance physically progressing along a Route, tick by
tick. In this baseline component the ambulance behaves like any other
vehicle: it slows for congestion and stops for red lights / queues.
Emergency green-corridor behavior (requirement #7) will later override
`obey_signals` and congestion handling without changing this class's
public interface.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from network import RoadNetwork
from route import Route
from traffic import TrafficSimulator, SignalController


class AmbulanceStatus(Enum):
    IDLE = "idle"
    EN_ROUTE = "en_route"
    WAITING_AT_SIGNAL = "waiting_at_signal"
    ARRIVED = "arrived"


@dataclass
class Ambulance:
    """Live state of a single ambulance."""
    id: str
    route: Route
    max_speed_kmh: float = 100.0
    status: AmbulanceStatus = AmbulanceStatus.IDLE

    # Progress state
    segment_index: int = 0          # index into route.segment_ids currently being traversed
    distance_into_segment_km: float = 0.0
    elapsed_time_s: float = 0.0

    def current_segment_id(self) -> Optional[str]:
        if self.segment_index >= len(self.route.segment_ids):
            return None
        return self.route.segment_ids[self.segment_index]

    def current_node(self) -> str:
        return self.route.node_ids[self.segment_index]

    def next_node(self) -> Optional[str]:
        if self.segment_index + 1 < len(self.route.node_ids):
            return self.route.node_ids[self.segment_index + 1]
        return None

    def upcoming_nodes(self, count: int) -> List[str]:
        """
        The next `count` intersections ahead of the ambulance's current
        position (used later for predictive pre-clearing, requirement #8).
        """
        start = self.segment_index + 1
        return self.route.node_ids[start:start + count]

    # Note: distance-traveled and progress-fraction require segment lengths
    # from the RoadNetwork, so those live on AmbulanceMover (below) to keep
    # this dataclass free of a network dependency.


class AmbulanceMover:
    """
    Advances an Ambulance's position over time, given the road network
    and current traffic state. This is the baseline (non-emergency) mover:
    it respects traffic signals and slows for congestion.
    """

    def __init__(self, network: RoadNetwork, traffic_sim: TrafficSimulator,
                 signal_controller: SignalController) -> None:
        self.network = network
        self.traffic_sim = traffic_sim
        self.signals = signal_controller

    def dispatch(self, ambulance: Ambulance) -> None:
        """Mark an idle ambulance as en route at the start of its route."""
        if ambulance.route.is_trivial():
            ambulance.status = AmbulanceStatus.ARRIVED
            return
        ambulance.status = AmbulanceStatus.EN_ROUTE
        ambulance.segment_index = 0
        ambulance.distance_into_segment_km = 0.0
        ambulance.elapsed_time_s = 0.0

    def step(self, ambulance: Ambulance, dt_s: float, obey_signals: bool = True) -> None:
        """
        Advance the ambulance by dt_s seconds of simulated time.
        Handles: congestion-adjusted speed, arriving at an intersection,
        waiting for a red light / queue, and final arrival.
        """
        if ambulance.status in (AmbulanceStatus.ARRIVED, AmbulanceStatus.IDLE):
            return

        remaining_dt = dt_s
        ambulance.elapsed_time_s += dt_s

        while remaining_dt > 0 and ambulance.status != AmbulanceStatus.ARRIVED:
            seg_id = ambulance.current_segment_id()
            if seg_id is None:
                ambulance.status = AmbulanceStatus.ARRIVED
                return

            segment = self.network.get_segment(seg_id)
            congestion_factor = self.traffic_sim.get_congestion_factor(seg_id)
            effective_speed_kmh = min(ambulance.max_speed_kmh,
                                       segment.speed_limit_kmh) / congestion_factor
            effective_speed_kms = effective_speed_kmh / 3600.0  # km per second

            distance_remaining_on_segment = segment.length_km - ambulance.distance_into_segment_km
            max_travel_this_tick = effective_speed_kms * remaining_dt

            if max_travel_this_tick < distance_remaining_on_segment:
                # Still mid-segment after this tick
                ambulance.distance_into_segment_km += max_travel_this_tick
                ambulance.status = AmbulanceStatus.EN_ROUTE
                remaining_dt = 0.0
            else:
                # Reaches the end of the segment this tick
                time_to_reach = distance_remaining_on_segment / effective_speed_kms \
                    if effective_speed_kms > 0 else remaining_dt
                remaining_dt -= time_to_reach

                arrival_node = segment.to_id
                if obey_signals and not self.signals.is_green(arrival_node, ambulance.elapsed_time_s):
                    # Must wait for the light; hold position at the intersection
                    ambulance.status = AmbulanceStatus.WAITING_AT_SIGNAL
                    ambulance.distance_into_segment_km = segment.length_km
                    remaining_dt = 0.0
                else:
                    queue_delay = 0.0
                    if obey_signals:
                        queue_delay = self.traffic_sim.get_queue_length(arrival_node) * 2.0
                    if queue_delay > remaining_dt:
                        ambulance.status = AmbulanceStatus.WAITING_AT_SIGNAL
                        ambulance.distance_into_segment_km = segment.length_km
                        remaining_dt = 0.0
                    else:
                        remaining_dt -= queue_delay
                        ambulance.segment_index += 1
                        ambulance.distance_into_segment_km = 0.0
                        if ambulance.segment_index >= len(ambulance.route.segment_ids):
                            ambulance.status = AmbulanceStatus.ARRIVED
                        else:
                            ambulance.status = AmbulanceStatus.EN_ROUTE

    def distance_traveled_km(self, ambulance: Ambulance) -> float:
        """Total distance covered so far along the route."""
        traveled = 0.0
        for i in range(ambulance.segment_index):
            seg = self.network.get_segment(ambulance.route.segment_ids[i])
            traveled += seg.length_km
        return traveled + ambulance.distance_into_segment_km

    def progress_fraction(self, ambulance: Ambulance) -> float:
        if ambulance.route.total_distance_km <= 0:
            return 1.0
        return min(1.0, self.distance_traveled_km(ambulance) / ambulance.route.total_distance_km)
