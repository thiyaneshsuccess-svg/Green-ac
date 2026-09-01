"""
tools.py — Tool/function interfaces between the AI decision engine and
the simulation.

These six functions are the ONLY way anything — AI or otherwise — is
allowed to read live simulation state or request a change to it:

    get_traffic()
    get_ambulance_position()
    get_available_routes()
    change_signal()
    reroute_ambulance()
    notify_hospital()

The AI decision engine (ai_engine.py) never imports RoadNetwork,
TrafficSimulator, SignalController, Ambulance, or HospitalRegistry
directly. It only ever sees the plain-JSON output of the `get_*` tools,
and can only act by calling `change_signal` / `reroute_ambulance` /
`notify_hospital` — each of which validates its input and rejects
anything that doesn't check out *before* touching simulation state.
"""

from typing import Any, Dict, List, Optional

from network import RoadNetwork
from traffic import TrafficSimulator, SignalController
from route import RoutePlanner, Route, generate_route_candidates
from ambulance import Ambulance
from eta import ETACalculator
from hospital import HospitalRegistry
from green_corridor import GreenCorridorController, SignalIntervention


ALLOWED_SIGNAL_ACTIONS = {"force_green", "release"}


class ToolError(Exception):
    """Raised when a tool call is invalid — bad id, disconnected route, etc."""


class SimulationTools:
    """
    Facade exposing exactly six methods. Everything else about the
    simulation (network topology, traffic internals, ambulance mover)
    stays behind this interface.
    """

    def __init__(self, network: RoadNetwork, traffic_sim: TrafficSimulator,
                 signals: SignalController, planner: RoutePlanner,
                 eta_calculator: ETACalculator, mover, corridor: GreenCorridorController,
                 hospitals: HospitalRegistry) -> None:
        self.network = network
        self.traffic_sim = traffic_sim
        self.signals = signals
        self.planner = planner
        self.eta_calculator = eta_calculator
        self.mover = mover
        self.corridor = corridor
        self.hospitals = hospitals

    # ------------------------------------------------------------------
    # Read-only tools
    # ------------------------------------------------------------------

    def get_traffic(self, at_time_s: float) -> Dict[str, Any]:
        """Live traffic density/congestion per segment, queue lengths, and signal states."""
        segments = {
            seg_id: {
                "density": round(self.traffic_sim.get_density(seg_id), 3),
                "congestion_factor": round(self.traffic_sim.get_congestion_factor(seg_id), 3),
            }
            for seg_id in self.network.all_segment_ids()
        }
        queues = {
            node_id: self.traffic_sim.get_queue_length(node_id)
            for node_id in self.network.all_intersection_ids()
        }
        signals = {
            node_id: {
                "green": self.signals.is_green(node_id, at_time_s),
                "time_until_green": round(self.signals.time_until_green(node_id, at_time_s), 2),
            }
            for node_id in self.network.all_intersection_ids()
        }
        return {"segments": segments, "queues": queues, "signals": signals, "at_time_s": at_time_s}

    def get_ambulance_position(self, ambulance: Ambulance) -> Dict[str, Any]:
        """Where the ambulance currently is and how far along its route it's gotten."""
        return {
            "current_node": ambulance.current_node(),
            "next_node": ambulance.next_node(),
            "status": ambulance.status.value,
            "elapsed_time_s": ambulance.elapsed_time_s,
            "distance_traveled_km": round(self.mover.distance_traveled_km(ambulance), 3),
            "progress_fraction": round(self.mover.progress_fraction(ambulance), 4),
            "remaining_route": ambulance.route.node_ids[ambulance.segment_index:],
        }

    def get_available_routes(self, ambulance: Ambulance, at_time_s: float,
                              k: int = 2) -> List[Dict[str, Any]]:
        """
        Up to k candidate routes from the ambulance's CURRENT node to its
        destination, each with distance and a live, congestion/signal-aware
        ETA. The primary route is the current congestion-weighted shortest
        path; a second candidate (if one exists) deliberately avoids the
        primary route's segments so a genuine alternative is offered.
        """
        current_node = ambulance.current_node()
        destination = ambulance.route.destination

        def congestion_weight(segment):
            return segment.length_km * self.traffic_sim.get_congestion_factor(segment.id)

        candidates = generate_route_candidates(self.planner, current_node, destination,
                                                 congestion_weight, k=k)

        results = []
        for route in candidates[:k]:
            eta = self.eta_calculator.calculate(route, start_time_s=at_time_s, obey_signals=True)
            results.append({
                "nodes": route.node_ids,
                "segment_ids": route.segment_ids,
                "distance_km": route.total_distance_km,
                "eta_seconds": round(eta.total_time_s, 2),
            })
        return results

    # ------------------------------------------------------------------
    # Action tools — the only places simulation state is mutated
    # ------------------------------------------------------------------

    def change_signal(self, junction_id: str, action: str, until_time_s: Optional[float] = None,
                       requested_at_s: float = 0.0, reason: str = "") -> Dict[str, Any]:
        """Force a junction's signal green, or release a prior override, with validation."""
        if junction_id not in self.network.intersections:
            raise ToolError(f"Unknown junction '{junction_id}'")
        if action not in ALLOWED_SIGNAL_ACTIONS:
            raise ToolError(f"Unsupported signal action '{action}' (allowed: {sorted(ALLOWED_SIGNAL_ACTIONS)})")

        if action == "force_green":
            if until_time_s is None or until_time_s <= requested_at_s:
                raise ToolError("force_green requires an until_time_s in the future")
            self.signals.force_green(junction_id, until_time_s)
            self.corridor.interventions.append(SignalIntervention(
                junction_id=junction_id,
                action="force_green",
                requested_sim_time_s=requested_at_s,
                predicted_arrival_s=0.0,   # not applicable for a directly-requested change
                effective_until_s=until_time_s,
                queue_clearing_s=0.0,
                reason=reason or f"AI-directed force_green at junction '{junction_id}'",
            ))
        else:  # release
            self.signals.clear_override(junction_id)

        return {"junction_id": junction_id, "action": action, "until_time_s": until_time_s}

    def reroute_ambulance(self, ambulance: Ambulance, new_node_path: List[str]) -> Dict[str, Any]:
        """Replace the ambulance's route with a new, validated path from its current node."""
        if not new_node_path or len(new_node_path) < 2:
            raise ToolError("new_node_path must contain at least two nodes")
        if new_node_path[0] != ambulance.current_node():
            raise ToolError(
                f"new_node_path must start at the ambulance's current node "
                f"('{ambulance.current_node()}'), got '{new_node_path[0]}'"
            )
        if new_node_path[-1] != ambulance.route.destination:
            raise ToolError(
                f"new_node_path must end at the ambulance's destination "
                f"('{ambulance.route.destination}'), got '{new_node_path[-1]}'"
            )

        segment_ids: List[str] = []
        total_distance = 0.0
        for u, v in zip(new_node_path, new_node_path[1:]):
            segment = self.network.find_segment(u, v)
            if segment is None:
                raise ToolError(f"No road segment connects '{u}' -> '{v}'")
            segment_ids.append(segment.id)
            total_distance += segment.length_km

        ambulance.route = Route(node_ids=list(new_node_path), segment_ids=segment_ids,
                                 total_distance_km=total_distance)
        ambulance.segment_index = 0
        ambulance.distance_into_segment_km = 0.0

        return {"new_route": ambulance.route.node_ids, "distance_km": total_distance}

    def notify_hospital(self, ambulance: Ambulance, eta_seconds: float, at_time_s: float) -> Dict[str, Any]:
        """Notify the destination hospital of the ambulance's updated ETA."""
        hospital = self.hospitals.find_by_node(ambulance.route.destination)
        if hospital is None:
            raise ToolError(f"No hospital registered at destination node '{ambulance.route.destination}'")
        try:
            record = self.hospitals.notify(hospital.id, ambulance.id, eta_seconds, at_time_s)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "hospital_id": hospital.id,
            "hospital_name": hospital.name,
            "eta_seconds": record.eta_seconds,
            "sim_time_s": record.sim_time_s,
        }
