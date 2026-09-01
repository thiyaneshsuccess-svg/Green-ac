"""
green_corridor.py — Traffic-Control Layer

Sits on top of the existing signal simulation (traffic.py) and adds the
emergency-mode behavior an ambulance needs:

  1. Normal signal behavior        -> delegated to SignalController (unchanged)
  2. Emergency mode                -> GreenCorridorController.emergency_mode flag
  3. Green corridor creation       -> pre_clear_upcoming() forces junctions green
  4. Approach -> should-change?    -> should_change_signal()
  5. Predictive pre-clear (next 2) -> upcoming_junctions() + pre_clear_upcoming()
  6. Queue-clearing estimate       -> queue_clearing_time()
  7. Intervention logging          -> SignalIntervention records in .interventions
  8. Baseline vs emergency ETA     -> compare_eta()

This is purely rule-based / deterministic — no external AI API is
involved. It's designed to be exactly what a future AI decision engine
would call *through* (it never lets a caller reach into SignalController
or TrafficSimulator directly to force a light; every change goes through
`pre_clear_upcoming`, which validates, applies, and logs).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from network import RoadNetwork
from route import Route
from ambulance import Ambulance
from traffic import TrafficSimulator, SignalController
from eta import ETACalculator, ETAResult


@dataclass
class SignalIntervention:
    """A single record of the controller forcing a signal green for the ambulance."""
    junction_id: str
    action: str                 # "force_green" (only action type for now)
    requested_sim_time_s: float  # when the decision was made
    predicted_arrival_s: float   # when the ambulance is expected to reach the junction
    effective_until_s: float     # signal held green until this simulation time
    queue_clearing_s: float
    reason: str


@dataclass
class ETAComparison:
    """Baseline (normal signals) vs. emergency (idealized green corridor) ETA."""
    baseline_seconds: float
    emergency_seconds: float
    time_saved_seconds: float
    time_saved_percent: float


class GreenCorridorController:
    """
    Rule-based traffic-control layer for ambulance green corridors.

    Normal mode: signals behave exactly as SignalController schedules them
    (requirement #1) — this controller doesn't touch anything.

    Emergency mode: as the ambulance moves, the next two junctions ahead
    are continuously evaluated; any that would be red when the ambulance
    is predicted to arrive get forced green (with enough hold time to also
    clear any queued vehicles), and every such intervention is logged.
    """

    def __init__(self, network: RoadNetwork, traffic_sim: TrafficSimulator,
                 signals: SignalController, pre_clear_lookahead: int = 2,
                 green_hold_buffer_s: float = 8.0) -> None:
        self.network = network
        self.traffic_sim = traffic_sim
        self.signals = signals
        self.pre_clear_lookahead = pre_clear_lookahead
        self.green_hold_buffer_s = green_hold_buffer_s

        self.emergency_mode: bool = False
        self.interventions: List[SignalIntervention] = []

    # ---- mode control (requirement #2) ----

    def enable_emergency_mode(self) -> None:
        self.emergency_mode = True

    def disable_emergency_mode(self) -> None:
        """Turn emergency mode off and release any active overrides immediately."""
        self.emergency_mode = False
        for junction_id in list(self.network.all_intersection_ids()):
            self.signals.clear_override(junction_id)

    # ---- junction identification (requirement #4, #5) ----

    def next_junction(self, ambulance: Ambulance) -> Optional[str]:
        """The single intersection the ambulance is currently heading toward."""
        return ambulance.next_node()

    def upcoming_junctions(self, ambulance: Ambulance, count: Optional[int] = None) -> List[str]:
        """The next `count` intersections ahead (defaults to the configured lookahead of 2)."""
        n = count if count is not None else self.pre_clear_lookahead
        return ambulance.upcoming_nodes(n)

    # ---- prediction ----

    def predicted_arrival_time(self, ambulance: Ambulance, target_node: str) -> float:
        """
        Predict the simulation time (seconds) at which the ambulance would
        reach `target_node`, assuming it is NOT stopped by red lights or
        queues along the way (i.e. only distance + congestion matter).
        This is what lets us ask "would this signal be red when I arrive?"
        without the answer being circular.
        """
        route = ambulance.route
        if target_node not in route.node_ids:
            raise ValueError(f"'{target_node}' is not on the ambulance's route")

        target_index = route.node_ids.index(target_node)
        if target_index <= ambulance.segment_index:
            raise ValueError(f"'{target_node}' is not ahead of the ambulance's current position")

        t = ambulance.elapsed_time_s
        for i in range(ambulance.segment_index, target_index):
            seg_id = route.segment_ids[i]
            segment = self.network.get_segment(seg_id)
            congestion_factor = self.traffic_sim.get_congestion_factor(seg_id)
            effective_speed_kms = (segment.speed_limit_kmh / congestion_factor) / 3600.0

            length_km = segment.length_km
            if i == ambulance.segment_index:
                length_km = max(0.0, length_km - ambulance.distance_into_segment_km)

            t += length_km / effective_speed_kms if effective_speed_kms > 0 else 0.0

        return t

    # ---- decision (requirement #4) ----

    def should_change_signal(self, ambulance: Ambulance, junction_id: str) -> bool:
        """
        True if the junction is predicted to be red at the moment the
        ambulance would naturally reach it (and therefore needs an
        intervention to keep the corridor clear).
        """
        predicted_arrival = self.predicted_arrival_time(ambulance, junction_id)
        return not self.signals.is_green(junction_id, predicted_arrival)

    # ---- queue-clearing estimate (requirement #6) ----

    def queue_clearing_time(self, junction_id: str) -> float:
        """Seconds needed to clear whatever vehicle queue currently sits at this junction."""
        return self.traffic_sim.queues[junction_id].clearance_time_s()

    # ---- green corridor creation + predictive pre-clearing (requirements #3, #5, #7) ----

    def pre_clear_upcoming(self, ambulance: Ambulance, at_time_s: float) -> List[SignalIntervention]:
        """
        Evaluate the next `pre_clear_lookahead` junctions ahead of the
        ambulance and force-green any that would otherwise be red when the
        ambulance arrives. No-ops (and applies nothing) unless emergency
        mode is enabled. Every intervention made is recorded and returned.
        """
        if not self.emergency_mode:
            return []

        applied: List[SignalIntervention] = []
        for junction_id in self.upcoming_junctions(ambulance):
            if not self.should_change_signal(ambulance, junction_id):
                continue

            predicted_arrival = self.predicted_arrival_time(ambulance, junction_id)
            queue_time = self.queue_clearing_time(junction_id)
            hold_until = predicted_arrival + queue_time + self.green_hold_buffer_s

            self.signals.force_green(junction_id, hold_until)

            reason = (
                f"Junction {junction_id} predicted red at t={predicted_arrival:.1f}s "
                f"(ambulance ETA there); forcing green and holding until t={hold_until:.1f}s "
                f"to also clear a {queue_time:.1f}s vehicle queue."
            )
            record = SignalIntervention(
                junction_id=junction_id,
                action="force_green",
                requested_sim_time_s=at_time_s,
                predicted_arrival_s=predicted_arrival,
                effective_until_s=hold_until,
                queue_clearing_s=queue_time,
                reason=reason,
            )
            self.interventions.append(record)
            applied.append(record)

        return applied

    def get_interventions(self) -> List[SignalIntervention]:
        """Full audit log of every signal intervention made so far."""
        return list(self.interventions)

    # ---- ETA comparison (requirement #8) ----

    def compare_eta(self, route: Route, eta_calculator: ETACalculator,
                     start_time_s: float = 0.0) -> ETAComparison:
        """
        Compare the baseline ETA (signals behave normally, requirement #1)
        against the idealized emergency green-corridor ETA (every junction
        along the route stays green and no queue delay applies, i.e. a
        perfectly executed corridor) and report the time saved.
        """
        baseline: ETAResult = eta_calculator.calculate(route, start_time_s=start_time_s, obey_signals=True)
        emergency: ETAResult = eta_calculator.calculate(route, start_time_s=start_time_s, obey_signals=False)

        saved = baseline.total_time_s - emergency.total_time_s
        percent = (saved / baseline.total_time_s * 100.0) if baseline.total_time_s > 0 else 0.0

        return ETAComparison(
            baseline_seconds=baseline.total_time_s,
            emergency_seconds=emergency.total_time_s,
            time_saved_seconds=saved,
            time_saved_percent=percent,
        )
