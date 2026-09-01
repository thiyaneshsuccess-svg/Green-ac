"""
route_selection.py — Dynamic Route Selection

Generates at least two candidate ambulance routes, evaluates each one on
distance / traffic density / queue delay / estimated travel time / total
ETA, and — only when the ambulance's current route has become
significantly worse than an alternative — consults the AI decision
engine about switching, applies the switch through the reroute_ambulance
tool, recalculates ETA, and explains what happened.

Built entirely on top of already-existing components:
  - route.py       (RoutePlanner, generate_route_candidates)
  - eta.py          (ETACalculator does the actual travel-time math)
  - tools.py         (reroute_ambulance is the only way a route is
                       actually changed)
  - ai_engine.py     (the decision engine consulted before switching)
"""

from dataclasses import dataclass
from statistics import mean
from typing import List, Optional, Tuple

from network import RoadNetwork
from route import Route, RoutePlanner, generate_route_candidates
from traffic import TrafficSimulator, SignalController
from eta import ETACalculator
from ambulance import Ambulance
from tools import SimulationTools
from ai_engine import (
    HybridDecisionEngine, build_decision_context, validate_decision, DecisionValidationError,
)


# A route must be BOTH this much worse by ratio AND this much worse in
# absolute seconds before a reroute is even considered — guards against
# rerouting over noise (e.g. a 2% difference on a very long route).
DEFAULT_DEGRADATION_RATIO = 1.15
DEFAULT_DEGRADATION_SECONDS = 30.0


@dataclass
class RouteEvaluation:
    """Every metric requested for a single candidate route."""
    nodes: List[str]
    distance_km: float
    avg_traffic_density: float
    queue_delay_s: float
    estimated_travel_time_s: float
    total_eta_s: float

    def summary(self) -> str:
        return (f"{'->'.join(self.nodes)}: {self.distance_km:.1f}km, "
                f"density {self.avg_traffic_density:.2f}, "
                f"queue delay {self.queue_delay_s:.1f}s, "
                f"travel time {self.estimated_travel_time_s:.1f}s, "
                f"total ETA {self.total_eta_s:.1f}s")


def evaluate_route(network: RoadNetwork, traffic_sim: TrafficSimulator,
                    eta_calculator: ETACalculator, route: Route, at_time_s: float) -> RouteEvaluation:
    """Compute distance / density / queue delay / travel time / total ETA for one route."""
    if route.is_trivial():
        return RouteEvaluation(nodes=route.node_ids, distance_km=0.0, avg_traffic_density=0.0,
                                queue_delay_s=0.0, estimated_travel_time_s=0.0, total_eta_s=0.0)

    result = eta_calculator.calculate(route, start_time_s=at_time_s, obey_signals=True)
    travel_time_s = sum(s.free_flow_time_s * s.congestion_factor for s in result.segment_etas)
    queue_delay_s = sum(s.queue_clearance_s for s in result.segment_etas)
    avg_density = mean(traffic_sim.get_density(seg_id) for seg_id in route.segment_ids)

    return RouteEvaluation(
        nodes=route.node_ids,
        distance_km=route.total_distance_km,
        avg_traffic_density=round(avg_density, 3),
        queue_delay_s=round(queue_delay_s, 2),
        estimated_travel_time_s=round(travel_time_s, 2),
        total_eta_s=round(result.total_time_s, 2),
    )


def _congestion_weight(traffic_sim: TrafficSimulator):
    def weight_fn(segment):
        return segment.length_km * traffic_sim.get_congestion_factor(segment.id)
    return weight_fn


@dataclass
class RerouteOutcome:
    """Result of one dynamic-route-selection check."""
    triggered: bool                  # did the current route degrade enough to even ask?
    rerouted: bool                   # did the ambulance actually switch routes?
    decision_source: Optional[str]   # "remote" / "local" / None if never consulted
    old_route: List[str]
    new_route: List[str]
    old_eta_s: float
    new_eta_s: float
    explanation: str


class DynamicRouteSelector:
    """
    Watches the ambulance's current route against live alternatives and,
    only when the current route becomes significantly worse, consults the
    AI decision engine about switching — then actually switches (via the
    reroute_ambulance tool), recalculates ETA, and explains what happened.
    """

    def __init__(self, network: RoadNetwork, traffic_sim: TrafficSimulator,
                 signals: SignalController, planner: RoutePlanner, eta_calculator: ETACalculator,
                 degradation_ratio: float = DEFAULT_DEGRADATION_RATIO,
                 degradation_seconds: float = DEFAULT_DEGRADATION_SECONDS) -> None:
        self.network = network
        self.traffic_sim = traffic_sim
        self.signals = signals
        self.planner = planner
        self.eta_calculator = eta_calculator
        self.degradation_ratio = degradation_ratio
        self.degradation_seconds = degradation_seconds

    def evaluate_current_and_alternatives(
        self, ambulance: Ambulance, at_time_s: float, k: int = 2
    ) -> Tuple[RouteEvaluation, List[RouteEvaluation]]:
        """Create at least two possible routes and evaluate all of them (current + alternatives)."""
        current_node = ambulance.current_node()
        destination = ambulance.route.destination
        remaining_ids = ambulance.route.segment_ids[ambulance.segment_index:]

        remaining_route = Route(
            node_ids=ambulance.route.node_ids[ambulance.segment_index:],
            segment_ids=remaining_ids,
            total_distance_km=sum(self.network.get_segment(sid).length_km for sid in remaining_ids),
        )
        current_eval = evaluate_route(self.network, self.traffic_sim, self.eta_calculator,
                                       remaining_route, at_time_s)

        candidates = generate_route_candidates(self.planner, current_node, destination,
                                                 _congestion_weight(self.traffic_sim), k=k)
        alt_evals = [
            evaluate_route(self.network, self.traffic_sim, self.eta_calculator, route, at_time_s)
            for route in candidates if route.node_ids != remaining_route.node_ids
        ]
        return current_eval, alt_evals

    def find_significantly_better_alternative(
        self, current_eval: RouteEvaluation, alternatives: List[RouteEvaluation]
    ) -> Optional[RouteEvaluation]:
        """
        Returns the best alternative if the current route is significantly
        worse than it (both the ratio AND the absolute-seconds threshold
        must be crossed), else None.
        """
        if not alternatives or current_eval.total_eta_s <= 0:
            return None
        best = min(alternatives, key=lambda a: a.total_eta_s)
        if best.total_eta_s <= 0:
            return None
        ratio = current_eval.total_eta_s / best.total_eta_s
        absolute_gap = current_eval.total_eta_s - best.total_eta_s
        if ratio >= self.degradation_ratio and absolute_gap >= self.degradation_seconds:
            return best
        return None

    def consider_reroute(self, ambulance: Ambulance, tools: SimulationTools,
                          engine: HybridDecisionEngine, at_time_s: float, k: int = 2) -> RerouteOutcome:
        """
        1. Evaluate the current route + at least one alternative.
        2. If the current route isn't significantly worse, do nothing.
        3. Otherwise, ask the decision engine whether to reroute.
        4. If it agrees, switch the ambulance's route via the tool.
        5. Recalculate ETA and explain old vs. new.
        """
        current_eval, alt_evals = self.evaluate_current_and_alternatives(ambulance, at_time_s, k=k)
        better = self.find_significantly_better_alternative(current_eval, alt_evals)

        if better is None:
            return RerouteOutcome(
                triggered=False, rerouted=False, decision_source=None,
                old_route=current_eval.nodes, new_route=current_eval.nodes,
                old_eta_s=current_eval.total_eta_s, new_eta_s=current_eval.total_eta_s,
                explanation=(
                    f"Current route is still efficient (ETA {current_eval.total_eta_s:.1f}s); "
                    f"no alternative is significantly better, so no reroute was requested."
                ),
            )

        # Step 1: ask the decision engine whether to reroute
        context = build_decision_context(tools, ambulance, at_time_s, k_routes=k)
        decision = engine.decide(context)
        try:
            validate_decision(decision, context, self.network)
        except DecisionValidationError:
            decision = engine.local_engine.decide(context)
            validate_decision(decision, context, self.network)
            engine.last_source = "local (validation fallback)"

        if not decision.reroute or decision.selected_route == current_eval.nodes:
            return RerouteOutcome(
                triggered=True, rerouted=False, decision_source=engine.last_source,
                old_route=current_eval.nodes, new_route=current_eval.nodes,
                old_eta_s=current_eval.total_eta_s, new_eta_s=current_eval.total_eta_s,
                explanation=(
                    f"Route degraded to {current_eval.total_eta_s:.1f}s (vs. {better.total_eta_s:.1f}s "
                    f"available via {'->'.join(better.nodes)}), but the decision engine chose to stay "
                    f"on the current route: {decision.reason}"
                ),
            )

        # Step 2: change the ambulance's route — the ONLY mutation point
        tools.reroute_ambulance(ambulance, decision.selected_route)

        # Step 3: recalculate ETA on the new route
        new_eval = evaluate_route(self.network, self.traffic_sim, self.eta_calculator,
                                   ambulance.route, at_time_s)

        # Step 4 + 5: explain why, with old vs. new ETA
        percent_worse = ((current_eval.total_eta_s / better.total_eta_s) - 1.0) * 100.0
        explanation = (
            f"Rerouted from {'->'.join(current_eval.nodes)} (ETA {current_eval.total_eta_s:.1f}s) "
            f"to {'->'.join(new_eval.nodes)} (ETA {new_eval.total_eta_s:.1f}s) because the original "
            f"route had become {percent_worse:.0f}% worse than the best alternative. "
            f"Decision engine ({engine.last_source}) reasoning: {decision.reason}"
        )

        return RerouteOutcome(
            triggered=True, rerouted=True, decision_source=engine.last_source,
            old_route=current_eval.nodes, new_route=new_eval.nodes,
            old_eta_s=current_eval.total_eta_s, new_eta_s=new_eval.total_eta_s,
            explanation=explanation,
        )
