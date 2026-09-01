import pytest
from network import RoadNetwork, Intersection, RoadSegment
from traffic import TrafficSimulator, SignalController
from route import RoutePlanner
from ambulance import Ambulance, AmbulanceMover
from eta import ETACalculator
from green_corridor import GreenCorridorController
from hospital import HospitalRegistry, Hospital
from tools import SimulationTools
from ai_engine import HybridDecisionEngine, AIDecision, AIServiceUnavailable, DecisionValidationError
from route_selection import (
    DynamicRouteSelector, RouteEvaluation, evaluate_route,
    DEFAULT_DEGRADATION_RATIO, DEFAULT_DEGRADATION_SECONDS,
)


def _build(simple_network, d_has_signal=False):
    route = RoutePlanner(simple_network).shortest_path("A", "D")
    traffic_sim = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)

    # Disable all signals so these tests isolate distance/density/queue-delay
    # math cleanly, without incidental signal-wait noise from the seeded schedule.
    for node in simple_network.intersections.values():
        node.has_signal = False

    signals = SignalController(simple_network, seed=1)
    planner = RoutePlanner(simple_network)
    eta_calc = ETACalculator(simple_network, traffic_sim, signals)
    mover = AmbulanceMover(simple_network, traffic_sim, signals)
    corridor = GreenCorridorController(simple_network, traffic_sim, signals)

    hospitals = HospitalRegistry()
    hospitals.register(Hospital(id="H1", name="City General", node_id="D"))

    ambulance = Ambulance(id="amb-1", route=route, max_speed_kmh=100.0)
    mover.dispatch(ambulance)

    tools = SimulationTools(simple_network, traffic_sim, signals, planner, eta_calc,
                             mover, corridor, hospitals)
    selector = DynamicRouteSelector(simple_network, traffic_sim, signals, planner, eta_calc)
    engine = HybridDecisionEngine()  # local only, deterministic

    return tools, ambulance, signals, traffic_sim, selector, engine


# ---- "create at least two possible ambulance routes" + per-route metrics ----

def test_evaluate_route_computes_all_five_metrics(simple_network):
    tools, ambulance, _, _, selector, _ = _build(simple_network)
    route = RoutePlanner(simple_network).shortest_path("A", "D")  # A->B->D, 20km, zero congestion
    ev = evaluate_route(simple_network, tools.traffic_sim, tools.eta_calculator, route, at_time_s=0.0)

    assert ev.nodes == ["A", "B", "D"]
    assert ev.distance_km == pytest.approx(20.0)
    assert ev.avg_traffic_density == pytest.approx(0.0)
    assert ev.queue_delay_s == pytest.approx(0.0)
    assert ev.estimated_travel_time_s == pytest.approx(720.0)
    assert ev.total_eta_s == pytest.approx(720.0)


def test_evaluate_route_reflects_queue_delay(simple_network):
    tools, ambulance, _, traffic_sim, _, _ = _build(simple_network)
    traffic_sim.queues["B"].queue_length = 5  # 10s clearance
    route = RoutePlanner(simple_network).shortest_path("A", "D")
    ev = evaluate_route(simple_network, traffic_sim, tools.eta_calculator, route, at_time_s=0.0)

    assert ev.queue_delay_s == pytest.approx(10.0)
    assert ev.estimated_travel_time_s == pytest.approx(720.0)  # travel time excludes queue delay
    assert ev.total_eta_s == pytest.approx(730.0)              # total ETA includes it


def test_evaluate_route_reflects_density(simple_network):
    tools, ambulance, _, traffic_sim, _, _ = _build(simple_network)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.4)
    route = RoutePlanner(simple_network).shortest_path("A", "D")
    ev = evaluate_route(simple_network, traffic_sim, tools.eta_calculator, route, at_time_s=0.0)

    assert ev.avg_traffic_density == pytest.approx(0.4)
    assert ev.total_eta_s > 720.0  # congestion slows things down


def test_evaluate_current_and_alternatives_returns_at_least_two_routes(simple_network):
    tools, ambulance, _, _, selector, _ = _build(simple_network)
    current_eval, alt_evals = selector.evaluate_current_and_alternatives(ambulance, at_time_s=0.0, k=2)

    assert current_eval.nodes == ["A", "B", "D"]
    assert len(alt_evals) >= 1
    assert alt_evals[0].nodes == ["A", "C", "D"]


# ---- degradation threshold ----

def test_no_trigger_when_current_route_is_fine(simple_network):
    tools, ambulance, _, _, selector, _ = _build(simple_network)
    current_eval, alt_evals = selector.evaluate_current_and_alternatives(ambulance, at_time_s=0.0)
    # A->B->D (20km, 720s) is much better than A->C->D (30km) here, so nothing should trigger
    assert selector.find_significantly_better_alternative(current_eval, alt_evals) is None


def test_trigger_when_current_route_becomes_much_worse(simple_network):
    tools, ambulance, _, traffic_sim, selector, _ = _build(simple_network)
    # A->C->D is 1080s baseline; push A->B->D (720s baseline) well past it
    traffic_sim.queues["B"].queue_length = 300  # +600s queue delay -> current ETA 1320s
    current_eval, alt_evals = selector.evaluate_current_and_alternatives(ambulance, at_time_s=0.0)

    better = selector.find_significantly_better_alternative(current_eval, alt_evals)
    assert better is not None
    assert better.nodes == ["A", "C", "D"]


def test_no_trigger_when_ratio_threshold_not_met(simple_network):
    tools, ambulance, _, traffic_sim, selector, _ = _build(simple_network)
    # Push current ETA to ~1188s (only ~10% worse than the 1080s alternative,
    # below the 15% ratio threshold) even though the absolute gap is large.
    traffic_sim.queues["B"].queue_length = 234  # +468s -> current ETA ~1188s
    current_eval, alt_evals = selector.evaluate_current_and_alternatives(ambulance, at_time_s=0.0)
    assert selector.find_significantly_better_alternative(current_eval, alt_evals) is None


# ---- full consider_reroute cycle ----

def test_consider_reroute_does_nothing_when_route_is_fine(simple_network):
    tools, ambulance, signals, traffic_sim, selector, engine = _build(simple_network)
    outcome = selector.consider_reroute(ambulance, tools, engine, at_time_s=0.0)

    assert outcome.triggered is False
    assert outcome.rerouted is False
    assert outcome.decision_source is None
    assert ambulance.route.node_ids == ["A", "B", "D"]  # unchanged
    assert outcome.old_eta_s == outcome.new_eta_s


def test_consider_reroute_switches_route_and_explains(simple_network):
    tools, ambulance, signals, traffic_sim, selector, engine = _build(simple_network)
    traffic_sim.queues["B"].queue_length = 300  # +600s -> current ETA 1320s, well past A->C->D (1080s)

    outcome = selector.consider_reroute(ambulance, tools, engine, at_time_s=0.0)

    # 1. decision engine was consulted
    assert outcome.decision_source == "local"
    # 2. route was changed
    assert outcome.rerouted is True
    assert ambulance.route.node_ids == ["A", "C", "D"]
    assert outcome.new_route == ["A", "C", "D"]
    # 3. ETA was recalculated
    assert outcome.new_eta_s == pytest.approx(1080.0)  # A->C->D, 30km @ 100km/h, no congestion
    # 4. explanation present and mentions both routes
    assert "Rerouted" in outcome.explanation
    assert "A->B->D" in outcome.explanation
    assert "A->C->D" in outcome.explanation
    # 5. old vs new ETA both displayed
    assert outcome.old_eta_s == pytest.approx(1320.0)
    assert outcome.old_eta_s > outcome.new_eta_s
    assert outcome.old_route == ["A", "B", "D"]


def test_consider_reroute_falls_back_to_local_on_bad_remote_decision(simple_network):
    tools, ambulance, signals, traffic_sim, selector, _ = _build(simple_network)
    traffic_sim.queues["B"].queue_length = 300  # +600s -> current ETA 1320s, well past A->C->D (1080s)

    bogus = AIDecision(selected_route=["A", "NOPE", "D"], signal_actions=[],
                        pre_clear_junctions=[], reroute=True, hospital_eta=1.0,
                        reason="a broken remote answer")

    class BrokenRemote:
        def decide(self, context):
            return bogus

    engine = HybridDecisionEngine(remote_engine=BrokenRemote())
    outcome = selector.consider_reroute(ambulance, tools, engine, at_time_s=0.0)

    assert outcome.decision_source == "local (validation fallback)"
    assert outcome.rerouted is True
    assert ambulance.route.node_ids == ["A", "C", "D"]


def test_consider_reroute_falls_back_when_remote_unavailable(simple_network):
    tools, ambulance, signals, traffic_sim, selector, _ = _build(simple_network)
    traffic_sim.queues["B"].queue_length = 300  # +600s -> current ETA 1320s, well past A->C->D (1080s)

    class OutageRemote:
        def decide(self, context):
            raise AIServiceUnavailable("simulated outage")

    engine = HybridDecisionEngine(remote_engine=OutageRemote())
    outcome = selector.consider_reroute(ambulance, tools, engine, at_time_s=0.0)

    assert outcome.decision_source == "local"
    assert outcome.rerouted is True
