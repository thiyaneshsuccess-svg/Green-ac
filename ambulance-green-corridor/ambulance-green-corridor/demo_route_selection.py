"""
demo_route_selection.py

End-to-end smoke test of dynamic route selection, run outside pytest.
Simulates the current route (A->B->D) developing a bad traffic jam at B
after the ambulance is already dispatched, and shows the selector
detecting the degradation, consulting the decision engine, switching to
A->C->D, and explaining why — with old vs. new ETA displayed.

Run with: python demo_route_selection.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network import RoadNetwork, Intersection, RoadSegment
from traffic import TrafficSimulator, SignalController
from route import RoutePlanner
from eta import ETACalculator
from ambulance import Ambulance, AmbulanceMover
from green_corridor import GreenCorridorController
from hospital import HospitalRegistry, Hospital
from tools import SimulationTools
from ai_engine import HybridDecisionEngine
from route_selection import DynamicRouteSelector


def main():
    net = RoadNetwork()
    for nid, x, y in [("A", 0, 0), ("B", 10, 0), ("C", 0, 10), ("D", 10, 10)]:
        net.add_intersection(Intersection(id=nid, x=x, y=y, has_signal=False))
    net.add_segment(RoadSegment(id="AB", from_id="A", to_id="B", length_km=10, speed_limit_kmh=100), bidirectional=True)
    net.add_segment(RoadSegment(id="BD", from_id="B", to_id="D", length_km=10, speed_limit_kmh=100), bidirectional=True)
    net.add_segment(RoadSegment(id="AC", from_id="A", to_id="C", length_km=15, speed_limit_kmh=100), bidirectional=True)
    net.add_segment(RoadSegment(id="CD", from_id="C", to_id="D", length_km=15, speed_limit_kmh=100), bidirectional=True)

    traffic_sim = TrafficSimulator(net, seed=1)
    for seg_id in net.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)
    signals = SignalController(net, seed=1)

    planner = RoutePlanner(net)
    eta_calc = ETACalculator(net, traffic_sim, signals)
    mover = AmbulanceMover(net, traffic_sim, signals)
    corridor = GreenCorridorController(net, traffic_sim, signals)

    hospitals = HospitalRegistry()
    hospitals.register(Hospital(id="H1", name="City General Hospital", node_id="D"))

    ambulance = Ambulance(id="amb-1", route=planner.shortest_path("A", "D"), max_speed_kmh=100.0)
    mover.dispatch(ambulance)

    tools = SimulationTools(net, traffic_sim, signals, planner, eta_calc, mover, corridor, hospitals)
    selector = DynamicRouteSelector(net, traffic_sim, signals, planner, eta_calc)
    engine = HybridDecisionEngine()  # no ANTHROPIC_API_KEY set -> uses the local engine

    print(f"Dispatched on route: {' -> '.join(ambulance.route.node_ids)}\n")

    current_eval, alt_evals = selector.evaluate_current_and_alternatives(ambulance, at_time_s=0.0)
    print("=== Route evaluations (before the jam) ===")
    print("Current: " + current_eval.summary())
    for a in alt_evals:
        print("Alt:     " + a.summary())

    print("\n--- A major queue forms at junction B (300 vehicles backed up) ---\n")
    traffic_sim.queues["B"].queue_length = 300

    outcome = selector.consider_reroute(ambulance, tools, engine, at_time_s=0.0)

    print(f"Triggered:        {outcome.triggered}")
    print(f"Rerouted:         {outcome.rerouted}")
    print(f"Decision source:  {outcome.decision_source}")
    print(f"Old route:        {' -> '.join(outcome.old_route)}  (ETA {outcome.old_eta_s:.1f}s)")
    print(f"New route:        {' -> '.join(outcome.new_route)}  (ETA {outcome.new_eta_s:.1f}s)")
    print(f"\nExplanation: {outcome.explanation}")


if __name__ == "__main__":
    main()
