"""
demo_ai_engine.py

End-to-end smoke test of the AI decision engine, run outside pytest.
No ANTHROPIC_API_KEY is set here, so this exercises exactly the required
"AI API unavailable -> automatic local deterministic engine" path.

Run with: python demo_ai_engine.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network import RoadNetwork, Intersection, RoadSegment
from traffic import TrafficSimulator, SignalController, SignalTiming
from route import RoutePlanner
from eta import ETACalculator
from ambulance import Ambulance, AmbulanceMover
from green_corridor import GreenCorridorController
from hospital import HospitalRegistry, Hospital
from tools import SimulationTools
from ai_engine import HybridDecisionEngine, RemoteAIDecisionEngine, build_decision_context, run_decision_cycle


def main():
    net = RoadNetwork()
    for nid, x, y in [("A", 0, 0), ("B", 10, 0), ("C", 0, 10), ("D", 10, 10)]:
        net.add_intersection(Intersection(id=nid, x=x, y=y))
    net.add_segment(RoadSegment(id="AB", from_id="A", to_id="B", length_km=10, speed_limit_kmh=100), bidirectional=True)
    net.add_segment(RoadSegment(id="BD", from_id="B", to_id="D", length_km=10, speed_limit_kmh=100), bidirectional=True)
    net.add_segment(RoadSegment(id="AC", from_id="A", to_id="C", length_km=15, speed_limit_kmh=100), bidirectional=True)
    net.add_segment(RoadSegment(id="CD", from_id="C", to_id="D", length_km=15, speed_limit_kmh=100), bidirectional=True)

    traffic_sim = TrafficSimulator(net, seed=1)
    for seg_id in net.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)

    signals = SignalController(net, seed=1)
    # Force B red right when the ambulance would arrive, to prove pre-clearing kicks in
    signals.timings["B"] = SignalTiming(intersection_id="B", cycle_length_s=60.0,
                                         green_duration_s=30.0, offset_s=50.0)
    net.intersections["D"].has_signal = False

    planner = RoutePlanner(net)
    eta_calc = ETACalculator(net, traffic_sim, signals)
    mover = AmbulanceMover(net, traffic_sim, signals)
    corridor = GreenCorridorController(net, traffic_sim, signals)

    hospitals = HospitalRegistry()
    hospitals.register(Hospital(id="H1", name="City General Hospital", node_id="D"))

    ambulance = Ambulance(id="amb-1", route=planner.shortest_path("A", "D"), max_speed_kmh=100.0)
    mover.dispatch(ambulance)

    tools = SimulationTools(net, traffic_sim, signals, planner, eta_calc, mover, corridor, hospitals)

    # Real API interface is wired up, but with no ANTHROPIC_API_KEY set this
    # will raise AIServiceUnavailable internally and HybridDecisionEngine
    # falls back to the local deterministic engine automatically.
    remote = RemoteAIDecisionEngine(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    engine = HybridDecisionEngine(remote_engine=remote)

    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    print("=== Structured JSON input to the AI ===")
    print(json.dumps(context, indent=2))

    # Compare BEFORE any intervention is applied, so this reflects true baseline vs. idealized corridor
    comparison = corridor.compare_eta(ambulance.route, eta_calc, start_time_s=0.0)
    print(f"\nBaseline ETA (normal signals):      {comparison.baseline_seconds:.1f}s")
    print(f"Emergency ETA (idealized corridor):  {comparison.emergency_seconds:.1f}s")
    print(f"Time saved:                          {comparison.time_saved_seconds:.1f}s ({comparison.time_saved_percent:.1f}%)")

    result = run_decision_cycle(tools, engine, ambulance, at_time_s=0.0)

    print(f"\n=== Decision (source: {result.source}) ===")
    print(json.dumps(result.decision.to_dict(), indent=2))

    print(f"\nSignal B green at t=360s (ambulance's predicted arrival)? {signals.is_green('B', 360.0)}")
    print(f"Signal interventions logged: {len(corridor.get_interventions())}")
    for i in corridor.get_interventions():
        print(f"  - {i.reason}")
    print(f"Rerouted: {result.rerouted}")
    print(f"Hospital notification: {result.hospital_notification}")


if __name__ == "__main__":
    main()
