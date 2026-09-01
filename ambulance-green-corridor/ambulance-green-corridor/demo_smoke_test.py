"""
demo_smoke_test.py

Quick end-to-end sanity check of components A-E, run outside pytest.
Builds a slightly bigger 3x3 grid network, dispatches an ambulance, and
prints ETA + live position every few ticks until arrival.

Run with:  python demo_smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network import RoadNetwork, Intersection, RoadSegment
from traffic import TrafficSimulator, SignalController
from route import RoutePlanner
from eta import ETACalculator
from ambulance import Ambulance, AmbulanceMover, AmbulanceStatus


def build_grid(width=3, height=3, spacing_km=2.0) -> RoadNetwork:
    net = RoadNetwork()
    for x in range(width):
        for y in range(height):
            net.add_intersection(Intersection(id=f"{x}_{y}", x=x * spacing_km, y=y * spacing_km))

    seg_id = 0
    for x in range(width):
        for y in range(height):
            if x + 1 < width:
                net.add_segment(RoadSegment(id=f"s{seg_id}", from_id=f"{x}_{y}", to_id=f"{x+1}_{y}",
                                             length_km=spacing_km, speed_limit_kmh=60.0),
                                 bidirectional=True)
                seg_id += 1
            if y + 1 < height:
                net.add_segment(RoadSegment(id=f"s{seg_id}", from_id=f"{x}_{y}", to_id=f"{x}_{y+1}",
                                             length_km=spacing_km, speed_limit_kmh=60.0),
                                 bidirectional=True)
                seg_id += 1
    return net


def main():
    SEED = 42  # deterministic "demo mode"
    net = build_grid()
    net.validate()

    traffic_sim = TrafficSimulator(net, seed=SEED)
    signals = SignalController(net, seed=SEED)
    planner = RoutePlanner(net)
    eta_calc = ETACalculator(net, traffic_sim, signals)
    mover = AmbulanceMover(net, traffic_sim, signals)

    start, end = "0_0", "2_2"
    route = planner.shortest_path(start, end,
                                   weight_fn=lambda seg: seg.length_km * traffic_sim.get_congestion_factor(seg.id))
    print(f"Route: {' -> '.join(route.node_ids)}  ({route.total_distance_km:.1f} km)")

    eta_result = eta_calc.calculate(route, start_time_s=0.0, obey_signals=True)
    print(f"Baseline ETA: {eta_result.total_time_min:.2f} min")
    for seg_eta in eta_result.segment_etas:
        print(f"  {seg_eta.from_id} -> {seg_eta.to_id}: "
              f"{seg_eta.travel_time_s:5.1f}s "
              f"(free-flow {seg_eta.free_flow_time_s:.1f}s x{seg_eta.congestion_factor:.2f}, "
              f"signal wait {seg_eta.signal_wait_s:.1f}s, queue {seg_eta.queue_clearance_s:.1f}s)")

    ambulance = Ambulance(id="amb-1", route=route, max_speed_kmh=100.0)
    mover.dispatch(ambulance)

    print("\nSimulating movement (10s ticks):")
    tick = 0
    while ambulance.status != AmbulanceStatus.ARRIVED and tick < 500:
        mover.step(ambulance, dt_s=10.0, obey_signals=True)
        traffic_sim.step(10.0)
        tick += 1
        if tick % 6 == 0:  # print every ~minute
            print(f"  t={ambulance.elapsed_time_s:6.1f}s  status={ambulance.status.value:18s} "
                  f"progress={mover.progress_fraction(ambulance)*100:5.1f}%  "
                  f"at_node={ambulance.current_node()}")

    print(f"\nArrived after {ambulance.elapsed_time_s:.1f}s "
          f"({ambulance.elapsed_time_s/60:.2f} min), "
          f"distance {mover.distance_traveled_km(ambulance):.2f} km")


if __name__ == "__main__":
    main()
