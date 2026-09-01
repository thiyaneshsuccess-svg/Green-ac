import pytest
from traffic import TrafficSimulator, SignalController, SignalTiming
from route import RoutePlanner
from eta import ETACalculator


def _route_a_to_d(simple_network):
    planner = RoutePlanner(simple_network)
    return planner.shortest_path("A", "D")  # A->B->D, 20km total


def test_free_flow_eta_matches_manual_calculation(simple_network):
    route = _route_a_to_d(simple_network)
    traffic_sim = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)  # congestion_factor == 1.0

    signals = SignalController(simple_network, seed=1)
    calc = ETACalculator(simple_network, traffic_sim, signals)

    result = calc.calculate(route, start_time_s=0.0, obey_signals=False)

    # AB: 10km @ 100km/h = 0.1hr = 360s ; BD: same = 360s
    assert result.total_time_s == pytest.approx(720.0)
    assert result.total_time_min == pytest.approx(12.0)
    assert result.arrival_time_s == pytest.approx(720.0)
    assert len(result.segment_etas) == 2


def test_higher_congestion_increases_eta(simple_network):
    route = _route_a_to_d(simple_network)
    signals = SignalController(simple_network, seed=1)

    sim_low = TrafficSimulator(simple_network, seed=1)
    sim_high = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        sim_low.set_density(seg_id, 0.1)
        sim_high.set_density(seg_id, 0.9)

    calc_low = ETACalculator(simple_network, sim_low, signals)
    calc_high = ETACalculator(simple_network, sim_high, signals)

    eta_low = calc_low.calculate(route, obey_signals=False).total_time_s
    eta_high = calc_high.calculate(route, obey_signals=False).total_time_s

    assert eta_high > eta_low


def test_red_light_adds_wait_time(simple_network):
    route = _route_a_to_d(simple_network)
    traffic_sim = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)

    signals = SignalController(simple_network, seed=1)
    # Force intersection B to be red at the moment the ambulance arrives (t=360s)
    signals.timings["B"] = SignalTiming(intersection_id="B", cycle_length_s=60.0,
                                         green_duration_s=30.0, offset_s=50.0)
    # phase = (360 + 50) % 60 = 50 -> red (>=30) ; wait = 60 - 50 = 10s

    calc = ETACalculator(simple_network, traffic_sim, signals)
    result = calc.calculate(route, start_time_s=0.0, obey_signals=True)

    assert result.segment_etas[0].signal_wait_s == pytest.approx(10.0)
    assert result.total_time_s == pytest.approx(720.0 + 10.0)


def test_queue_clearance_adds_delay(simple_network):
    route = _route_a_to_d(simple_network)
    traffic_sim = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)
    traffic_sim.queues["B"].queue_length = 5  # 5 vehicles * 2s = 10s

    # Make B and D always-green so only the queue delay shows up (must disable
    # signals before constructing the controller, which snapshots timings)
    simple_network.intersections["B"].has_signal = False
    simple_network.intersections["D"].has_signal = False
    signals = SignalController(simple_network, seed=1)

    calc = ETACalculator(simple_network, traffic_sim, signals)
    result = calc.calculate(route, start_time_s=0.0, obey_signals=True)

    assert result.segment_etas[0].queue_clearance_s == pytest.approx(10.0)
    assert result.total_time_s == pytest.approx(720.0 + 10.0)


def test_obey_signals_false_ignores_red_lights_and_queues(simple_network):
    route = _route_a_to_d(simple_network)
    traffic_sim = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)
    traffic_sim.queues["B"].queue_length = 10

    signals = SignalController(simple_network, seed=1)
    signals.timings["B"] = SignalTiming(intersection_id="B", cycle_length_s=60.0,
                                         green_duration_s=30.0, offset_s=50.0)

    calc = ETACalculator(simple_network, traffic_sim, signals)
    result = calc.calculate(route, obey_signals=False)

    assert result.total_time_s == pytest.approx(720.0)
