import pytest
from traffic import TrafficSimulator, SignalController


def test_signal_deterministic_offset_with_seed(simple_network):
    ctrl_a = SignalController(simple_network, seed=42)
    ctrl_b = SignalController(simple_network, seed=42)
    for node_id in simple_network.all_intersection_ids():
        assert ctrl_a.timings[node_id].offset_s == ctrl_b.timings[node_id].offset_s


def test_signal_different_seeds_can_differ(simple_network):
    ctrl_a = SignalController(simple_network, seed=1)
    ctrl_b = SignalController(simple_network, seed=2)
    offsets_a = [ctrl_a.timings[n].offset_s for n in simple_network.all_intersection_ids()]
    offsets_b = [ctrl_b.timings[n].offset_s for n in simple_network.all_intersection_ids()]
    assert offsets_a != offsets_b


def test_signal_cycles_green_then_red(simple_network):
    ctrl = SignalController(simple_network, default_cycle_s=60, default_green_s=30, seed=0)
    node_id = simple_network.all_intersection_ids()[0]
    offset = ctrl.timings[node_id].offset_s

    # At t such that phase=0 -> green
    t_green = (-offset) % 60
    assert ctrl.is_green(node_id, t_green) is True

    # At t such that phase=45 -> red (past green_duration of 30)
    t_red = (45 - offset) % 60
    assert ctrl.is_green(node_id, t_red) is False


def test_time_until_green_is_zero_when_already_green(simple_network):
    ctrl = SignalController(simple_network, default_cycle_s=60, default_green_s=30, seed=0)
    node_id = simple_network.all_intersection_ids()[0]
    offset = ctrl.timings[node_id].offset_s
    t_green = (-offset) % 60
    assert ctrl.time_until_green(node_id, t_green) == 0.0


def test_no_signal_node_is_always_green(simple_network):
    simple_network.intersections["A"].has_signal = False
    ctrl = SignalController(simple_network, seed=0)
    assert ctrl.is_green("A", 12345) is True
    assert ctrl.time_until_green("A", 12345) == 0.0


def test_traffic_density_deterministic_with_seed(simple_network):
    sim_a = TrafficSimulator(simple_network, seed=7)
    sim_b = TrafficSimulator(simple_network, seed=7)
    for _ in range(20):
        sim_a.step(5.0)
        sim_b.step(5.0)
    for seg_id in simple_network.all_segment_ids():
        assert sim_a.get_density(seg_id) == pytest.approx(sim_b.get_density(seg_id))


def test_density_stays_within_bounds(simple_network):
    sim = TrafficSimulator(simple_network, seed=3, volatility=0.5)
    for _ in range(200):
        sim.step(10.0)
    for seg_id in simple_network.all_segment_ids():
        d = sim.get_density(seg_id)
        assert 0.0 <= d <= 1.5


def test_congestion_factor_increases_with_density(simple_network):
    sim = TrafficSimulator(simple_network, seed=1)
    seg_id = simple_network.all_segment_ids()[0]

    sim.set_density(seg_id, 0.1)
    low = sim.get_congestion_factor(seg_id)

    sim.set_density(seg_id, 0.9)
    high = sim.get_congestion_factor(seg_id)

    assert high > low
    assert low >= 1.0


def test_manual_density_override(simple_network):
    sim = TrafficSimulator(simple_network, seed=1)
    seg_id = simple_network.all_segment_ids()[0]
    sim.set_density(seg_id, 0.77)
    assert sim.get_density(seg_id) == pytest.approx(0.77)
