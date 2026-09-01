import pytest
from traffic import TrafficSimulator, SignalController, SignalTiming
from route import RoutePlanner
from ambulance import Ambulance, AmbulanceMover
from eta import ETACalculator
from green_corridor import GreenCorridorController


def _setup(simple_network, b_offset=None, d_has_signal=False):
    """
    A->B->D route (20km, zero congestion). D's signal is disabled by
    default so tests can focus purely on B's behavior. If b_offset is
    given, B's signal timing is overridden to a known, testable offset.
    """
    route = RoutePlanner(simple_network).shortest_path("A", "D")
    traffic_sim = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)

    if not d_has_signal:
        simple_network.intersections["D"].has_signal = False

    signals = SignalController(simple_network, seed=1)
    if b_offset is not None:
        signals.timings["B"] = SignalTiming(intersection_id="B", cycle_length_s=60.0,
                                             green_duration_s=30.0, offset_s=b_offset)

    mover = AmbulanceMover(simple_network, traffic_sim, signals)
    ambulance = Ambulance(id="amb-1", route=route, max_speed_kmh=100.0)
    mover.dispatch(ambulance)

    controller = GreenCorridorController(simple_network, traffic_sim, signals)
    eta_calc = ETACalculator(simple_network, traffic_sim, signals)

    return ambulance, mover, traffic_sim, signals, controller, eta_calc, route


# ---- requirement: normal signal behavior is untouched ----

def test_normal_mode_does_not_alter_signals(simple_network):
    ambulance, _, _, signals, controller, _, _ = _setup(simple_network, b_offset=50.0)
    assert controller.emergency_mode is False
    # B is red at predicted arrival (t=360s); normal mode must leave it that way
    assert signals.is_green("B", 360.0) is False
    controller.pre_clear_upcoming(ambulance, at_time_s=0.0)
    assert signals.is_green("B", 360.0) is False
    assert controller.get_interventions() == []


# ---- "the correct junction is identified" ----

def test_next_junction_is_identified_correctly(simple_network):
    ambulance, _, _, _, controller, _, _ = _setup(simple_network)
    assert controller.next_junction(ambulance) == "B"


# ---- "the next two junctions are identified" ----

def test_next_two_junctions_are_identified(simple_network):
    ambulance, _, _, _, controller, _, _ = _setup(simple_network)
    assert controller.upcoming_junctions(ambulance) == ["B", "D"]
    assert controller.upcoming_junctions(ambulance, count=1) == ["B"]


def test_predicted_arrival_time_matches_manual_calculation(simple_network):
    ambulance, _, _, _, controller, _, _ = _setup(simple_network)
    # AB and BD are each 10km @ 100km/h with zero congestion => 360s each
    assert controller.predicted_arrival_time(ambulance, "B") == pytest.approx(360.0)
    assert controller.predicted_arrival_time(ambulance, "D") == pytest.approx(720.0)


def test_predicted_arrival_rejects_node_behind_ambulance(simple_network):
    ambulance, _, _, _, controller, _, _ = _setup(simple_network)
    with pytest.raises(ValueError):
        controller.predicted_arrival_time(ambulance, "A")


# ---- "calculate whether the signal should be changed" ----

def test_should_change_signal_true_when_predicted_red(simple_network):
    ambulance, _, _, _, controller, _, _ = _setup(simple_network, b_offset=50.0)
    # phase at t=360: (360+50)%60 = 50 -> red
    assert controller.should_change_signal(ambulance, "B") is True


def test_should_change_signal_false_when_predicted_green(simple_network):
    ambulance, _, _, _, controller, _, _ = _setup(simple_network, b_offset=0.0)
    # phase at t=360: (360+0)%60 = 0 -> green
    assert controller.should_change_signal(ambulance, "B") is False


# ---- queue-clearing estimate ----

def test_queue_clearing_time_scales_with_queue_length(simple_network):
    ambulance, _, traffic_sim, _, controller, _, _ = _setup(simple_network)
    traffic_sim.queues["B"].queue_length = 5
    assert controller.queue_clearing_time("B") == pytest.approx(10.0)  # 5 * 2.0s/vehicle
    traffic_sim.queues["B"].queue_length = 0
    assert controller.queue_clearing_time("B") == pytest.approx(0.0)


# ---- "signals change correctly" (green corridor creation + predictive pre-clear) ----

def test_pre_clear_forces_green_and_records_intervention(simple_network):
    ambulance, _, _, signals, controller, _, _ = _setup(simple_network, b_offset=50.0)
    controller.enable_emergency_mode()

    applied = controller.pre_clear_upcoming(ambulance, at_time_s=0.0)

    assert len(applied) == 1
    assert applied[0].junction_id == "B"
    assert applied[0].action == "force_green"
    assert applied[0].predicted_arrival_s == pytest.approx(360.0)
    # no queue set => hold_until = predicted_arrival + 0 + buffer(8.0)
    assert applied[0].effective_until_s == pytest.approx(368.0)

    # The signal must now actually read green at the predicted arrival time
    assert signals.is_green("B", 360.0) is True

    log = controller.get_interventions()
    assert len(log) == 1
    assert log[0].junction_id == "B"
    assert "B" in log[0].reason


def test_pre_clear_hold_time_includes_queue_clearing(simple_network):
    ambulance, _, traffic_sim, signals, controller, _, _ = _setup(simple_network, b_offset=50.0)
    traffic_sim.queues["B"].queue_length = 5  # 10s clearance
    controller.enable_emergency_mode()

    applied = controller.pre_clear_upcoming(ambulance, at_time_s=0.0)

    assert applied[0].queue_clearing_s == pytest.approx(10.0)
    # hold_until = 360 (predicted) + 10 (queue) + 8 (buffer) = 378
    assert applied[0].effective_until_s == pytest.approx(378.0)


def test_pre_clear_skips_junctions_already_green(simple_network):
    ambulance, _, _, _, controller, _, _ = _setup(simple_network, b_offset=0.0)  # green at t=360
    controller.enable_emergency_mode()

    applied = controller.pre_clear_upcoming(ambulance, at_time_s=0.0)

    assert applied == []
    assert controller.get_interventions() == []


def test_pre_clear_only_evaluates_next_two_junctions(simple_network):
    # 3x3-style extension isn't needed here: with only B and D ahead, this
    # confirms the lookahead never exceeds the two configured junctions,
    # even when both would need clearing.
    ambulance, _, _, signals, controller, _, _ = _setup(
        simple_network, b_offset=50.0, d_has_signal=True
    )
    signals.timings["D"] = SignalTiming(intersection_id="D", cycle_length_s=60.0,
                                         green_duration_s=30.0, offset_s=50.0)
    controller.enable_emergency_mode()

    applied = controller.pre_clear_upcoming(ambulance, at_time_s=0.0)

    junction_ids = {i.junction_id for i in applied}
    assert junction_ids == {"B", "D"}
    assert len(applied) == 2  # exactly the configured lookahead of 2, nothing more


def test_disable_emergency_mode_releases_overrides(simple_network):
    ambulance, _, _, signals, controller, _, _ = _setup(simple_network, b_offset=50.0)
    controller.enable_emergency_mode()
    controller.pre_clear_upcoming(ambulance, at_time_s=0.0)
    assert signals.is_green("B", 360.0) is True

    controller.disable_emergency_mode()
    assert signals.is_green("B", 360.0) is False  # back to the normal (red) schedule


# ---- ETA / time-saved comparison ----

def test_compare_eta_baseline_worse_than_emergency_with_red_light(simple_network):
    ambulance, _, _, signals, controller, eta_calc, route = _setup(simple_network, b_offset=50.0)

    comparison = controller.compare_eta(route, eta_calc, start_time_s=0.0)

    assert comparison.baseline_seconds == pytest.approx(720.0 + 10.0)  # 10s wait at B
    assert comparison.emergency_seconds == pytest.approx(720.0)
    assert comparison.time_saved_seconds == pytest.approx(10.0)
    assert comparison.time_saved_percent == pytest.approx(10.0 / 730.0 * 100.0)


def test_compare_eta_no_savings_when_all_green(simple_network):
    ambulance, _, _, signals, controller, eta_calc, route = _setup(simple_network, b_offset=0.0)

    comparison = controller.compare_eta(route, eta_calc, start_time_s=0.0)

    assert comparison.baseline_seconds == pytest.approx(720.0)
    assert comparison.emergency_seconds == pytest.approx(720.0)
    assert comparison.time_saved_seconds == pytest.approx(0.0)
    assert comparison.time_saved_percent == pytest.approx(0.0)
