import pytest
from traffic import TrafficSimulator, SignalController, SignalTiming
from route import RoutePlanner
from ambulance import Ambulance, AmbulanceMover, AmbulanceStatus


def _setup(simple_network, no_signals=True):
    route = RoutePlanner(simple_network).shortest_path("A", "D")  # A->B->D, 20km
    traffic_sim = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        traffic_sim.set_density(seg_id, 0.0)
    if no_signals:
        for node in simple_network.intersections.values():
            node.has_signal = False
    signals = SignalController(simple_network, seed=1)
    mover = AmbulanceMover(simple_network, traffic_sim, signals)
    ambulance = Ambulance(id="amb-1", route=route, max_speed_kmh=100.0)
    return ambulance, mover, traffic_sim, signals


def test_dispatch_sets_status_en_route(simple_network):
    ambulance, mover, _, _ = _setup(simple_network)
    assert ambulance.status == AmbulanceStatus.IDLE
    mover.dispatch(ambulance)
    assert ambulance.status == AmbulanceStatus.EN_ROUTE


def test_ambulance_reaches_destination_with_no_congestion_or_signals(simple_network):
    ambulance, mover, _, _ = _setup(simple_network, no_signals=True)
    mover.dispatch(ambulance)

    # Total free-flow time is 720s (see test_eta). Step in 60s increments.
    for _ in range(13):  # 780s total, comfortably past 720s
        mover.step(ambulance, dt_s=60.0, obey_signals=True)

    assert ambulance.status == AmbulanceStatus.ARRIVED
    assert mover.progress_fraction(ambulance) == pytest.approx(1.0)
    assert mover.distance_traveled_km(ambulance) == pytest.approx(20.0)


def test_ambulance_partial_progress_mid_route(simple_network):
    ambulance, mover, _, _ = _setup(simple_network, no_signals=True)
    mover.dispatch(ambulance)

    mover.step(ambulance, dt_s=180.0, obey_signals=True)  # 180s of a 360s first segment

    assert ambulance.status == AmbulanceStatus.EN_ROUTE
    assert ambulance.segment_index == 0
    # 100 km/h for 180s = 5 km
    assert mover.distance_traveled_km(ambulance) == pytest.approx(5.0)
    assert mover.progress_fraction(ambulance) == pytest.approx(5.0 / 20.0)


def test_ambulance_waits_at_red_light(simple_network):
    ambulance, mover, traffic_sim, signals = _setup(simple_network, no_signals=False)
    # Force B red at the moment the ambulance would arrive (t=360s)
    signals.timings["B"] = SignalTiming(intersection_id="B", cycle_length_s=60.0,
                                         green_duration_s=30.0, offset_s=50.0)
    mover.dispatch(ambulance)

    mover.step(ambulance, dt_s=360.0, obey_signals=True)  # exactly reaches B as light turns red

    assert ambulance.status == AmbulanceStatus.WAITING_AT_SIGNAL
    assert ambulance.segment_index == 0  # hasn't advanced onto BD yet
    assert mover.distance_traveled_km(ambulance) == pytest.approx(10.0)  # sitting at B


def test_ambulance_upcoming_nodes(simple_network):
    ambulance, mover, _, _ = _setup(simple_network, no_signals=True)
    mover.dispatch(ambulance)
    assert ambulance.upcoming_nodes(2) == ["B", "D"]


def test_higher_congestion_slows_ambulance(simple_network):
    route = RoutePlanner(simple_network).shortest_path("A", "D")
    for node in simple_network.intersections.values():
        node.has_signal = False
    signals = SignalController(simple_network, seed=1)

    sim_low = TrafficSimulator(simple_network, seed=1)
    sim_high = TrafficSimulator(simple_network, seed=1)
    for seg_id in simple_network.all_segment_ids():
        sim_low.set_density(seg_id, 0.1)
        sim_high.set_density(seg_id, 0.9)

    amb_low = Ambulance(id="low", route=route, max_speed_kmh=100.0)
    amb_high = Ambulance(id="high", route=route, max_speed_kmh=100.0)
    mover_low = AmbulanceMover(simple_network, sim_low, signals)
    mover_high = AmbulanceMover(simple_network, sim_high, signals)
    mover_low.dispatch(amb_low)
    mover_high.dispatch(amb_high)

    mover_low.step(amb_low, dt_s=120.0)
    mover_high.step(amb_high, dt_s=120.0)

    assert mover_low.distance_traveled_km(amb_low) > mover_high.distance_traveled_km(amb_high)
