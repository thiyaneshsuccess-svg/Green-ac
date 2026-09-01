import pytest
from traffic import TrafficSimulator, SignalController, SignalTiming
from route import RoutePlanner
from ambulance import Ambulance, AmbulanceMover
from eta import ETACalculator
from green_corridor import GreenCorridorController
from hospital import HospitalRegistry, Hospital
from tools import SimulationTools, ToolError


def _build_tools(simple_network, b_offset=None, d_has_signal=False):
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
    return tools, ambulance, signals, traffic_sim


# ---- get_traffic ----

def test_get_traffic_shape(simple_network):
    tools, ambulance, signals, traffic_sim = _build_tools(simple_network)
    traffic = tools.get_traffic(at_time_s=0.0)
    assert set(traffic.keys()) == {"segments", "queues", "signals", "at_time_s"}
    assert "AB" in traffic["segments"]
    assert "B" in traffic["queues"]
    assert "B" in traffic["signals"]
    assert isinstance(traffic["signals"]["B"]["green"], bool)


# ---- get_ambulance_position ----

def test_get_ambulance_position(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    pos = tools.get_ambulance_position(ambulance)
    assert pos["current_node"] == "A"
    assert pos["next_node"] == "B"
    assert pos["status"] == "en_route"
    assert pos["remaining_route"] == ["A", "B", "D"]


# ---- get_available_routes ----

def test_get_available_routes_returns_primary_and_alternate(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    routes = tools.get_available_routes(ambulance, at_time_s=0.0, k=2)
    node_lists = [r["nodes"] for r in routes]
    assert ["A", "B", "D"] in node_lists
    assert ["A", "C", "D"] in node_lists
    for r in routes:
        assert r["eta_seconds"] > 0


def test_get_available_routes_k1_returns_only_primary(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    routes = tools.get_available_routes(ambulance, at_time_s=0.0, k=1)
    assert len(routes) == 1
    assert routes[0]["nodes"] == ["A", "B", "D"]


# ---- change_signal ----

def test_change_signal_force_green_applies_and_logs(simple_network):
    tools, ambulance, signals, _ = _build_tools(simple_network, b_offset=50.0)
    assert signals.is_green("B", 360.0) is False

    result = tools.change_signal("B", "force_green", until_time_s=400.0, requested_at_s=0.0,
                                  reason="test override")
    assert result["junction_id"] == "B"
    assert signals.is_green("B", 360.0) is True
    assert len(tools.corridor.get_interventions()) == 1
    assert tools.corridor.get_interventions()[0].reason == "test override"


def test_change_signal_release_clears_override(simple_network):
    tools, ambulance, signals, _ = _build_tools(simple_network, b_offset=50.0)
    tools.change_signal("B", "force_green", until_time_s=400.0, requested_at_s=0.0)
    assert signals.is_green("B", 360.0) is True

    tools.change_signal("B", "release")
    assert signals.is_green("B", 360.0) is False


def test_change_signal_rejects_unknown_junction(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    with pytest.raises(ToolError):
        tools.change_signal("ZZZ", "force_green", until_time_s=100.0)


def test_change_signal_rejects_bad_action(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    with pytest.raises(ToolError):
        tools.change_signal("B", "make_purple", until_time_s=100.0)


def test_change_signal_rejects_force_green_without_future_time(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    with pytest.raises(ToolError):
        tools.change_signal("B", "force_green", until_time_s=None)
    with pytest.raises(ToolError):
        tools.change_signal("B", "force_green", until_time_s=5.0, requested_at_s=10.0)


# ---- reroute_ambulance ----

def test_reroute_ambulance_accepts_valid_alternate(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    result = tools.reroute_ambulance(ambulance, ["A", "C", "D"])
    assert result["new_route"] == ["A", "C", "D"]
    assert ambulance.route.node_ids == ["A", "C", "D"]
    assert ambulance.segment_index == 0
    assert ambulance.distance_into_segment_km == 0.0


def test_reroute_ambulance_rejects_wrong_start(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    with pytest.raises(ToolError):
        tools.reroute_ambulance(ambulance, ["C", "D"])  # ambulance is at A, not C


def test_reroute_ambulance_rejects_wrong_destination(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    with pytest.raises(ToolError):
        tools.reroute_ambulance(ambulance, ["A", "B"])  # doesn't end at D


def test_reroute_ambulance_rejects_disconnected_path(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    with pytest.raises(ToolError):
        tools.reroute_ambulance(ambulance, ["A", "D"])  # no direct A-D segment


# ---- notify_hospital ----

def test_notify_hospital_success(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    result = tools.notify_hospital(ambulance, eta_seconds=720.0, at_time_s=0.0)
    assert result["hospital_id"] == "H1"
    assert result["eta_seconds"] == 720.0


def test_notify_hospital_no_hospital_registered_raises(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    tools.hospitals.hospitals.clear()
    with pytest.raises(ToolError):
        tools.notify_hospital(ambulance, eta_seconds=720.0, at_time_s=0.0)
