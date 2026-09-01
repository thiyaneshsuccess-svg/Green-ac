import pytest
from traffic import TrafficSimulator, SignalController, SignalTiming
from route import RoutePlanner
from ambulance import Ambulance, AmbulanceMover
from eta import ETACalculator
from green_corridor import GreenCorridorController
from hospital import HospitalRegistry, Hospital
from tools import SimulationTools, ToolError
from ai_engine import (
    AIDecision, DecisionValidationError, AIServiceUnavailable,
    LocalDecisionEngine, RemoteAIDecisionEngine, HybridDecisionEngine,
    build_decision_context, validate_decision, run_decision_cycle,
)


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


VALID_DECISION_DICT = {
    "selected_route": ["A", "B", "D"],
    "signal_actions": [{"junction": "B", "action": "force_green"}],
    "pre_clear_junctions": ["B"],
    "reroute": False,
    "hospital_eta": 720.0,
    "reason": "Because it is fastest.",
}


# ---- AIDecision strict schema ----

def test_from_dict_accepts_valid_decision():
    decision = AIDecision.from_dict(VALID_DECISION_DICT)
    assert decision.selected_route == ["A", "B", "D"]
    assert decision.reroute is False
    assert decision.hospital_eta == 720.0


def test_from_dict_rejects_missing_key():
    bad = dict(VALID_DECISION_DICT)
    del bad["reason"]
    with pytest.raises(DecisionValidationError):
        AIDecision.from_dict(bad)


def test_from_dict_rejects_extra_key():
    bad = dict(VALID_DECISION_DICT)
    bad["extra_field"] = "nope"
    with pytest.raises(DecisionValidationError):
        AIDecision.from_dict(bad)


def test_from_dict_rejects_wrong_type_for_reroute():
    bad = dict(VALID_DECISION_DICT)
    bad["reroute"] = "false"  # string, not bool
    with pytest.raises(DecisionValidationError):
        AIDecision.from_dict(bad)


def test_from_dict_rejects_negative_eta():
    bad = dict(VALID_DECISION_DICT)
    bad["hospital_eta"] = -5.0
    with pytest.raises(DecisionValidationError):
        AIDecision.from_dict(bad)


def test_from_dict_rejects_empty_reason():
    bad = dict(VALID_DECISION_DICT)
    bad["reason"] = "   "
    with pytest.raises(DecisionValidationError):
        AIDecision.from_dict(bad)


def test_from_dict_rejects_non_object():
    with pytest.raises(DecisionValidationError):
        AIDecision.from_dict(["not", "a", "dict"])


def test_round_trip_to_dict_from_dict():
    decision = AIDecision.from_dict(VALID_DECISION_DICT)
    assert AIDecision.from_dict(decision.to_dict()) == decision


# ---- build_decision_context ----

def test_context_has_all_required_fields(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    expected_keys = {
        "ambulance_location", "destination", "current_route", "available_routes",
        "traffic_density", "queue_lengths", "signal_states", "eta_for_each_route",
        "estimated_congestion",
    }
    assert set(context.keys()) == expected_keys
    assert context["ambulance_location"] == "A"
    assert context["destination"] == "D"
    assert len(context["available_routes"]) == len(context["eta_for_each_route"])


# ---- LocalDecisionEngine ----

def test_local_engine_picks_lowest_eta_route(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    decision = LocalDecisionEngine().decide(context)

    best_idx = context["eta_for_each_route"].index(min(context["eta_for_each_route"]))
    assert decision.selected_route == context["available_routes"][best_idx]
    assert decision.reroute is False  # A->B->D is already the current route
    assert decision.hospital_eta == pytest.approx(context["eta_for_each_route"][best_idx])
    assert decision.reason.strip() != ""


def test_local_engine_requests_preclear_when_next_junction_red(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network, b_offset=50.0)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    decision = LocalDecisionEngine().decide(context)

    assert "B" in decision.pre_clear_junctions
    assert {"junction": "B", "action": "force_green"} in decision.signal_actions


def test_local_engine_no_preclear_when_all_green(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network, b_offset=0.0)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    decision = LocalDecisionEngine().decide(context)

    assert decision.pre_clear_junctions == []
    assert decision.signal_actions == []


def test_local_engine_output_always_passes_validation(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network, b_offset=50.0)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    decision = LocalDecisionEngine().decide(context)
    validate_decision(decision, context, tools.network)  # should not raise


# ---- validate_decision (state-level, beyond schema) ----

def test_validate_rejects_route_not_offered(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    bad = AIDecision(selected_route=["A", "Z", "D"], signal_actions=[], pre_clear_junctions=[],
                      reroute=True, hospital_eta=1.0, reason="bogus")
    with pytest.raises(DecisionValidationError):
        validate_decision(bad, context, tools.network)


def test_validate_rejects_preclear_beyond_lookahead(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    # "D" is 2 hops ahead in a 3-node route (A,B,D) -> index 2, which IS within
    # the next-two window (indices 1,2), so use a bogus junction not on the route.
    bad = AIDecision(selected_route=["A", "B", "D"], signal_actions=[],
                      pre_clear_junctions=["C"], reroute=False, hospital_eta=1.0, reason="bogus")
    with pytest.raises(DecisionValidationError):
        validate_decision(bad, context, tools.network)


def test_validate_rejects_unsupported_signal_action(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    bad = AIDecision(selected_route=context["available_routes"][0],
                      signal_actions=[{"junction": "B", "action": "make_purple"}],
                      pre_clear_junctions=[], reroute=False, hospital_eta=1.0, reason="bogus")
    with pytest.raises(DecisionValidationError):
        validate_decision(bad, context, tools.network)


def test_validate_rejects_implausible_hospital_eta(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    bad = AIDecision(selected_route=context["available_routes"][0], signal_actions=[],
                      pre_clear_junctions=[], reroute=False, hospital_eta=999999.0, reason="bogus")
    with pytest.raises(DecisionValidationError):
        validate_decision(bad, context, tools.network)


# ---- RemoteAIDecisionEngine unavailability ----

def test_remote_engine_unavailable_without_api_key(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    engine = RemoteAIDecisionEngine(api_key=None)
    with pytest.raises(AIServiceUnavailable):
        engine.decide(context)


# ---- HybridDecisionEngine fallback behavior ----

class _FailingRemote:
    def decide(self, context):
        raise AIServiceUnavailable("simulated API outage")


class _InvalidJsonRemote:
    def decide(self, context):
        raise DecisionValidationError("simulated malformed AI response")


class _WorkingRemote:
    def __init__(self, decision):
        self.decision = decision
        self.called = False

    def decide(self, context):
        self.called = True
        return self.decision


def test_hybrid_falls_back_to_local_when_remote_unavailable(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network, b_offset=50.0)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)

    hybrid = HybridDecisionEngine(remote_engine=_FailingRemote())
    decision = hybrid.decide(context)

    assert hybrid.last_source == "local"
    validate_decision(decision, context, tools.network)  # local output must be valid


def test_hybrid_falls_back_when_remote_returns_invalid_json(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)

    hybrid = HybridDecisionEngine(remote_engine=_InvalidJsonRemote())
    decision = hybrid.decide(context)

    assert hybrid.last_source == "local"


def test_hybrid_uses_remote_when_it_succeeds(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)
    good_decision = AIDecision(selected_route=context["available_routes"][0], signal_actions=[],
                                pre_clear_junctions=[], reroute=False,
                                hospital_eta=context["eta_for_each_route"][0], reason="remote said so")
    remote = _WorkingRemote(good_decision)

    hybrid = HybridDecisionEngine(remote_engine=remote)
    decision = hybrid.decide(context)

    assert remote.called is True
    assert hybrid.last_source == "remote"
    assert decision.reason == "remote said so"


def test_hybrid_with_no_remote_configured_uses_local(simple_network):
    tools, ambulance, _, _ = _build_tools(simple_network)
    context = build_decision_context(tools, ambulance, at_time_s=0.0)

    hybrid = HybridDecisionEngine()  # no remote engine at all
    decision = hybrid.decide(context)

    assert hybrid.last_source == "local"


# ---- full orchestrated cycle: gather -> decide -> validate -> execute via tools ----

def test_run_decision_cycle_applies_preclear_and_notifies_hospital(simple_network):
    tools, ambulance, signals, _ = _build_tools(simple_network, b_offset=50.0)
    hybrid = HybridDecisionEngine()  # local engine only, no AI API configured

    assert signals.is_green("B", 360.0) is False

    result = run_decision_cycle(tools, hybrid, ambulance, at_time_s=0.0)

    assert result.source == "local"
    assert signals.is_green("B", 360.0) is True  # the red junction got pre-cleared
    assert len(result.applied_signal_changes) == 1
    assert result.hospital_notification is not None
    assert result.hospital_notification["hospital_id"] == "H1"
    assert result.rerouted is False  # A->B->D was already the current route


def test_run_decision_cycle_falls_back_on_a_bad_remote_decision(simple_network):
    tools, ambulance, signals, _ = _build_tools(simple_network, b_offset=50.0)

    # A "remote" engine that returns a schema-valid but state-INVALID decision
    # (route not among the ones offered) — must be caught by validate_decision
    # inside run_decision_cycle and silently replaced with the local engine's output.
    bogus_decision = AIDecision(selected_route=["A", "NOPE", "D"], signal_actions=[],
                                 pre_clear_junctions=[], reroute=True, hospital_eta=1.0,
                                 reason="a broken remote answer")

    class BrokenRemote:
        def decide(self, context):
            return bogus_decision

    hybrid = HybridDecisionEngine(remote_engine=BrokenRemote())
    result = run_decision_cycle(tools, hybrid, ambulance, at_time_s=0.0)

    assert result.source == "local (validation fallback)"
    assert result.decision.selected_route == ["A", "B", "D"]
    assert signals.is_green("B", 360.0) is True
