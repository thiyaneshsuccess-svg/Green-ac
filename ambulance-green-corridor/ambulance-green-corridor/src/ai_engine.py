"""
ai_engine.py — AI decision engine.

Flow for one decision cycle:

  1. build_decision_context() gathers exactly the structured JSON the
     assignment specifies, using ONLY the read-only tools in tools.py.
  2. HybridDecisionEngine.decide() tries a real AI API (RemoteAIDecisionEngine)
     first; if it's unavailable, times out, errors, or returns something
     that fails schema validation, it automatically falls back to
     LocalDecisionEngine — a fully deterministic, rule-based engine.
  3. validate_decision() re-checks the decision against live network/route
     state (not just JSON shape) before anything is executed. This runs
     regardless of which engine produced the decision.
  4. run_decision_cycle() applies a validated decision EXCLUSIVELY through
     SimulationTools (change_signal / reroute_ambulance / notify_hospital).

Neither decision engine ever touches RoadNetwork, TrafficSimulator,
SignalController, Ambulance, or HospitalRegistry directly — they only
see/return plain JSON-serializable data.
"""

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from tools import SimulationTools, ALLOWED_SIGNAL_ACTIONS, ToolError
from ambulance import Ambulance
from network import RoadNetwork


REQUIRED_OUTPUT_KEYS = {
    "selected_route", "signal_actions", "pre_clear_junctions",
    "reroute", "hospital_eta", "reason",
}


class DecisionValidationError(Exception):
    """Raised when a decision (from any engine) fails schema or state validation."""


class AIServiceUnavailable(Exception):
    """Raised by RemoteAIDecisionEngine when the external API can't be reached or used."""


# ---------------------------------------------------------------------------
# Strict output schema
# ---------------------------------------------------------------------------

@dataclass
class AIDecision:
    selected_route: List[str]
    signal_actions: List[Dict[str, Any]]
    pre_clear_junctions: List[str]
    reroute: bool
    hospital_eta: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> "AIDecision":
        """Strictly parse + validate the schema. Raises DecisionValidationError on any deviation."""
        if not isinstance(data, dict):
            raise DecisionValidationError("Decision must be a JSON object")

        keys = set(data.keys())
        missing = REQUIRED_OUTPUT_KEYS - keys
        extra = keys - REQUIRED_OUTPUT_KEYS
        if missing:
            raise DecisionValidationError(f"Missing required keys: {sorted(missing)}")
        if extra:
            raise DecisionValidationError(f"Unexpected extra keys: {sorted(extra)}")

        if not isinstance(data["selected_route"], list) or \
                not all(isinstance(n, str) for n in data["selected_route"]):
            raise DecisionValidationError("selected_route must be a list of strings")

        if not isinstance(data["signal_actions"], list):
            raise DecisionValidationError("signal_actions must be a list")
        for action in data["signal_actions"]:
            if not isinstance(action, dict) or "junction" not in action or "action" not in action:
                raise DecisionValidationError("each signal_action must be an object with 'junction' and 'action'")

        if not isinstance(data["pre_clear_junctions"], list) or \
                not all(isinstance(n, str) for n in data["pre_clear_junctions"]):
            raise DecisionValidationError("pre_clear_junctions must be a list of strings")

        if not isinstance(data["reroute"], bool):
            raise DecisionValidationError("reroute must be a boolean")

        eta = data["hospital_eta"]
        if isinstance(eta, bool) or not isinstance(eta, (int, float)):
            raise DecisionValidationError("hospital_eta must be a number")
        if eta < 0:
            raise DecisionValidationError("hospital_eta must be non-negative")

        if not isinstance(data["reason"], str) or not data["reason"].strip():
            raise DecisionValidationError("reason must be a non-empty string")

        return cls(
            selected_route=list(data["selected_route"]),
            signal_actions=[dict(a) for a in data["signal_actions"]],
            pre_clear_junctions=list(data["pre_clear_junctions"]),
            reroute=bool(data["reroute"]),
            hospital_eta=float(eta),
            reason=str(data["reason"]),
        )


# ---------------------------------------------------------------------------
# Structured JSON input (the context handed to whichever engine is used)
# ---------------------------------------------------------------------------

def build_decision_context(tools: SimulationTools, ambulance: Ambulance,
                            at_time_s: float, k_routes: int = 2) -> Dict[str, Any]:
    """Gather exactly the structured input fields the AI needs, via read-only tools only."""
    position = tools.get_ambulance_position(ambulance)
    traffic = tools.get_traffic(at_time_s)
    routes = tools.get_available_routes(ambulance, at_time_s, k=k_routes)

    return {
        "ambulance_location": position["current_node"],
        "destination": ambulance.route.destination,
        "current_route": position["remaining_route"],
        "available_routes": [r["nodes"] for r in routes],
        "traffic_density": {seg: info["density"] for seg, info in traffic["segments"].items()},
        "queue_lengths": traffic["queues"],
        "signal_states": traffic["signals"],
        "eta_for_each_route": [r["eta_seconds"] for r in routes],
        "estimated_congestion": {seg: info["congestion_factor"] for seg, info in traffic["segments"].items()},
    }


# ---------------------------------------------------------------------------
# Validation against live state (defense in depth beyond schema checking)
# ---------------------------------------------------------------------------

def validate_decision(decision: AIDecision, context: Dict[str, Any], network: RoadNetwork) -> None:
    """
    Re-checks a schema-valid AIDecision against the actual network and the
    context it was given. Raises DecisionValidationError on any problem.
    Applied identically no matter which engine produced the decision.
    """
    available = context["available_routes"]
    if decision.selected_route not in available:
        raise DecisionValidationError(
            f"selected_route {decision.selected_route} is not one of the offered available_routes"
        )
    if not decision.selected_route or decision.selected_route[0] != context["ambulance_location"]:
        raise DecisionValidationError("selected_route must start at the ambulance's current location")
    if decision.selected_route[-1] != context["destination"]:
        raise DecisionValidationError("selected_route must end at the destination")

    for u, v in zip(decision.selected_route, decision.selected_route[1:]):
        if network.find_segment(u, v) is None:
            raise DecisionValidationError(f"selected_route contains a disconnected hop: '{u}' -> '{v}'")

    lookahead_nodes = set(decision.selected_route[1:3])  # next two junctions ahead, per requirement
    if len(decision.pre_clear_junctions) > 2:
        raise DecisionValidationError("pre_clear_junctions must not exceed the next two junctions ahead")
    for junction in decision.pre_clear_junctions:
        if junction not in network.intersections:
            raise DecisionValidationError(f"pre_clear_junctions references unknown junction '{junction}'")
        if junction not in lookahead_nodes:
            raise DecisionValidationError(
                f"pre_clear_junctions must be within the next two junctions ahead; got '{junction}'"
            )

    for action in decision.signal_actions:
        junction = action.get("junction")
        act = action.get("action")
        if junction not in network.intersections:
            raise DecisionValidationError(f"signal_actions references unknown junction '{junction}'")
        if act not in ALLOWED_SIGNAL_ACTIONS:
            raise DecisionValidationError(f"signal_actions has an unsupported action '{act}'")

    etas = context.get("eta_for_each_route", [])
    max_reasonable_eta = (max(etas) * 3.0 + 60.0) if etas else 3600.0
    if decision.hospital_eta > max_reasonable_eta:
        raise DecisionValidationError(
            f"hospital_eta ({decision.hospital_eta}) is implausibly larger than any offered route ETA"
        )


# ---------------------------------------------------------------------------
# Local deterministic decision engine (always available, no network calls)
# ---------------------------------------------------------------------------

class LocalDecisionEngine:
    """
    Fully deterministic, rule-based decision engine. Given the same
    context, always returns the same decision. Used as the default engine
    and as the automatic fallback when a remote AI API is unavailable or
    returns something invalid.
    """

    def decide(self, context: Dict[str, Any]) -> AIDecision:
        available_routes = context["available_routes"]
        etas = context["eta_for_each_route"]
        if not available_routes or len(available_routes) != len(etas):
            raise DecisionValidationError("Context must include at least one route with a matching ETA")

        best_idx = min(range(len(available_routes)), key=lambda i: etas[i])
        selected_route = available_routes[best_idx]
        best_eta = etas[best_idx]

        current_route = context["current_route"]
        reroute = selected_route != current_route

        lookahead = selected_route[1:3]
        signal_states = context["signal_states"]

        pre_clear_junctions: List[str] = []
        signal_actions: List[Dict[str, Any]] = []
        for junction in lookahead:
            state = signal_states.get(junction, {"green": True})
            if not state.get("green", True):
                pre_clear_junctions.append(junction)
                signal_actions.append({"junction": junction, "action": "force_green"})

        reason_parts = [
            f"Selected route {'->'.join(selected_route)} (ETA {best_eta:.1f}s), "
            f"the lowest of {len(available_routes)} offered option(s)."
        ]
        if reroute:
            reason_parts.append(
                f"This differs from the current route {'->'.join(current_route)}, so rerouting is requested."
            )
        if pre_clear_junctions:
            reason_parts.append(
                f"Junction(s) {', '.join(pre_clear_junctions)} are red and within the next two "
                f"junctions ahead, so pre-clearing is requested for them."
            )
        else:
            reason_parts.append("No junction in the next two ahead needs clearing; all are already green.")

        return AIDecision(
            selected_route=selected_route,
            signal_actions=signal_actions,
            pre_clear_junctions=pre_clear_junctions,
            reroute=reroute,
            hospital_eta=float(best_eta),
            reason=" ".join(reason_parts),
        )


# ---------------------------------------------------------------------------
# Remote AI engine interface (real API, with automatic fallback on failure)
# ---------------------------------------------------------------------------

class RemoteAIDecisionEngine:
    """
    Thin client for an external AI API (Anthropic Messages API). Given the
    same structured context as LocalDecisionEngine, asks the model to
    return ONLY the strict output JSON schema.

    Never called directly by anything that executes decisions — it's only
    ever used through HybridDecisionEngine, which validates whatever comes
    back and falls back automatically if this raises or returns garbage.
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 8.0) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout_s = timeout_s

    def decide(self, context: Dict[str, Any]) -> AIDecision:
        if not self.api_key:
            raise AIServiceUnavailable("ANTHROPIC_API_KEY is not configured")

        body = json.dumps({
            "model": self.MODEL,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": self._build_prompt(context)}],
        }).encode("utf-8")

        request = urllib.request.Request(
            self.API_URL, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            raise AIServiceUnavailable(f"AI API request failed: {exc}") from exc

        text = "".join(
            block.get("text", "") for block in payload.get("content", [])
            if block.get("type") == "text"
        ).strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AIServiceUnavailable(f"AI response was not valid JSON: {exc}") from exc

        return AIDecision.from_dict(data)  # may raise DecisionValidationError

    @staticmethod
    def _build_prompt(context: Dict[str, Any]) -> str:
        return (
            "You are the traffic-control decision engine for an ambulance green corridor.\n"
            f"Context (structured JSON):\n{json.dumps(context)}\n\n"
            "Respond with ONLY strict JSON matching exactly this schema — no prose, "
            "no markdown code fences, no extra keys:\n"
            '{"selected_route": [], "signal_actions": [{"junction": "", "action": ""}], '
            '"pre_clear_junctions": [], "reroute": false, "hospital_eta": 0, "reason": ""}\n\n'
            "Rules: selected_route must be one of the available_routes (as given, in order), "
            "starting at ambulance_location and ending at destination. signal_actions entries "
            "must use action 'force_green' or 'release' on a junction from signal_states. "
            "pre_clear_junctions may only include junctions from the next two positions in "
            "selected_route after ambulance_location. reroute is true only if selected_route "
            "differs from current_route. hospital_eta should be the ETA (in seconds) of "
            "selected_route."
        )


# ---------------------------------------------------------------------------
# Hybrid engine: remote-first with automatic, transparent local fallback
# ---------------------------------------------------------------------------

class HybridDecisionEngine:
    """
    Tries the remote AI engine (if configured) first. Falls back to the
    local deterministic engine automatically if the remote engine is
    unavailable, errors, times out, or returns a decision that fails
    schema validation. `last_source` records which engine actually
    produced the most recent decision, for transparency/logging.
    """

    def __init__(self, local_engine: Optional[LocalDecisionEngine] = None,
                 remote_engine: Optional[RemoteAIDecisionEngine] = None) -> None:
        self.local_engine = local_engine or LocalDecisionEngine()
        self.remote_engine = remote_engine
        self.last_source: str = "local"

    def decide(self, context: Dict[str, Any]) -> AIDecision:
        if self.remote_engine is not None:
            try:
                decision = self.remote_engine.decide(context)
                self.last_source = "remote"
                return decision
            except (AIServiceUnavailable, DecisionValidationError):
                pass  # fall through to the deterministic engine

        decision = self.local_engine.decide(context)
        self.last_source = "local"
        return decision


# ---------------------------------------------------------------------------
# Orchestration: gather -> decide -> validate -> execute (via tools only)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    decision: AIDecision
    source: str
    applied_signal_changes: List[Dict[str, Any]]
    rerouted: bool
    hospital_notification: Optional[Dict[str, Any]]


def run_decision_cycle(tools: SimulationTools, engine: HybridDecisionEngine,
                        ambulance: Ambulance, at_time_s: float) -> ExecutionResult:
    """
    One full AI decision-and-execution cycle. The engine never touches
    simulation state directly — every effect below goes through a
    SimulationTools method, and nothing is applied until it passes
    validate_decision().
    """
    context = build_decision_context(tools, ambulance, at_time_s)

    decision = engine.decide(context)
    try:
        validate_decision(decision, context, tools.network)
    except DecisionValidationError:
        # Safety net: even a "successfully returned" decision that fails
        # state validation never gets executed — fall back to the local
        # engine, whose output is trusted to always be valid by construction.
        decision = engine.local_engine.decide(context)
        validate_decision(decision, context, tools.network)
        engine.last_source = "local (validation fallback)"

    applied_signals = []
    for action in decision.signal_actions:
        junction = action["junction"]
        until = action.get("until_time_s")
        if until is None and action.get("action") == "force_green":
            # No explicit hold time given — derive one the same way the
            # green-corridor layer itself would: predicted arrival + queue
            # clearance + a safety buffer.
            try:
                predicted = tools.corridor.predicted_arrival_time(ambulance, junction)
                queue_time = tools.corridor.queue_clearing_time(junction)
                until = predicted + queue_time + tools.corridor.green_hold_buffer_s
            except ValueError:
                until = at_time_s + 30.0
        applied_signals.append(tools.change_signal(
            junction, action["action"],
            until_time_s=until, requested_at_s=at_time_s, reason=decision.reason,
        ))

    rerouted = False
    if decision.reroute:
        tools.reroute_ambulance(ambulance, decision.selected_route)
        rerouted = True

    hospital_notification = None
    try:
        hospital_notification = tools.notify_hospital(ambulance, decision.hospital_eta, at_time_s)
    except ToolError:
        hospital_notification = None  # no hospital registered at this destination; not fatal

    return ExecutionResult(
        decision=decision,
        source=engine.last_source,
        applied_signal_changes=applied_signals,
        rerouted=rerouted,
        hospital_notification=hospital_notification,
    )
