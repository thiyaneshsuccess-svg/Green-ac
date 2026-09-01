"""
app.py — Ambulance Green Corridor: live Streamlit dashboard.

Wires together every backend component built so far (network, traffic,
signals, ambulance movement, ETA, the green-corridor traffic-control
layer, the AI decision engine with local fallback, and dynamic route
selection) into a single animated, judge-readable demo.

Run with:  streamlit run app.py
"""

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from network import RoadNetwork, Intersection, RoadSegment
from traffic import TrafficSimulator, SignalController
from route import RoutePlanner, Route
from eta import ETACalculator
from ambulance import Ambulance, AmbulanceMover, AmbulanceStatus
from green_corridor import GreenCorridorController
from hospital import HospitalRegistry, Hospital
from tools import SimulationTools
from ai_engine import HybridDecisionEngine, RemoteAIDecisionEngine, build_decision_context
from route_selection import DynamicRouteSelector, evaluate_route


# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------

TICK_SECONDS = 18.0    # simulated seconds advanced per animation frame
FRAME_DELAY = 0.18     # real seconds between frames while playing
GRID_SPACING = 170
MARGIN = 70

# 3x3 grid layout: (column, row)
GRID_LAYOUT = {
    "A": (0, 0), "B": (1, 0), "C": (2, 0),
    "D": (0, 1), "E": (1, 1), "F": (2, 1),
    "G": (0, 2), "H": (1, 2), "I": (2, 2),
}
EDGE_LENGTHS_KM = {
    ("A", "B"): 5.0, ("B", "C"): 7.0, ("D", "E"): 6.0, ("E", "F"): 5.0,
    ("G", "H"): 4.0, ("H", "I"): 6.0, ("A", "D"): 6.0, ("D", "G"): 5.0,
    ("B", "E"): 4.0, ("E", "H"): 7.0, ("C", "F"): 5.0, ("F", "I"): 4.0,
}
ORIGIN_NODE = "A"
DESTINATION_NODE = "I"

STATUS_LABELS = {
    "idle": "⚪ Idle",
    "en_route": "🟢 En Route",
    "waiting_at_signal": "🟠 Waiting at Signal",
    "arrived": "✅ Arrived",
}


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------

def build_world(seed: int) -> dict:
    net = RoadNetwork()
    for node_id, (col, row) in GRID_LAYOUT.items():
        net.add_intersection(Intersection(
            id=node_id, x=MARGIN + col * GRID_SPACING, y=MARGIN + row * GRID_SPACING,
        ))
    for i, ((u, v), length_km) in enumerate(EDGE_LENGTHS_KM.items()):
        net.add_segment(RoadSegment(id=f"s{i}", from_id=u, to_id=v,
                                     length_km=length_km, speed_limit_kmh=60.0),
                         bidirectional=True)

    traffic_sim = TrafficSimulator(net, seed=seed)
    signals = SignalController(net, seed=seed)
    baseline_signals = SignalController(net, seed=seed)  # pristine, never overridden — true baseline

    planner = RoutePlanner(net)
    eta_calc = ETACalculator(net, traffic_sim, signals)
    baseline_eta_calc = ETACalculator(net, traffic_sim, baseline_signals)
    mover = AmbulanceMover(net, traffic_sim, signals)
    corridor = GreenCorridorController(net, traffic_sim, signals)
    selector = DynamicRouteSelector(net, traffic_sim, signals, planner, eta_calc)

    hospitals = HospitalRegistry()
    hospitals.register(Hospital(id="H1", name="City General Hospital", node_id=DESTINATION_NODE))

    ambulance = Ambulance(id="amb-1", route=planner.shortest_path(ORIGIN_NODE, DESTINATION_NODE),
                           max_speed_kmh=110.0)
    mover.dispatch(ambulance)

    tools = SimulationTools(net, traffic_sim, signals, planner, eta_calc, mover, corridor, hospitals)

    remote = RemoteAIDecisionEngine(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    engine = HybridDecisionEngine(remote_engine=remote)

    return {
        "net": net, "traffic_sim": traffic_sim, "signals": signals,
        "baseline_signals": baseline_signals, "planner": planner,
        "eta_calc": eta_calc, "baseline_eta_calc": baseline_eta_calc,
        "mover": mover, "corridor": corridor, "selector": selector,
        "hospitals": hospitals, "ambulance": ambulance, "tools": tools, "engine": engine,
        "sim_time": 0.0, "running": False, "emergency_mode": False,
        "reroute_count": 0, "last_decision": None, "last_outcome": None,
    }


def reset_world() -> None:
    st.session_state.world = build_world(seed=int(time.time() * 1000) % 100000)


def remaining_route(w: dict) -> Route:
    amb = w["ambulance"]
    idx = amb.segment_index
    node_ids = amb.route.node_ids[idx:]
    segment_ids = amb.route.segment_ids[idx:]
    total_km = sum(w["net"].get_segment(s).length_km for s in segment_ids)
    return Route(node_ids=node_ids, segment_ids=segment_ids, total_distance_km=total_km)


def compute_live_eta(w: dict, signals: SignalController, eta_calculator: ETACalculator,
                      at_time_s: float) -> float:
    """
    Accurate ETA-from-now for the ambulance's remaining journey.

    Uses GreenCorridorController.predicted_arrival_time for the partial
    segment currently being traversed (it correctly accounts for distance
    already covered — a plain Route/ETACalculator sweep does not, since it
    always measures from a segment's start node), then hands off to
    ETACalculator for the rest of the route, which consists of full,
    not-yet-started segments where that ambiguity doesn't exist.
    """
    ambulance = w["ambulance"]
    if ambulance.status == AmbulanceStatus.ARRIVED:
        return 0.0
    next_node = ambulance.next_node()
    if next_node is None:
        return 0.0

    arrival_at_next = w["corridor"].predicted_arrival_time(ambulance, next_node)
    signal_wait = signals.time_until_green(next_node, arrival_at_next)
    queue_wait = w["traffic_sim"].get_queue_length(next_node) * 2.0
    departure_time = arrival_at_next + signal_wait + queue_wait

    idx = ambulance.segment_index + 1
    rest_segment_ids = ambulance.route.segment_ids[idx:]
    if not rest_segment_ids:
        return departure_time - at_time_s

    rest_route = Route(
        node_ids=ambulance.route.node_ids[idx:],
        segment_ids=rest_segment_ids,
        total_distance_km=sum(w["net"].get_segment(s).length_km for s in rest_segment_ids),
    )
    rest_result = eta_calculator.calculate(rest_route, start_time_s=departure_time, obey_signals=True)
    return rest_result.arrival_time_s - at_time_s


# ---------------------------------------------------------------------------
# One simulation tick
# ---------------------------------------------------------------------------

def run_tick(w: dict) -> None:
    ambulance = w["ambulance"]
    if ambulance.status == AmbulanceStatus.ARRIVED:
        w["running"] = False
        return

    w["sim_time"] += TICK_SECONDS
    w["traffic_sim"].step(TICK_SECONDS)
    w["mover"].step(ambulance, TICK_SECONDS, obey_signals=True)

    if ambulance.status == AmbulanceStatus.ARRIVED:
        w["running"] = False

    # AI decision, for display, every tick (cheap: local engine, or a real API call if configured)
    try:
        context = build_decision_context(w["tools"], ambulance, w["sim_time"])
        w["last_decision"] = w["engine"].decide(context)
    except Exception:
        pass

    if w["emergency_mode"] and ambulance.status != AmbulanceStatus.ARRIVED:
        w["corridor"].pre_clear_upcoming(ambulance, w["sim_time"])
        outcome = w["selector"].consider_reroute(ambulance, w["tools"], w["engine"], w["sim_time"])
        w["last_outcome"] = outcome
        if outcome.rerouted:
            w["reroute_count"] += 1


# ---------------------------------------------------------------------------
# Network SVG rendering
# ---------------------------------------------------------------------------

def density_color(density: float) -> str:
    if density < 0.3:
        return "#3ABF7E"
    if density < 0.6:
        return "#E8A23D"
    return "#DD4B4B"


def render_network_svg(w: dict) -> str:
    net = w["net"]
    traffic_sim = w["traffic_sim"]
    signals = w["signals"]
    ambulance = w["ambulance"]
    sim_time = w["sim_time"]
    route_segment_ids = set(ambulance.route.segment_ids)
    upcoming = set(ambulance.upcoming_nodes(2)) if w["emergency_mode"] else set()

    width = MARGIN * 2 + 2 * GRID_SPACING
    height = width

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;background:#0E1420;border-radius:14px;">',
        "<defs>",
        '<style>',
        '@keyframes pulseGold { 0%,100% { opacity: 1; r: 15; } 50% { opacity: 0.45; r: 19; } }',
        '@keyframes pulseAmb { 0%,100% { filter: drop-shadow(0 0 4px #ff4d4d); } '
        '50% { filter: drop-shadow(0 0 12px #ff4d4d); } }',
        '.cleared { animation: pulseGold 1.1s ease-in-out infinite; }',
        '.amb-glow { animation: pulseAmb 1.1s ease-in-out infinite; }',
        '.veh { fill: #55617a; opacity: 0.85; }',
        '</style>',
        "</defs>",
    ]

    # Edges (roads)
    for seg_id in net.all_segment_ids():
        segment = net.get_segment(seg_id)
        u = net.get_intersection(segment.from_id)
        v = net.get_intersection(segment.to_id)
        if segment.from_id > segment.to_id:
            continue  # bidirectional pairs share one visual line
        density = traffic_sim.get_density(seg_id)
        on_route = seg_id in route_segment_ids
        color = "#3D7FD9" if on_route else density_color(density)
        stroke_width = 7 if on_route else 4
        parts.append(
            f'<line x1="{u.x}" y1="{u.y}" x2="{v.x}" y2="{v.y}" '
            f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" opacity="0.9"/>'
        )
        # scattered "traffic" dots, density-proportional, deterministic per edge
        n_dots = min(6, round(density * 8))
        for k in range(n_dots):
            frac = (k + 1) / (n_dots + 1)
            dx = u.x + (v.x - u.x) * frac
            dy = u.y + (v.y - u.y) * frac
            jitter = 6 if (hash((seg_id, k)) % 2 == 0) else -6
            perp_x, perp_y = -(v.y - u.y), (v.x - u.x)
            norm = max(1.0, (perp_x ** 2 + perp_y ** 2) ** 0.5)
            dx += (perp_x / norm) * jitter
            dy += (perp_y / norm) * jitter
            parts.append(f'<circle class="veh" cx="{dx:.1f}" cy="{dy:.1f}" r="3.2"/>')

    # Hospital marker
    hospital_node = net.get_intersection(DESTINATION_NODE)
    parts.append(
        f'<rect x="{hospital_node.x - 22}" y="{hospital_node.y - 46}" width="44" height="30" rx="6" '
        f'fill="#1C2536" stroke="#7FA8E0" stroke-width="2"/>'
        f'<text x="{hospital_node.x}" y="{hospital_node.y - 26}" text-anchor="middle" '
        f'font-size="18" fill="#E24B4A" font-weight="700">+</text>'
        f'<text x="{hospital_node.x}" y="{hospital_node.y - 52}" text-anchor="middle" '
        f'font-size="11" fill="#B9C4D6" font-family="sans-serif">HOSPITAL</text>'
    )

    # Intersections + signals
    for node_id in net.all_intersection_ids():
        node = net.get_intersection(node_id)
        is_override = signals.has_override(node_id, sim_time)
        is_green = signals.is_green(node_id, sim_time)
        fill = "#3ABF7E" if is_green else "#DD4B4B"
        if node_id == ambulance.current_node() or node_id == ambulance.next_node():
            pass  # ring drawn below regardless
        if is_override:
            parts.append(
                f'<circle class="cleared" cx="{node.x}" cy="{node.y}" r="15" '
                f'fill="none" stroke="#F2C94C" stroke-width="3"/>'
            )
        elif node_id in upcoming:
            parts.append(
                f'<circle cx="{node.x}" cy="{node.y}" r="14" fill="none" '
                f'stroke="#F2C94C" stroke-width="2" stroke-dasharray="3,3" opacity="0.8"/>'
            )
        parts.append(f'<circle cx="{node.x}" cy="{node.y}" r="9" fill="{fill}" '
                     f'stroke="#0E1420" stroke-width="2"/>')
        parts.append(f'<text x="{node.x}" y="{node.y + 26}" text-anchor="middle" font-size="12" '
                     f'fill="#B9C4D6" font-family="sans-serif">{node_id}</text>')

    # Ambulance position (interpolated along current segment)
    seg_id = ambulance.current_segment_id()
    if seg_id is not None:
        segment = net.get_segment(seg_id)
        u = net.get_intersection(segment.from_id)
        v = net.get_intersection(segment.to_id)
        frac = (ambulance.distance_into_segment_km / segment.length_km) if segment.length_km > 0 else 0.0
        ax = u.x + (v.x - u.x) * frac
        ay = u.y + (v.y - u.y) * frac
    else:
        dest = net.get_intersection(ambulance.route.destination)
        ax, ay = dest.x, dest.y

    parts.append(
        f'<circle class="amb-glow" cx="{ax:.1f}" cy="{ay:.1f}" r="11" fill="#FF4D4D" '
        f'stroke="#ffffff" stroke-width="2"/>'
        f'<text x="{ax:.1f}" y="{ay:.1f}" text-anchor="middle" dominant-baseline="central" '
        f'font-size="13">🚑</text>'
    )

    # Legend
    legend_y = height - 20
    legend_items = [
        ("#3ABF7E", "Clear"), ("#E8A23D", "Moderate"), ("#DD4B4B", "Congested / Red signal"),
        ("#F2C94C", "AI-cleared signal"), ("#3D7FD9", "Ambulance route"),
    ]
    lx = 14
    for color, label in legend_items:
        parts.append(f'<rect x="{lx}" y="{legend_y - 9}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{lx + 17}" y="{legend_y}" font-size="10.5" fill="#B9C4D6" '
                     f'font-family="sans-serif">{label}</text>')
        lx += 17 + len(label) * 6 + 18

    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Ambulance Green Corridor", layout="wide", page_icon="🚑")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .banner {
        background: linear-gradient(90deg, #16213A 0%, #0E1420 100%);
        border-radius: 12px; padding: 14px 20px; margin-bottom: 14px;
        border: 1px solid #24304A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "world" not in st.session_state:
    reset_world()
w = st.session_state.world

st.markdown(
    '<div class="banner"><h2 style="margin:0;">🚑 AI-Powered Ambulance Green Corridor</h2>'
    '<p style="margin:4px 0 0 0;color:#9AA7BD;">Live digital-twin simulation — routing, '
    'traffic signals, and AI decisions, end to end.</p></div>',
    unsafe_allow_html=True,
)

# ---- Controls ----
btn_cols = st.columns([1.3, 1.6, 1, 1.6])
with btn_cols[0]:
    label = "⏸ PAUSE" if w["running"] else "▶ START SIMULATION"
    if st.button(label, type="primary", use_container_width=True):
        w["running"] = not w["running"]
with btn_cols[1]:
    em_label = "🚨 EMERGENCY MODE: ON" if w["emergency_mode"] else "🚨 ACTIVATE EMERGENCY MODE"
    if st.button(em_label, use_container_width=True):
        w["emergency_mode"] = not w["emergency_mode"]
        if w["emergency_mode"]:
            w["corridor"].enable_emergency_mode()
        else:
            w["corridor"].disable_emergency_mode()
with btn_cols[2]:
    if st.button("⟲ RESET", use_container_width=True):
        reset_world()
        st.rerun()
with btn_cols[3]:
    source = w["engine"].last_source
    if source == "remote":
        st.success("🟢 LIVE AI (remote API)", icon="🟢")
    else:
        st.warning("🟡 FALLBACK MODE (local engine)", icon="🟡")

# ---- Advance simulation before rendering, so the frame reflects this tick ----
if w["running"]:
    run_tick(w)

ambulance = w["ambulance"]
current_eval = evaluate_route(w["net"], w["traffic_sim"], w["eta_calc"], remaining_route(w), w["sim_time"])
baseline_eta = compute_live_eta(w, w["baseline_signals"], w["baseline_eta_calc"], w["sim_time"])
optimized_eta = compute_live_eta(w, w["signals"], w["eta_calc"], w["sim_time"])
time_saved = baseline_eta - optimized_eta
intersections_cleared = len({i.junction_id for i in w["corridor"].get_interventions()})

# ---- LEFT / RIGHT ----
left, right = st.columns([2, 1.1])

with left:
    st.markdown(render_network_svg(w), unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.markdown("##### Ambulance Status")
        st.metric("Status", STATUS_LABELS.get(ambulance.status.value, ambulance.status.value))
        c1, c2 = st.columns(2)
        c1.metric("Current location", ambulance.current_node())
        c2.metric("Destination", DESTINATION_NODE)
        st.metric("ETA (live)", f"{optimized_eta:.0f} s")
        st.progress(min(1.0, w["mover"].progress_fraction(ambulance)))
        st.caption(f"Traffic density on route: **{current_eval.avg_traffic_density:.2f}** · "
                   f"Queue delay: **{current_eval.queue_delay_s:.0f}s**")

    with st.container(border=True):
        st.markdown("##### AI Decision")
        decision = w["last_decision"]
        if decision is None:
            st.caption("Press START SIMULATION to begin the decision loop.")
        else:
            st.write(f"**Selected route:** {' → '.join(decision.selected_route)}")
            st.write(f"**Reroute requested:** {'Yes' if decision.reroute else 'No'}")
            st.write(f"**Pre-clear junctions:** "
                     f"{', '.join(decision.pre_clear_junctions) if decision.pre_clear_junctions else '—'}")
            st.markdown("**Reason:**")
            st.info(decision.reason)

# ---- BOTTOM metrics ----
st.markdown("&nbsp;", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Baseline ETA", f"{baseline_eta:.0f}s")
m2.metric("Optimized ETA", f"{optimized_eta:.0f}s",
          delta=f"-{time_saved:.0f}s" if time_saved > 0 else "0s", delta_color="inverse")
m3.metric("Time Saved", f"{max(0.0, time_saved):.0f}s")
m4.metric("Intersections Cleared", intersections_cleared)
m5.metric("Reroutes", w["reroute_count"])

if ambulance.status == AmbulanceStatus.ARRIVED:
    st.success(f"🏁 Ambulance arrived at {DESTINATION_NODE} — "
               f"{w['hospitals'].find_by_node(DESTINATION_NODE).name} notified.")

# ---- Animation loop ----
if w["running"] and ambulance.status != AmbulanceStatus.ARRIVED:
    time.sleep(FRAME_DELAY)
    st.rerun()
