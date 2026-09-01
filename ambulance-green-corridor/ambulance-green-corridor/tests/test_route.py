import pytest
from route import RoutePlanner, Route


def test_shortest_path_picks_shorter_route(simple_network):
    planner = RoutePlanner(simple_network)
    route = planner.shortest_path("A", "D")
    assert route.node_ids == ["A", "B", "D"]
    assert route.segment_ids == ["AB", "BD"]
    assert route.total_distance_km == pytest.approx(20.0)


def test_shortest_path_avoids_longer_alternative(simple_network):
    planner = RoutePlanner(simple_network)
    route = planner.shortest_path("A", "D")
    assert "C" not in route.node_ids


def test_same_start_and_end_is_trivial(simple_network):
    planner = RoutePlanner(simple_network)
    route = planner.shortest_path("A", "A")
    assert route.is_trivial()
    assert route.total_distance_km == 0.0
    assert len(route) == 0


def test_unknown_node_raises(simple_network):
    planner = RoutePlanner(simple_network)
    with pytest.raises(ValueError):
        planner.shortest_path("A", "ZZZ")
    with pytest.raises(ValueError):
        planner.shortest_path("ZZZ", "A")


def test_custom_weight_function_changes_route(simple_network):
    planner = RoutePlanner(simple_network)

    # Force the AB segment to look extremely costly so A->C->D wins instead
    def weight_fn(segment):
        if segment.id in ("AB", "AB_r"):
            return 1000.0
        return segment.length_km

    route = planner.shortest_path("A", "D", weight_fn=weight_fn)
    assert route.node_ids == ["A", "C", "D"]


def test_route_helper_properties(simple_network):
    planner = RoutePlanner(simple_network)
    route = planner.shortest_path("A", "D")
    assert route.origin == "A"
    assert route.destination == "D"
    assert not route.is_trivial()
