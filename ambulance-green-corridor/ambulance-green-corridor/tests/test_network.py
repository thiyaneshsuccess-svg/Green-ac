import pytest
from network import RoadNetwork, Intersection, RoadSegment


def test_add_intersection_and_lookup(simple_network):
    node = simple_network.get_intersection("A")
    assert node.id == "A"
    assert node.x == 0.0 and node.y == 0.0


def test_duplicate_intersection_raises():
    net = RoadNetwork()
    net.add_intersection(Intersection(id="A", x=0, y=0))
    with pytest.raises(ValueError):
        net.add_intersection(Intersection(id="A", x=1, y=1))


def test_segment_endpoints_must_exist():
    net = RoadNetwork()
    net.add_intersection(Intersection(id="A", x=0, y=0))
    with pytest.raises(ValueError):
        net.add_segment(RoadSegment(id="AZ", from_id="A", to_id="Z", length_km=1.0))


def test_bidirectional_segment_creates_both_directions(simple_network):
    forward = simple_network.find_segment("A", "B")
    backward = simple_network.find_segment("B", "A")
    assert forward is not None and backward is not None
    assert forward.length_km == backward.length_km == 10.0


def test_neighbors(simple_network):
    neighbor_ids = {n_id for n_id, _ in simple_network.neighbors("A")}
    assert neighbor_ids == {"B", "C"}


def test_distance_calculation():
    a = Intersection(id="A", x=0.0, y=0.0)
    b = Intersection(id="B", x=3.0, y=4.0)
    assert a.distance_to(b) == pytest.approx(5.0)  # 3-4-5 triangle


def test_capacity_scales_with_lanes():
    seg = RoadSegment(id="X", from_id="A", to_id="B", length_km=1.0, lanes=3,
                       capacity_vehicles_per_lane=40)
    assert seg.capacity == 120


def test_free_flow_time():
    seg = RoadSegment(id="X", from_id="A", to_id="B", length_km=50.0, speed_limit_kmh=100.0)
    assert seg.free_flow_time_hr() == pytest.approx(0.5)


def test_validate_passes_on_consistent_network(simple_network):
    simple_network.validate()  # should not raise
