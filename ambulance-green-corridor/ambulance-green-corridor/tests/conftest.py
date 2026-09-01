"""
conftest.py — shared pytest fixtures.

Builds a small, deterministic road network used across test modules:

        A ----- 10km, 100km/h ----- B ----- 10km, 100km/h ----- D
        |                                                       |
        +------------- 30km, 100km/h (slow long way) -----------+
                              via C

So A->B->D (20km) is the fast/short route, and A->C->D (30km) is the
deliberately worse alternative — useful for testing that routing picks
the cheaper path and that rerouting logic (later) has something to
switch to.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from network import RoadNetwork, Intersection, RoadSegment  # noqa: E402


@pytest.fixture
def simple_network() -> RoadNetwork:
    net = RoadNetwork()
    net.add_intersection(Intersection(id="A", x=0.0, y=0.0))
    net.add_intersection(Intersection(id="B", x=10.0, y=0.0))
    net.add_intersection(Intersection(id="C", x=0.0, y=10.0))
    net.add_intersection(Intersection(id="D", x=10.0, y=10.0))

    net.add_segment(RoadSegment(id="AB", from_id="A", to_id="B",
                                 length_km=10.0, speed_limit_kmh=100.0), bidirectional=True)
    net.add_segment(RoadSegment(id="BD", from_id="B", to_id="D",
                                 length_km=10.0, speed_limit_kmh=100.0), bidirectional=True)
    net.add_segment(RoadSegment(id="AC", from_id="A", to_id="C",
                                 length_km=15.0, speed_limit_kmh=100.0), bidirectional=True)
    net.add_segment(RoadSegment(id="CD", from_id="C", to_id="D",
                                 length_km=15.0, speed_limit_kmh=100.0), bidirectional=True)
    return net
