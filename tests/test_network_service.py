"""Travel-time factor: every travel time the service hands out is scaled the same way."""

import numpy as np
import pytest

from src.components.network_service import NetworkXService
from tests.conftest import InMemoryPathCache, make_street_graph


@pytest.fixture
def services():
    graph = make_street_graph(seed=7)
    db = InMemoryPathCache(graph)
    return NetworkXService(graph, db=db), NetworkXService(graph, db=db, travel_time_factor=2.5)


def test_all_outputs_scale_by_the_factor(services):
    free, slow = services
    nodes = list(range(free.N))
    assert np.allclose(slow.get_l1_cache(), 2.5 * free.get_l1_cache())
    assert np.allclose(
        slow.get_travel_time_matrix(nodes[:5], nodes[5:]),
        2.5 * free.get_travel_time_matrix(nodes[:5], nodes[5:]),
    )
    for o, d in [(0, 35), (12, 3), (7, 7)]:
        assert slow.get_travel_time(o, d) == pytest.approx(2.5 * free.get_travel_time(o, d))
        assert slow.get_shortest_path(o, d).travel_time == pytest.approx(
            2.5 * free.get_shortest_path(o, d).travel_time
        )
        p_free = free.get_shortest_path_with_travel_time(o, d)
        p_slow = slow.get_shortest_path_with_travel_time(o, d)
        assert p_slow.nodes == p_free.nodes  # same route, only slower
        assert p_slow.travel_distance == pytest.approx(p_free.travel_distance)
        # what vehicles replay matches what the planner sees
        assert sum(p_slow.travel_times) == pytest.approx(slow.get_travel_time(o, d))


def test_factor_must_be_positive():
    graph = make_street_graph(seed=7)
    with pytest.raises(ValueError):
        NetworkXService(graph, db=InMemoryPathCache(graph), travel_time_factor=0)
