from dataclasses import dataclass, field
from typing import Set, Tuple, Dict, Optional

# Assuming FeasibleRoute is defined elsewhere
from src.interfaces import FeasibleRoute


@dataclass
class RVGraph:
    """
    Container for the pairwise request-vehicle graph.
    Supports O(1) lookups for both R-R (Shareability) and R-V (Reachability).
    """

    # --- Internal Storage ---
    # R-R: Undirected Adjacency {r_id: {neighbor_r_ids}}
    _rr_adj: Dict[str, Set[str]] = field(default_factory=dict)

    # R-V: Directed Adjacency {r_id: {v_ids}}
    _rv_adj: Dict[str, Set[str]] = field(default_factory=dict)

    # V-R: Reverse Index {v_id: {r_ids}} (Critical for Intersection Pruning)
    _vr_adj: Dict[str, Set[str]] = field(default_factory=dict)

    # Details
    _rr_details: Dict[Tuple[str, str], FeasibleRoute] = field(default_factory=dict)
    _rv_details: Dict[Tuple[str, str], FeasibleRoute] = field(default_factory=dict)

    # --- Public Accessors (Properties) ---

    @property
    def rr_edges(self) -> Dict[str, Set[str]]:
        """Read-only view of Request-Request adjacency."""
        return self._rr_adj

    @property
    def rv_edges(self) -> Dict[str, Set[str]]:
        """Read-only view of Request-Vehicle adjacency (Which vehicles serve this req?)."""
        return self._rv_adj

    # --- Modifiers ---

    def add_rr_edge(self, r1_id: str, r2_id: str, route: FeasibleRoute) -> None:
        """Add undirected shareability edge."""
        self._rr_adj.setdefault(r1_id, set()).add(r2_id)
        self._rr_adj.setdefault(r2_id, set()).add(r1_id)
        self._rr_details[(r1_id, r2_id)] = route
        self._rr_details[(r2_id, r1_id)] = route

    def add_rv_edge(self, r_id: str, v_id: str, route: FeasibleRoute) -> None:
        """Add directed reachability edge."""
        self._rv_adj.setdefault(r_id, set()).add(v_id)
        self._rv_details[(r_id, v_id)] = route
        # Maintain Reverse Index
        self._vr_adj.setdefault(v_id, set()).add(r_id)

    # --- Fast Lookups ---

    def get_connected_vehicles(self, r_id: str) -> Set[str]:
        """Who can serve this request? (Used for Intersection Pruning)"""
        return self._rv_adj.get(r_id, set())

    def get_connected_requests(self, v_id: str) -> Set[str]:
        """Who can this vehicle serve? (Used for Vehicle-Centric steps)"""
        return self._vr_adj.get(v_id, set())

    def get_shareable_neighbors(self, r_id: str) -> Set[str]:
        """Who can share with this request? (Used for Clique Expansion)"""
        return self._rr_adj.get(r_id, set())

    def get_rr_route(self, r1: str, r2: str) -> Optional[FeasibleRoute]:
        return self._rr_details.get((r1, r2))

    def get_rv_route(self, r: str, v: str) -> Optional[FeasibleRoute]:
        return self._rv_details.get((r, v))
