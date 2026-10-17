
import sys
import unittest
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Mocking the minimal structure needed from the main codebase
@dataclass(frozen=True)
class Stop:
    id: str
    node_id: int
    hard_earliest: float
    hard_latest: float
    type: str = "request"

@dataclass
class RouteTopology:
    stops: List[Stop]
    total_travel_time: float
    earliest_arrivals: List[float]
    latest_arrivals: List[float]

    @property
    def is_valid(self) -> bool:
        # Check E <= L for all nodes
        for e, l in zip(self.earliest_arrivals, self.latest_arrivals):
            if e > l + 1e-6:
                return False
        return True

class MockNetworkService:
    def get_travel_time(self, u: int, v: int) -> float:
        # Simple 1D line metric: distance = abs(u - v)
        # speed = 1 unit/sec
        return float(abs(u - v))

# --- The Logic to Test ---

def validate_sequence(sequence: List[Stop], network: MockNetworkService) -> Optional[RouteTopology]:
    """
    Validates a fixed sequence of stops and computes tightest time windows.
    Returns RouteTopology if valid, None otherwise.
    """
    n = len(sequence)
    if n == 0:
        return RouteTopology([], 0.0, [], [])

    # 1. Forward Pass (Earliest Arrival)
    earliest = [0.0] * n
    # The first node can start no earlier than its hard_earliest
    # But essentially it depends on previous node.
    # For a virtual start (t=0) or just the first request?
    # Let's assume the first stop is reached at max(hard_earliest, prev_arrival + travel)
    
    # We treat the sequence as "Vehicle Start -> ... " or just "Stop1 -> Stop2..."
    # If just request stops, the first stop effectively "starts" when we arrive? 
    # Actually, for the "Request Graph", there is no vehicle yet.
    # So we just respect the time windows relative to each other.
    
    current_time = sequence[0].hard_earliest
    earliest[0] = current_time
    
    for i in range(1, n):
        prev = sequence[i-1]
        curr = sequence[i]
        tt = network.get_travel_time(prev.node_id, curr.node_id)
        
        arrival = max(earliest[i-1] + tt, curr.hard_earliest)
        earliest[i] = arrival

    # 2. Backward Pass (Latest Arrival)
    latest = [0.0] * n
    latest[n-1] = sequence[n-1].hard_latest
    
    for i in range(n-2, -1, -1):
        prev = sequence[i] # u
        curr = sequence[i+1] # v
        tt = network.get_travel_time(prev.node_id, curr.node_id)
        
        # We must arrive at 'curr' by latest[i+1]
        # So we must leave 'prev' by latest[i+1] - tt
        # And we must satisfy prev.hard_latest
        
        required_by_next = latest[i+1] - tt
        latest[i] = min(prev.hard_latest, required_by_next)

    # 3. Check Validity
    for i in range(n):
        if earliest[i] > latest[i] + 1e-6:
            return None

    # Calculate total travel time (sum of edges)
    total_len = 0.0
    for i in range(n - 1):
        total_len += network.get_travel_time(sequence[i].node_id, sequence[i+1].node_id)

    return RouteTopology(sequence, total_len, earliest, latest)


def extend_permutations(
    base_topologies: List[RouteTopology],
    new_stops: List[Stop],
    network: MockNetworkService
) -> List[RouteTopology]:
    """
    Inserts new_stops into all base_topologies.
    Assumes new_stops is [Pickup, Dropoff] for a single request.
    """
    valid_results = []
    
    pickup = new_stops[0]
    dropoff = new_stops[1]

    for base in base_topologies:
        base_stops = base.stops
        n = len(base_stops)
        
        # Insert Pickup at index i (0 to n)
        for i in range(n + 1):
            # Insert Dropoff at index j (i+1 to n+1) 
            # (must be after pickup)
            
            # Optimization: Check primitive constraints for Pickup at i?
            
            # Temporary list with Pickup inserted
            temp_stops_p = base_stops[:i] + [pickup] + base_stops[i:]
            
            # Now insert Dropoff
            for j in range(i + 1, n + 2):
                final_stops = temp_stops_p[:j] + [dropoff] + temp_stops_p[j:]
                
                # Check Validity
                # (Ideally we'd do incremental checks, but full scan is O(N) which is fine for small K)
                topo = validate_sequence(final_stops, network)
                if topo:
                    valid_results.append(topo)
                    
    return valid_results


class TestPermutations(unittest.TestCase):
    def setUp(self):
        self.network = MockNetworkService()
        
    def test_basic_two_requests(self):
        # A: (0 -> 10), Time: [100, 200]
        stop_a_p = Stop("A_P", 0, 100, 200, "pickup")
        stop_a_d = Stop("A_D", 10, 100, 300, "dropoff") # TT=10, so min arrival 110
        
        # B: (5 -> 15), Time: [105, 200]
        stop_b_p = Stop("B_P", 5, 105, 200, "pickup")
        stop_b_d = Stop("B_D", 15, 105, 300, "dropoff")
        
        # 1. Base case: A
        topo_a = validate_sequence([stop_a_p, stop_a_d], self.network)
        self.assertIsNotNone(topo_a)
        assert topo_a is not None
        
        # 2. Insert B
        results = extend_permutations([topo_a], [stop_b_p, stop_b_d], self.network)
        
        # Possible valid permutations:
        # A->B->A->B (0 -> 5 -> 10 -> 15) Path: 5+5+5=15. 
        #   Start 100. P_B arr 105. D_A arr 110. D_B arr 115. All valid.
        # B->A->... 
        # Check we found some
        self.assertTrue(len(results) > 0)
        
        # Check specific order: P_A, P_B, D_A, D_B
        found = False
        for res in results:
            ids = [s.id for s in res.stops]
            if ids == ["A_P", "B_P", "A_D", "B_D"]:
                found = True
                break
        self.assertTrue(found, "Should find interleaved sequence")

    def test_impossible_constraint(self):
        # A: (0 -> 100) TT=100. [0, 50]. Impossible alone.
        stop_a_p = Stop("A_P", 0, 0, 50, "pickup")
        stop_a_d = Stop("A_D", 100, 0, 50, "dropoff") 
        
        topo = validate_sequence([stop_a_p, stop_a_d], self.network)
        self.assertIsNone(topo)

if __name__ == '__main__':
    unittest.main()
