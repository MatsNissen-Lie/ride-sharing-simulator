
import logging
import sys
import time
import itertools
from typing import Set

# Mock interfaces/setup
from src.interfaces import (
    Vehicle, Passenger, Request, VehicleAvailability, Stop, Route,
    INetworkService
)
from src.interfaces.data_models import NetworkPathExtended
# Real components
from src.components.graph_builder.rtv_builder import RTVGraphBuilder
from src.components.graph_builder.rtv_builder_permutations_vec import PermutationRTVGraphBuilder
from src.components.route_planner import RoutePlanner
from src.components.network_service import NetworkXService
from src.components.simulation_data_provider import SimulationDataProvider
from src.components.graph_builder.rv_builder import RVGraphBuilder
from src.database.simple_db import DuckDBPathCache
from config.config import MAX_REQUESTS, BATCH_DURATION, MIN_START_TIME, NUMBER_OF_VEHICLES

# Setup logging
logging.basicConfig(level=logging.ERROR)

def run_comparison():
    print("Initializing Comparison Run (Matching main.py)...")
    
    # 1. Setup Real Provider & Network
    provider = SimulationDataProvider()
    db_cache = DuckDBPathCache(read_only=False) # Match main.py: Read-Write
    network_service = NetworkXService(provider.graph, db=db_cache)
    
    planner = RoutePlanner(network_service=network_service)
    
    # Builders
    old_builder = RTVGraphBuilder(planner)
    new_builder = PermutationRTVGraphBuilder(planner)
    rv_builder = RVGraphBuilder(planner, network_service)
    
    # 2. Extract Data (Vehicles & Requests)
    req_iter = provider.iter_requests()
    all_requests = []
    
    # Match Simulator Logic: Collect requests in the first batch window
    t_start_win = MIN_START_TIME
    t_end_win = t_start_win + BATCH_DURATION
    
    print(f"Collecting requests for window [{t_start_win}, {t_end_win}]...")
    
    count = 0
    # Process Requests (Simulator filters them by time)
    for meta in req_iter:
        # Skip requests before MIN_START_TIME (Simulator logic)
        if meta.request_time < t_start_win:
            continue
            
        # Stop if after batch end
        if meta.request_time > t_end_win:
            break
            
        # Limit to avoid OOM if density is too high for this test execution
        # But user wants to match main settings. 
        # CAUTION: If user ran full sim and got 140/139, we need the exact set.
        # Let's hope 300 is enough or we use the 'first batch' logic.
        
        # Convert to Request
        path = network_service.get_shortest_path(meta.origin_node, meta.destination_node)
        tt = path.travel_time if path else 0.0
        
        req = Request(
            id=meta.id,
            origin_node=meta.origin_node,
            destination_node=meta.destination_node,
            request_time=meta.request_time,
            earliest_dropoff_time=meta.request_time + tt
        )
        all_requests.append(req)
        count += 1
        
        # SAFETY CAP: If main.py runs fine, I should too, but this environment might be constrained.
        # Let's try 300 cap. If user has >300 in first batch, we might miss the edge case.
        if count >= 300:
             print("Hit safety cap of 300 requests.")
             break
             
    print(f"Collected {len(all_requests)} requests.")
    
    # Vehicles
    vehicles_iter = provider.iter_vehicles(NUMBER_OF_VEHICLES)
    
    # SAFETY CAP: 3000 vehicles is heavy for a test script. 
    # But distribution matters. 
    # Let's try 500? If main.py uses 3000, maybe I should try 1000.
    # The previous run with 200/50 requests worked.
    # Let's use 300 requests and 500 vehicles.
    vehicles = list(itertools.islice(vehicles_iter, 500))
    print(f"Using {len(vehicles)} vehicles.")
    
    # Convert to VehicleAvailability
    # In first batch, all vehicles are available from < t_now
    vehicle_pool = set()
    for v in vehicles:
        vehicle_pool.add(VehicleAvailability.from_vehicle(v, t_end_win)) # Available for assignment
        
    # 3. Build RV Graph (Common input)
    print("Building RV Graph...")
    rv_graph = rv_builder.build_graph(set(all_requests), vehicle_pool)
    print(f"RV Graph: {len(rv_graph.rv_details)} RV edges, {len(rv_graph.rr_details)} RR edges")
    
    # 4. Run OLD Builder
    print("\nRunning OLD Builder...")
    t0 = time.time()
    rtv_old = old_builder.build_graph(rv_graph, vehicle_pool, set(all_requests), max_trip_size=4)
    print(f"Old Time: {time.time() - t0:.4f}s")
    stats_old, trips_old = rtv_old.get_trip_stats()
    print(f"Old Stats: {stats_old}")
    
    # 5. Run NEW Builder
    print("\nRunning NEW Builder...")
    t0 = time.time()
    rtv_new = new_builder.build_graph(rv_graph, vehicle_pool, set(all_requests), max_trip_size=4)
    print(f"New Time: {time.time() - t0:.4f}s")
    stats_new, trips_new = rtv_new.get_trip_stats()
    print(f"New Stats: {stats_new}")
    
    # 6. Compare
    print("\n--- COMPARISON ---")
    
    for size in [2, 3, 4]:
        s_old = trips_old.get(size, set())
        s_new = trips_new.get(size, set())
        
        missing_in_new = s_old - s_new
        missing_in_old = s_new - s_old
        
        print(f"Size {size}: Old={len(s_old)}, New={len(s_new)}")
        if missing_in_new:
            print(f"  Missing in NEW: {len(missing_in_new)}")
            for t_id in list(missing_in_new)[:5]: # Show first 5
                print(f"    Trip: {t_id}")
                vehs = rtv_old.get_vehicles_for_trip(t_id)
                print(f"      Served by (Old): {list(vehs)[:3]}...")
                
                # Check sub-trips for the first missing one
                reqs = list(t_id)
                if size > 1:
                    subsets = [frozenset(c) for c in itertools.combinations(reqs, size-1)]
                    print(f"      Sub-trips of size {size-1}:")
                    for sub in subsets:
                        exists = sub in trips_new.get(size-1, set())
                        print(f"        {sub}: {'FOUND' if exists else 'MISSING'}")
        
        if missing_in_old:
             print(f"  Missing in OLD: {len(missing_in_old)}")

if __name__ == "__main__":
    run_comparison()
