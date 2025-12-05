
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from src.utils.rtv_plot import plot_rtv_graph
from src.interfaces.graphs import RTVGraph
from src.interfaces.data_models import Request, VehicleAvailability, Passenger, Route, Stop, FeasibleRoute
from src.interfaces.data_models import TripId

def create_mock_data():
    # 1. Mock Road Graph
    G = nx.MultiDiGraph()
    G.graph["crs"] = "EPSG:4326"
    # Add nodes with x, y
    G.add_node(1, x=0.0, y=0.0)
    G.add_node(2, x=1.0, y=0.0)
    G.add_node(3, x=1.0, y=1.0)
    G.add_edge(1, 2)
    G.add_edge(2, 3)

    # 2. Mock Requests
    r1 = Request(id="r1", origin_node=1, destination_node=2, request_time=0.0, earliest_dropoff_time=100.0)
    r2 = Request(id="r2", origin_node=2, destination_node=3, request_time=0.0, earliest_dropoff_time=100.0)
    requests = [r1, r2]

    # 3. Mock Vehicles
    v1 = VehicleAvailability(
        vehicle_id="v1",
        capacity=4,
        available_at_node=1,
        available_from_time=0.0,
        passengers_onboard=frozenset(),
        remaining_stops_to_serve=()
    )
    vehicles = [v1]

    # 4. Mock RTV Graph
    rtv_graph = RTVGraph()
    
    # Trip with r1
    trip1 = frozenset({"r1"})
    stop1 = Stop(node=1, is_pickup=True, request_id="r1")
    stop2 = Stop(node=2, is_pickup=False, request_id="r1")
    route1 = Route(stops=(stop1, stop2), total_travel_time=10, total_travel_distance=100, total_delay_cost=0)
    feasible1 = FeasibleRoute(route=route1, cost=10)
    rtv_graph.add_trip_for_vehicle(trip1, "v1", feasible1)

    return rtv_graph, vehicles, requests, G

if __name__ == "__main__":
    print("Running reproduction script...")
    try:
        rtv_graph, vehicles, requests, G = create_mock_data()
        print("Mock data created.")
        
        # Mock osmnx.plot_graph to avoid actual plotting if needed, but we want to test it.
        # However, we need to handle plt.show() blocking.
        # We can patch plt.show to do nothing.
        
        import matplotlib.pyplot as plt
        plt.show = lambda block=False: print("plt.show() called")

        plot_rtv_graph(
            rtv_graph=rtv_graph,
            vehicle_avail_pool=vehicles,
            all_requests_in_pool=requests,
            manhattan_graph=G,
            title="Test Plot",
            output_dir=None
        )
        print("plot_rtv_graph finished successfully.")
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()
