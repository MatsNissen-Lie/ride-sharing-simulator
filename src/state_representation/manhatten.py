import osmnx as ox
import networkx as nx
from datetime import datetime


FALLBACK_SPEED_KPH = 20.0
FALLBACK_SPEED_MPS = FALLBACK_SPEED_KPH * 1000 / 3600


# Boeing, G. (2025). Modeling and Analyzing Urban Networks and Amenities with OSMnx. Geographical Analysis 57 (4), 567-577. doi:10.1111/gean.70009


# Define the place name for Manhattan
place_name = "Manhattan, New York City, New York, USA"

# how much time does it take to laod?
start_time = datetime.now()
# Retrieve the drivable road network for Manhattan
G = ox.graph_from_place(place_name, network_type="drive")

# Ensure each edge has a realistic travel time in seconds. NetworkX treats
# missing weights as 1 by default (see /networkx/networkx doc/tutorial.md),
# which explained the unreasonably short trip reported earlier. We seed times
# from OSMnx's helpers and fall back to 20 km/h where data is missing.
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

missing_travel_time_edges = 0
for _, _, data in G.edges(data=True):
    travel_time = data.get("travel_time")
    length_m = data.get("length")
    if (travel_time is None or travel_time <= 0) and length_m:
        data["travel_time"] = length_m / FALLBACK_SPEED_MPS
        data.setdefault("speed_kph", FALLBACK_SPEED_KPH)
        missing_travel_time_edges += 1

if missing_travel_time_edges:
    print(f"Applied fallback speed to {missing_travel_time_edges} edges")

# Output basic information about the graph
num_nodes = len(G.nodes)
num_edges = len(G.edges)
print(f"Retrieved Manhattan road network with {num_nodes} nodes and {num_edges} edges")

print(
    f"Seconds taken to load the graph: {(datetime.now() - start_time).total_seconds()} seconds"
)


# Define origin and destination addresses
origin_address = "Times Square, Manhattan, NY, USA"
dest_address = "Wall Street, Manhattan, NY, USA"
dest_address = "Trump Tower, Manhattan, NY, USA"

# Geocode the addresses to latitude-longitude using OSMnx
orig_lat, orig_lon = ox.geocode(origin_address)  # returns (lat, lon)
dest_lat, dest_lon = ox.geocode(dest_address)

# Find the nearest graph nodes to these coordinates
orig_node = ox.nearest_nodes(G, X=orig_lon, Y=orig_lat)
dest_node = ox.nearest_nodes(G, X=dest_lon, Y=dest_lat)

# Compute the shortest path (min travel time) between the nodes
route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
route_time_seconds = nx.shortest_path_length(
    G, orig_node, dest_node, weight="travel_time"
)
route_distance_meters = nx.shortest_path_length(
    G, orig_node, dest_node, weight="length"
)
if route_time_seconds > 0:
    avg_speed_kph = (route_distance_meters / 1000) / (route_time_seconds / 3600)
else:
    avg_speed_kph = 0.0

print(f"Nearest graph node to origin: {orig_node}")
print(f"Nearest graph node to destination: {dest_node}")
print(
    f"Shortest route has {len(route_nodes)-1} road segments and covers {route_distance_meters:.1f} meters"
)
print(
    f"Estimated travel time for shortest route: {route_time_seconds:.1f} seconds ({route_time_seconds/60:.1f} minutes)"
)
print(f"Average speed along route: {avg_speed_kph:.1f} km/h")
# Visualize the route on the map
fig, ax = ox.plot_graph_route(
    G, route_nodes, route_linewidth=4, node_size=0, bgcolor="k"
)
fig.savefig("manhattan_route.png")
print("Route map saved as 'manhattan_route.png'")
