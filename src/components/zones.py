"""Graph nodes to TLC taxi zones."""

import geopandas as gpd
import networkx as nx
import numpy as np

from config.paths import TAXI_ZONES_PATH


class ZoneIndex:
    """
    Zone of every graph node: the Manhattan taxi zone containing it, or the
    nearest one for the few nodes just outside the zone polygons.
    """

    def __init__(self, graph: nx.MultiDiGraph, zones_path=TAXI_ZONES_PATH):
        zones = gpd.read_file(zones_path)
        zones = zones.loc[zones["borough"] == "Manhattan", ["LocationID", "geometry"]]
        node_ids = np.fromiter(graph.nodes, dtype=np.int64)
        points = gpd.GeoDataFrame(
            {"node": node_ids},
            geometry=gpd.points_from_xy(
                [graph.nodes[n]["x"] for n in node_ids], [graph.nodes[n]["y"] for n in node_ids]
            ),
            crs=graph.graph.get("crs", "epsg:4326"),
        ).to_crs(zones.crs)
        joined = gpd.sjoin_nearest(points, zones, how="left").drop_duplicates("node")

        self._zone_of_node = np.full(int(node_ids.max()) + 1, -1, dtype=np.int64)
        self._zone_of_node[joined["node"].to_numpy()] = joined["LocationID"].to_numpy()
        self.zone_ids = np.unique(joined["LocationID"].to_numpy())

        # Representative node per zone: the zone's node closest to a point
        # inside the zone (used as a rebalancing destination).
        inside = zones.set_index("LocationID").geometry.representative_point()
        self.center_node: dict[int, int] = {}
        for zone_id in self.zone_ids:
            members = joined[joined["LocationID"] == zone_id]
            d = members.geometry.distance(inside.loc[zone_id])
            self.center_node[int(zone_id)] = int(members["node"].iloc[int(np.argmin(d.to_numpy()))])

    def zone(self, node: int) -> int:
        return int(self._zone_of_node[node])

    def zones_of(self, nodes) -> np.ndarray:
        return self._zone_of_node[np.asarray(nodes, dtype=np.int64)]
