import networkx as nx
import matplotlib.pyplot as plt

from src.state_representation.manhatten_fast import load_graph


def label_components(G: nx.Graph | nx.DiGraph) -> dict:
    """Return {node -> component_id} using SCC for digraphs, CC for undirected."""
    if G.is_directed():
        comps = list(nx.strongly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    node2comp = {}
    for cid, comp in enumerate(comps):
        for n in comp:
            node2comp[n] = cid
    return node2comp


def plot_components(
    G: nx.Graph | nx.DiGraph, pos: dict | None = None, with_labels: bool = False
):
    """
    Plot the graph with nodes colored by (strongly) connected component.
    - If `pos` not given, uses spring_layout (or uses 'x','y' node attrs if present).
    - No explicit colors are set; matplotlib's default color cycle distinguishes components.
    """
    # positions
    if pos is None:
        # use node attributes if you have precomputed positions (e.g., from OSM)
        if (
            len(G) > 0
            and "x" in next(iter(G.nodes(data=True)))[1]
            and "y" in next(iter(G.nodes(data=True)))[1]
        ):
            pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
        else:
            pos = nx.spring_layout(G, seed=42)

    node2comp = label_components(G)

    # invert mapping: comp_id -> list of nodes
    comp2nodes: dict[int, list] = {}
    for n, cid in node2comp.items():
        comp2nodes.setdefault(cid, []).append(n)

    plt.figure(figsize=(8, 6))

    # draw edges once (light alpha so components pop)
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # draw each component's nodes; default color cycle will differentiate them
    for cid, nodelist in comp2nodes.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, label=f"comp {cid}")

    if with_labels and len(G) <= 200:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("(Strongly) Connected Components")
    # optional: show legend only if components are few
    if len(comp2nodes) <= 10:
        plt.legend(frameon=False)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# usage:
# plot_components(G)                      # auto layouts
# plot_components(G, pos=precomputed_pos) # if you have coordinates

if __name__ == "__main__":
    # Example usage with a small directed graph
    G = load_graph()  # replace with actual graph loading

    plot_components(G, with_labels=True)
