"""Network generation utilities."""

import networkx as nx


def generate_graph(kind, n, seed=None, **kwargs):
    """Generate a networkx graph by name."""
    if kind == "grid":
        cols = kwargs.pop("m", n)
        graph = nx.grid_2d_graph(n, cols)
        return nx.convert_node_labels_to_integers(graph)

    generators = {
        "erdos_renyi": nx.erdos_renyi_graph,
        "watts_strogatz": nx.watts_strogatz_graph,
        "barabasi_albert": nx.barabasi_albert_graph,
    }
    if kind not in generators:
        raise ValueError(f"Unknown graph kind: {kind}")
    return generators[kind](n, seed=seed, **kwargs)
