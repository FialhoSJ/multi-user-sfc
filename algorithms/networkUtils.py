import networkx as nx

def pre_get_single_source_minimum_latency_path(graph):
    """Pre-calculate the shortest paths for all nodes in the network based on latency."""
    single_source_minimum_latency_path = {}
    for node in graph.nodes():
        single_source_minimum_latency_path[node] = \
            nx.single_source_dijkstra(graph, source=node, cutoff=None, weight='latency')
    return single_source_minimum_latency_path

def add_node_to_graph(graph, node_id, node_attributes):
    """Add a node to the graph with specific attributes."""
    graph.add_node(node_id, **node_attributes)

def add_edge_to_graph(graph, node1, node2, bandwidth_capacity=500, latency=1):
    """Add an edge between two nodes in the graph with bandwidth and latency."""
    graph.add_edge(node1, node2, bandwidth_capacity=bandwidth_capacity, latency=latency)

def get_shortest_path_length(graph, source, target):
    """Get the shortest path length from the source node to the target node based on latency."""
    try:
        return nx.dijkstra_path_length(graph, source, target, weight='latency')
    except nx.NetworkXNoPath:
        return float('inf')

def get_shortest_path(graph, source, target):
    """Get the shortest path from the source node to the target node based on latency."""
    try:
        return nx.dijkstra_path(graph, source, target, weight='latency')
    except nx.NetworkXNoPath:
        return []

def get_link_latency(graph, node1, node2):
    """Get the latency of the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['latency']

def get_node_cpu_used(graph, node_id):
    """Get the CPU used by a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cpu_used']

def get_node_cpu_free(graph, node_id):
    """Get the free CPU capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cpu_capacity'] - graph.nodes[node_id]['cpu_used']

def get_node_cpu_capacity(graph, node_id):
    """Get the total CPU capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cpu_capacity']

def get_node_cache_used(graph, node_id):
    """Get the cache used by a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cache_used']

def get_node_cache_free(graph, node_id):
    """Get the free cache capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cache_capacity'] - graph.nodes[node_id]['cache_used']

def get_node_cache_capacity(graph, node_id):
    """Get the total cache capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cache_capacity']

def get_link_bandwidth_used(graph, node1, node2):
    """Get the bandwidth used by the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['bandwidth_used']

def get_link_bandwidth_free(graph, node1, node2):
    """Get the free bandwidth capacity of the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['bandwidth_capacity'] - graph.edges[node1, node2]['bandwidth_used']

def get_link_bandwidth_capacity(graph, node1, node2):
    """Get the total bandwidth capacity of the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['bandwidth_capacity']

