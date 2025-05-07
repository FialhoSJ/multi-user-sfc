import numpy as np
from core.net_v2 import Net2
from topology.base.topology_base import TopologyBase

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 1
BANDWIDTH_CAPACITY = 5000.0
CPU_CAPACITY = 100.0
CACHE_CAPACITY = 100.0
RELIABILITY_RANGE = [0.95,0.99]

class LuxembourgV2(TopologyBase):
    def __init__(self) -> None:
        self.latency = 1
        self.bandwidth_capacity = BANDWIDTH_CAPACITY
        self.cpu_capacity = CPU_CAPACITY
        self.cache_capacity = CACHE_CAPACITY
        self.number_of_nodes = 35
        self.nodes = np.arange(1, 35)
        self.edge_computing_servers = np.array([25, 8, 14, 28, 2, 5, 9, 23, 18, 6, 33, 34])

        self.positions = {0: (5000, 6000), 1: (2884, 6739), 2: (8081, 4302), 3: (8881, 4995), 4: (8956, 2900),
                    5: (9693, 4040), 6: (4351, 6749), 7: (5341, 7590), 8: (6053, 8420), 9: (6230, 5831),
                    10: (6446, 6969), 11: (6776, 7725), 12: (6785, 5631), 13: (7000, 8517), 14: (7034, 7203),
                    15: (7041, 6100), 16: (7019, 3541), 17: (7295, 7026), 18: (7234, 5147), 19: (7336, 5733),
                    20: (7477, 6206), 21: (7467, 5394), 22: (7605, 8238), 23: (7551, 5914), 24: (8266, 7723),
                    25: (8403, 6861), 26: (9639, 7772), 27: (9868, 9199), 28: (10138, 9166), 29: (10433, 9612),
                    30: (10652, 9036), 31: (1690, 8623), 32: (8839, 4462), 33: (1492, 7139), 34: (4500, 5500)}

        self.topology = [(0, 34), (14, 17), (25, 24), (8, 13), (8, 22), (14, 10), (14, 11), (28, 27), (28, 29),
                    (28, 30), (2, 3), (2, 32), (5, 4), (9, 12), (23, 15), (23, 19), (23, 20), (18, 21), (6, 7),
                    (2, 16), (25, 26), (33, 1), (33, 31), (34, 9), (9, 18), (18, 2), (2, 5), (5, 23), (23, 25),
                    (25, 28), (28, 14), (14, 8), (8, 6), (6, 33), (33, 34)]

    
    def generate_substrate_network(self):
        net = Net2()

        # Adiciona os nós
        net.add_node(0, node_type='server', cpu_capacity=0.0)
        for node in self.nodes:
            if node in self.edge_computing_servers:
                net.add_node(node, node_type='server', cpu_capacity=self.cpu_capacity,cache_capacity=self.cache_capacity)
            else:
                net.add_node(node, node_type='router')

        # Adiciona as arestas com latência baseada na distância euclidiana
        for u, v in self.topology:
            net.add_edge(u, v, bandwidth_capacity=self.bandwidth_capacity, latency=1)

        net.total_cpu_capacity = len(self.edge_computing_servers) * CPU_CAPACITY
        net.total_cache_capacity = len(self.edge_computing_servers) * CACHE_CAPACITY
        net.total_bandwidth_capacity = len(self.topology)* BANDWIDTH_CAPACITY
        net.pre_get_single_source_minimum_latency_path()

        return net
    
    def get_topology_info(self):
        """Retorna informações da topologia."""
        nodes = list(np.arange(1, self.number_of_nodes))
        edges = self.topology  # Lista de arestas
        edge_computing_servers = self.edge_computing_servers.tolist()  # Nós de processamento

        return {
            "nodes": nodes,
            "edges": edges,
            "ec_servers": edge_computing_servers,
            "routers":[node for node in nodes if node not in edge_computing_servers and node != 0]
        }