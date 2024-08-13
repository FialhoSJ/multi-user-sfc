import networkx as nx
import matplotlib.pyplot as plt
from core.net import Net
import numpy as np

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 0
bandwidth_capacity = 1000
cpu_capacity = 100
cache_capacity = 100
class SantaMonica():
    def generate_substrate_network(self):

        #latency = 500 / LIGHT_SPEED
        latency = 1

        topology = [(1, 2), (1, 3), (4, 2), (4, 5), (20,4), (5,20),(19,20),(21,20),(11,21), 
            (11,12),(12,13),(14,13), (15,14), (14,29), (35,15),(29,35), (24,35), (24,23),(33,23),(33,30), 
            (30,33), (31,30),(16,31), (16,32), (34,32), (32,26),(33,32),(32,26), (25,26), 
            (25,16),(17,16),(17,18), (10,18), (18,9), (6,9),(9,19), (19,27), (25,27),(8,3), (8,7), (7,10), 
            (22, 28),(21,22),(28,35), (36,32)]

        substrate_network = Net()
        for edge in topology:
            if edge == topology[-1]:
                substrate_network.init_link_latency(edge[0] - 1, edge[1] - 1, CLOUD_LATENCY)
            else:
                substrate_network.init_link_latency(edge[0] - 1, edge[1] - 1, latency)
            substrate_network.init_bandwidth_capacity(edge[0] - 1, edge[1] - 1, bandwidth_capacity)

        processing_nodes = np.array([4,14,16,18,19,20,21,25,26,32,33,35,36]) - 1
        #processing_nodes = np.array([18,19,20,25,32,33,35,36]) - 1
        for node in range(36):
            if node in processing_nodes:
                substrate_network.init_node_cpu_capacity(node, cpu_capacity)
                substrate_network.init_node_cache_capacity(node, cache_capacity)
            else:
                substrate_network.init_node_cpu_capacity(node, 0)
                substrate_network.init_node_cache_capacity(node,0)

        substrate_network.pre_get_single_source_minimum_latency_path()
        substrate_network.update()
        return substrate_network
        
if __name__ == '__main__':
    substrate_network = SantaMonica().generate_substrate_network()
    nx.draw(substrate_network, with_labels=True)  # networkx draw()
    plt.draw()  # pyplot draw()
    plt.show()