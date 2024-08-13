import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from core.net import Net
from turtle import position

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 1
bandwidth_capacity = 1000 ###1000####
cpu_capacity = 100
cache_capacity = 100

class Luxembourg:
    def __init__(self) -> None:
        self.number_of_nodes = None
        self.processing_nodes = np.array([25, 8, 14, 28, 2, 5, 9, 23, 18, 6, 33, 34])
        
    def generate_substrate_network(self):
        latency = 1
  
        position ={0: (5000, 6000), 1: (2884, 6739), 2: (8081, 4302), 3: (8881, 4995), 4: (8956, 2900), 5: (9693, 4040), 6: (4351, 6749), 7: (5341, 7590), 8: (6053, 8420), 9: (6230, 5831), 10: (6446, 6969), 11: (6776, 7725), 12: (6785, 5631), 13: (7000, 8517), 14: (7034, 7203), 15: (7041, 6100), 16: (7019, 3541), 17: (7295, 7026), 18: (7234, 5147), 19: (7336, 5733), 20: (7477, 6206), 21: (7467, 5394), 22: (7605, 8238), 23: (7551, 5914), 24: (8266, 7723), 25: (8403, 6861), 26: (9639, 7772), 27: (9868, 9199), 28: (10138, 9166), 29: (10433, 9612), 30: (10652, 9036), 31: (1690, 8623), 32: (8839, 4462), 33: (1492, 7139), 34: (4500, 5500)}

        topology = [(0, 34), (14, 17), (25, 24), (8, 13), (8, 22), (14, 10), (14, 11), (28, 27), (28, 29), (28, 30), (2, 3), (2, 32), (5, 4), (9, 12), (23, 15), (23, 19), (23, 20), (18, 21), (6, 7), (2, 16), (25, 26), (33, 1), (33, 31), (34, 9), (9, 18), (18, 2), (2, 5), (5, 23), (23, 25), (25, 28), (28, 14), (14, 8), (8, 6), (6, 33), (33, 34)]

        substrate_network = Net()
        
        for edge in topology:
            if edge == topology[0]:
                substrate_network.init_link_latency(edge[0], edge[1], CLOUD_LATENCY)
            else:
                substrate_network.init_link_latency(edge[0], edge[1], latency)
            substrate_network.init_bandwidth_capacity(edge[0], edge[1], bandwidth_capacity)
        
        processing_nodes = self.processing_nodes
        #processing_nodes = np.array(np.arange(len(topology)))

        for node in range(0, len(list(position.keys()))):
            substrate_network.set_node_position(node,position[node])
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
    substrate_network = Luxembourg().generate_substrate_network()
    nx.draw(substrate_network, with_labels=True)
    plt.show()
