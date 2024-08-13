import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from turtle import position

from core.net import Net

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 1
bandwidth_capacity = 1000
cpu_capacity = 100
cache_capacity = 100

class Small_Luxembourg:
    def __init__(self) -> None:
        self.number_of_nodes = None
        self.processing_nodes = None
        
    def generate_substrate_network(self):
        latency = 1
  
        position ={0: (1492, 6892),1: (2884, 6892),2: (4351, 6892),3: (6446, 6892),4: (8403, 6500)}

        topology = [(0,1),(1,2),(2,3),(3,4)]

        substrate_network = Net()

        for edge in topology:
            if edge == topology[0]:
                substrate_network.init_link_latency(edge[0], edge[1], CLOUD_LATENCY)
            else:
                substrate_network.init_link_latency(edge[0], edge[1], latency)
            substrate_network.init_bandwidth_capacity(edge[0], edge[1], bandwidth_capacity)
        
        processing_nodes = np.array([1,2,3])
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
    substrate_network = Small_Luxembourg().generate_substrate_network()
    nx.draw(substrate_network, with_labels=True)
    plt.show()
