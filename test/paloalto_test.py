from turtle import position
import networkx as nx
import matplotlib.pyplot as plt
from core.net import Net
import numpy as np

LIGHT_SPEED = 3 * 10e8
CLOUD_LATENCY = 1
bandwidth_capacity = 1000
cpu_capacity = 100
cache_capacity = 100
def generate_substrate_network():

    #latency = 500 / LIGHT_SPEED
    latency = 1
    #reducing from the original topology 36 to 15 nodes
    topology = [(0,1), (1,2), (1,3), (2,4), (2,5), (3,6), (3,7), (4,8), (4,9), (5,10), (5,11), (6,12), (6,13), (7,14), (7,15)]
  
    topology = [(0,1), #Cloud 
                (1,2), (1,3), (2,4), (2,5), (3,6), (3,7), 
                (4,8), (4,9), (5,10), (5,11), (6,12), (6,13), (7,14), (7,15)
                ]
                
    position = {0:(8000,8000),
                1:(7278,1622), 2:(6400,2341), 3:(5000,3364), 4:(4440,3825), 5:(3932,4212), 6:(3400,4620), 7:(2290,5440),
                8:(2190,5760), 9:(3287,6056),10:(3868,5622),11:(4476,5613),12:(4845,4811),13:(5712,4120),14:(6700,3375),
                15:(5975,1833)
                }
    #mean = 0.5 
    #std = 2 
    #latency = np.random.normal(mean, std, size=len(topology))
   
    substrate_network = Net()
    for i, edge in enumerate(topology):
        if edge == topology[0]:
            substrate_network.init_link_latency(edge[0], edge[1], CLOUD_LATENCY)
        else:
            substrate_network.init_link_latency(edge[0], edge[1], latency)
            #substrate_network.init_link_latency(edge[0] - 1, edge[1] - 1, abs(latency[i]))
        substrate_network.init_bandwidth_capacity(edge[0], edge[1], bandwidth_capacity)

    processing_nodes = np.array([4,14,16,18,19,20,21,25,26,32,33,35,36])
    print(processing_nodes)
    processing_nodes = np.array(np.arange(len(topology)))
    #processing_nodes = np.array([18,19,20,25,32,33,35,36]) - 1
    for node in range(0,16):
        #print(node)
        
        substrate_network.init_node_cpu_capacity(node, cpu_capacity)
        substrate_network.init_node_cache_capacity(node, cache_capacity)
        #print(position[node])
        substrate_network.set_node_position(node,position[node])
        
        #if node in processing_nodes:
        #    substrate_network.init_node_cpu_capacity(node, cpu_capacity)
        #    substrate_network.init_node_cache_capacity(node, cache_capacity)
        #else:
        #    substrate_network.init_node_cpu_capacity(node, 0)
        #    substrate_network.init_node_cache_capacity(node,0)

    substrate_network.pre_get_single_source_minimum_latency_path()
    substrate_network.update()
    return substrate_network
if __name__ == '__main__':
    substrate_network = generate_substrate_network()
    #for node in range(1,15):
        #print(substrate_network.get_node_cpu_free(node))
        #print(substrate_network.get_node_position(node))
    a = substrate_network.all_shortest_paths()
    nx.draw(substrate_network, with_labels=True)  # networkx draw()
    #plt.draw()  # pyplot draw()
    plt.show()
