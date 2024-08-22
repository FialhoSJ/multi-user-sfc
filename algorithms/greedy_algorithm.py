"""
Consideration:
Algorithm should not do any modification on substrate network

It should only use network information and sfc information to
solve and give out a mapping and route info, that

route info :=
{
    src:  [1, 2, 3],
    vnf1: [3, 4, 5],
    vnf2: [5, 6, 7],
    vnf3: [7, 8 ,9],
    dst:  []
}

"""
import logging
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
from config import ROOT_PATH
ch = logging.FileHandler(ROOT_PATH + './logs/GreedyAlgorithm.log')
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class GreedyAlgorithm():
    '''Greedy Algorithm.
    This algorithm starts from the substrate network node which hosts src of an SFC, checks its neighbor nodes,
    finds the neighbor node with a shortest latency edge, and use the node to host the vnf.
    The algorithm greedily finds all nodes for hosting vnf.
    Finally, the algorithm finds a shortest path from the substrate node who hosts the last vnf in the SFC
    to the substrate node who hosts dst of the SFC.

    Deploy VNF one by one, with a shortest path from the node to the previous substrate node.
    '''
    def __init__(self):
        self.name = "Greedy Algorithm"
        self.substrate_network = None
        self.sfc = None
        self.node_info = None
        self.route_info = None
        self.latency = None


    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = None
        self.route_info = None
        self.latency = None

    def install_substrate_network(self, substrate_network):
        self.substrate_network = substrate_network
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        return self.sfc

    def start_algorithm(self):
        substrate_network = self.substrate_network
        sfc = self.sfc
        logger.info("Start algorithm")
        if self.algorithm(substrate_network, sfc):
            logger.info("End algorithm, success")
            return True
        logger.info("End algorithm, failed")
        return False

    def get_latency(self):
        return self.latency
    
    def get_route_info(self):
        return self.route_info

    def algorithm(self, substrate_network, sfc):
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advance
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        route_info = {}
        bandwidth_usage_info = {}

        latency = 0
        used_node = [dst_substrate_node, src_substrate_node]

        number_of_vnfs = sfc.get_number_of_vnfs()
        current_vnf = dst_vnf
        current_substrate_node = dst_substrate_node

        for i in range(number_of_vnfs - 1, -1, -1):
            edges = substrate_network.edges(current_substrate_node)
            prev_vnf = current_vnf.get_previous_vnf()  # Inverte a ordem, indo para a VNF anterior

            cpu_request = sfc.get_vnf_cpu_request(prev_vnf)
            cache_request = sfc.get_vnf_cache_request(prev_vnf)

            bandwidth_request = sfc.get_link_bandwidth_request(prev_vnf.id, current_vnf.id)

            min_latency = None
            node = None

            for e in edges:
                cpu_available = substrate_network.get_node_cpu_free(e[1])
                cache_available = substrate_network.get_node_cache_free(e[1])

                if cpu_request > cpu_available:
                    logger.debug("node %s has not %s cpu", e[1], cpu_request)
                    continue
                if cache_request > cache_available:
                    logger.debug("node %s has not %s cache", e[1], cache_request)
                    continue

                bandwidth_available = substrate_network.get_link_bandwidth_free(e[0], e[1])
                if bandwidth_request > bandwidth_available:
                    logger.debug("edge has not sufficient bandwidth, check next edge")
                    continue

                edge_latency = substrate_network.get_link_latency(e[0], e[1])
                if min_latency is None or edge_latency < min_latency:
                    min_latency = edge_latency
                    node = e[1]

            if node is not None:
                route_info[prev_vnf.id] = [node, current_substrate_node]
                used_node.append(node)
                latency += min_latency

                edge_key = frozenset((node, current_substrate_node))

                residual_bandwidth = None
                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = substrate_network.get_link_bandwidth_free(node, current_substrate_node) - bandwidth_request
                if residual_bandwidth < 0:
                    logger.warning('Bandwidth resources is not sufficient')
                    self.route_info = {}
                    self.latency = None
                    return False
                bandwidth_usage_info[edge_key] = residual_bandwidth

            else:
                logger.debug("node is not existing")
                self.route_info = {}
                self.latency = None
                return False

            current_substrate_node = node
            current_vnf = prev_vnf

        try:
            path = substrate_network.get_shortest_path(src_substrate_node, node)
            path_latency = substrate_network.get_shortest_path_length(src_substrate_node, node)
        except:
            logger.warning('have no path between src and first vnf: %s - %s', src_substrate_node, node)
            self.route_info = {}
            self.latency = None
            return False

        #bandwidth_request = sfc.get_link_bandwidth_request('src', current_vnf.id)
        bandwidth_available = substrate_network.get_minimum_free_bandwidth(path)
        bandwidth_request = 0 

        length = len(path)
        for i in range(0, length - 1):
            edge_key = frozenset((path[i], path[i + 1]))
            residual_bandwidth = None
            if edge_key in bandwidth_usage_info:
                residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
            else:
                residual_bandwidth = substrate_network.get_link_bandwidth_free(path[i], path[i + 1]) - bandwidth_request
            if residual_bandwidth < 0:
                logger.warning('Bandwidth resources is not sufficient')
                self.latency = None
                self.route_info = {}
                return False
            bandwidth_usage_info[edge_key] = residual_bandwidth

        if bandwidth_request > bandwidth_available:
            self.latency = None
            self.route_info = {}
            return False

        route_info['src'] = path
        route_info['dst'] = []
        latency += path_latency

        self.route_info = route_info
        self.latency = latency

        path = self.route_info['src']
        for i in range(len(path) - 1):
            edge_latency = self.substrate_network.get_link_latency(path[i], path[i + 1])
            self.latency = self.latency - edge_latency

        if len(self.route_info.keys()) != 6:
            self.latency = None
            self.route_info = {}
            return False
        if self.latency > sfc.get_latency_request():
            self.route_info = {}
            return False

        return True
