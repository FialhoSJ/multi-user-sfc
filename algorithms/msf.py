import copy
import logging
from config import ROOT_PATH
from algorithms.networkUtils import get_link_bandwidth_free,pre_get_single_source_minimum_latency_path, get_link_latency

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
import os
ch = logging.FileHandler(os.path.join(ROOT_PATH, 'logs', 'DynamicProgrammingAlgorithm.log'))
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')
class MSF():
    def __init__(self):
        self.name = "msf"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.graph = None
        self.forbidden_matches = {}
    def clear_all(self):
        #logger.debug('clear all')
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.graph = None
        self.latency = None
   
    def install_substrate_network(self, substrate_network,sfc_list,shareable_sfs=[]):
        self.substrate_network = substrate_network
        self.graph = copy.deepcopy(substrate_network.graph)
        self.add_mobile_user_to_graph(substrate_network,sfc_list)
        self.single_source_minimum_latency_path = pre_get_single_source_minimum_latency_path(self.graph)
        return self.substrate_network

    def add_mobile_user_to_graph(self,substrate_network,sfc_list):
        mobile_device_id = sfc_list[0].dst_node
        closer_router    = sfc_list[0].closer_router

        # Os recursos do Mobile Device devem estar disponíveis somente para sua SFC
        md_info =  substrate_network.md_graph._node[mobile_device_id]
        self.graph.add_node(mobile_device_id,type='mobile_device',
                                cpu_capacity=md_info['cpu_capacity'],
                                cache_capacity=md_info['cache_capacity'],
                                cpu_used=md_info['cpu_used'],
                                cache_used=md_info['cache_used'],
                                position=md_info['position'],
                                services=md_info['services'])
        router = self.graph._node[closer_router]
        wireless_free = router['w_channel_capacity'] - router['w_channel_used']
        
        # TODO Permitir que o próprio algoritmo escolha o roteador
        # TODO calcular a latência do sinal
        signal_latency = 1
        self.graph.add_edge(mobile_device_id, closer_router, bandwidth_capacity=wireless_free, bandwidth_used=0.00 , latency=signal_latency, services_in_transit={})

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        src_vnf = self.sfc.get_src_vnf()
        src_substrate_node = self.sfc.get_substrate_node(src_vnf)
        dst_vnf = self.sfc.get_dst_vnf()
        dst_substrate_node = self.sfc.get_substrate_node(dst_vnf)

        for node in self.graph.nodes():
            self.node_info[node] = {}
            for vnf_id, vnf in list(sfc.vnfs.items()):
                # Not include src and dst.
                self.node_info[node][vnf_id] = {}
                self.node_info[node][vnf_id]['flag'] = False  # whether vnf/id can be placed on node
                self.node_info[node][vnf_id]['latency'] = float('inf')
                self.node_info[node][vnf_id]['path'] = []
                self.node_info[node][vnf_id]['src_path'] = []
                self.node_info[node][vnf_id]['previous_substrate_node'] = None
                self.node_info[node][vnf_id]['current_substrate_nodes'] = []    # The meta information
                                                                                # in which is a set of substrate node
                                                                                # has been assigned to VNFs in order
                self.node_info[node][vnf_id]['bandwidth_usage_info'] = {}

            self.node_info[node][src_vnf.id] = {}
            self.node_info[node][src_vnf.id]['flag'] = False  # src cannot be placed on the node except src node
            self.node_info[node][dst_vnf.id] = {}

        self.node_info[src_substrate_node][src_vnf.id]['flag'] = True # src can be placed on the src node
        self.node_info[src_substrate_node][src_vnf.id]['latency'] = 0
        self.node_info[src_substrate_node][src_vnf.id]['src_path'] = []
        self.node_info[src_substrate_node][src_vnf.id]['path'] = []
        self.node_info[src_substrate_node][src_vnf.id]['current_substrate_nodes'] = [src_substrate_node]
        self.node_info[dst_substrate_node][dst_vnf.id]['flag'] = False
        self.node_info[dst_substrate_node][dst_vnf.id]['latency'] = float('inf')
        self.node_info[dst_substrate_node][dst_vnf.id]['src_path'] = []
        self.node_info[dst_substrate_node][dst_vnf.id]['path'] = []
        self.node_info[dst_substrate_node][dst_vnf.id]['current_substrate_nodes'] = []
        self.node_info[src_substrate_node][src_vnf.id]['bandwidth_usage_info'] = {}
        self.node_info[dst_substrate_node][dst_vnf.id]['bandwidth_usage_info'] = {}

        return self.sfc

    def submit_solution(self):
        def allocate_microservice(node_id, service_id, cpu_required, cache_required):
            node = self.graph.nodes[node_id]

            # Verifica se há recursos disponíveis
            if node['cpu_used'] + cpu_required > node['cpu_capacity']:
                raise ValueError(f"CPU excedida no nó {node_id} para serviço {service_id}")
            if node['cache_used'] + cache_required > node['cpu_capacity']:
                raise ValueError(f"Cache excedido no nó {node_id} para serviço {service_id}")

            if service_id in node['services']:
                node['services'][service_id]['copys'] += 1  # Serviço já instanciado
                if not self.is_shareable(service_id):  # Se não for compartilhável
                    node['cpu_used'] += cpu_required
                    node['cache_used'] += cache_required
            else:
                node['services'][service_id] = {'cpu': cpu_required, 'cache': cache_required, 'copys': 1}
                node['cpu_used'] += cpu_required
                node['cache_used'] += cache_required

        def allocate_bandwidth(node1, node2, bw_required, ms_name):
            edge = self.graph.edges[node1, node2]

            # Verifica se há banda disponível
            if edge['bandwidth_used'] + bw_required > edge['bandwidth_capacity']:
                raise ValueError(f"Banda excedida entre os nós {node1} e {node2} para serviço {ms_name}")

            if ms_name in edge['services_in_transit']:
                edge['services_in_transit'][ms_name]['copys'] += 1
                edge['bandwidth_used'] += bw_required
            else:
                edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bw_required}
                edge['bandwidth_used'] += bw_required

        for ms_name, path in self.route_info.items():
            if ms_name in ['src', 'dst']:
                continue

            vnf = self.sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            cpu_req = vnf.get_cpu_request()
            cache_req = vnf.get_cache_request()
            allocate_microservice(node_allocated, ms_name, cpu_req, cache_req)

            bw_req = vnf.get_outcome_interface_bandwidth()

            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    allocate_bandwidth(u, v, bw_req, ms_name)

    def is_shareable(self,service_name):
        # TODO Mudar para a informação de compartilháveis estar em uma variável separável.
        #if self.shareable_node:
        if True:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False

    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def check_solution(self):
        if not isinstance(self.latency, (int, float)) or self.latency < 0 or self.latency > self.sfc.get_latency_request() or not self.route_info:
            return False
        if len(list(self.route_info.keys()))!=6:
            return False
        return True

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def start_algorithm(self):#,is_backup):
        #substrate_network = self.substrate_network
        sfc = self.sfc
        self.algorithm(sfc)
        is_success = self.check_solution()
        if is_success:
            try:
                self.submit_solution()
                logger.info("Finished algorithm, success")
                return True  
            except:
                self.handle_failure() 
                return False
        else:
            self.handle_failure() 
            logger.info("End algorithm, failed")
            return False

    def algorithm(self,sfc):
        nodes = self.graph.nodes()
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)

        self.src_substrate_node = src_substrate_node
        self.dst_substrate_node = dst_substrate_node
        
        (node_latency, node_path) = self.single_source_minimum_latency_path[dst_substrate_node] # Get single source path from substrate node to all other substrate node

            
        vnf1 = src_vnf.get_next_vnf()
        self._dp(src_substrate_node, vnf1)

        vnf = vnf1.get_next_vnf()
        while vnf.id != dst_vnf.id:
            for node in nodes:
                self._dp(node, vnf)
            vnf = vnf.get_next_vnf()

        # For dst:

        # here node in latency and path results is the node host previous vnf
        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        previous_vnf_id = previous_vnf.id

        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, dst_vnf.id)

        for node, latency in list(node_latency.items()):
            if node == dst_substrate_node or node == src_substrate_node:
                # if node is ingress or egress, continue
                continue

            # Check bandwidth resources
            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[node][previous_vnf_id]['bandwidth_usage_info'])
            path = node_path[node]
            length = len(path)
            for i in range(0, length - 1):
                edge_key = frozenset((path[i], path[i + 1]))
                residual_bandwidth = None
                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    residual_bandwidth = get_link_bandwidth_free(self.graph,path[i], path[i + 1]) - bandwidth_request
                if residual_bandwidth < 0:
                    #logger.warning('Bandwidth resources is not sufficient to dst')
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth
            if not is_bandwidth_sufficient:
                # check next path
                continue
            _latency = self.node_info[node][previous_vnf_id]['latency']
            if not self.node_info[dst_substrate_node][dst_vnf.id]['latency'] \
                or _latency + latency < self.node_info[dst_substrate_node][dst_vnf.id]['latency']:
                self.node_info[dst_substrate_node][dst_vnf.id]['latency'] = _latency + latency
                self.node_info[dst_substrate_node][dst_vnf.id]['path'] = node_path[node]
                self.node_info[dst_substrate_node][dst_vnf.id]['path'].reverse()
                self.node_info[dst_substrate_node][dst_vnf.id]['current_substrate_nodes'] = self.node_info[node][previous_vnf_id]['current_substrate_nodes'][:]
                self.node_info[dst_substrate_node][dst_vnf.id]['current_substrate_nodes'].append(dst_substrate_node)
                self.node_info[dst_substrate_node][dst_vnf.id]['src_path'] = self.node_info[node][previous_vnf_id]['src_path'][:] + self.node_info[dst_substrate_node]['dst']['path'][:]
                self.node_info[dst_substrate_node][dst_vnf.id]['flag'] = True

        if self.node_info[dst_substrate_node][dst_vnf.id]['flag']:
            # There is a solution
            # Backtracking
            # Start from dst to backtracking to src
            previous_vnf = dst_vnf
            previous_substrate_node = dst_substrate_node
            #print("backtrack pvs node:", previous_substrate_node)
            while True:
                path = self.node_info[previous_substrate_node][previous_vnf.id]['path']
                if not path:
                    break
                previous_substrate_node = path[0]
                previous_vnf = sfc.get_previous_vnf(previous_vnf)
                if previous_vnf:
                    self.route_info[previous_vnf.id] = path
                else:
                    break
            self.route_info[dst_vnf.id] = []
            self.latency = self.node_info[dst_substrate_node][dst_vnf.id]['latency']
            prev_vnf_node = self.route_info[previous_vnf.id][0]
            # remove latency from dst to previous vnf
            #self.latency_minus_dst = self.latency - len(self.route_info[previous_vnf.id])
            if 'src' not in self.route_info.keys():
                return True
            path = self.route_info['src']
            for i in range(len(path) - 1):
                edge_latency = get_link_latency(self.graph,path[i], path[i + 1])
                self.latency = self.latency - edge_latency
            #TODO ajustar o MSF e o MusFICo para que eles lidem melhor com a queda de servidores e não deem latencia negativa
            
            if self.latency > sfc.get_latency_request() or self.latency < 0: #
                self.route_info = {}
                self.latency = None
                return False
            
            if len(list(self.route_info.keys()))!=6: # Não instanciou todas
                self.route_info = False
                self.latency = None
                return False
            #print("Deu certo: ",self.route_info)
            return True
        else:
            #print("falha: ",self.route_info)
            return False

    def _dp(self, substrate_node, vnf):
        """
        Start from substrate node substrate_node, calculate all paths and latency from substrate_node to other nodes N.
        update information in nodes N for vnf, if latency is minimum. 
        """
        # Get precedent of the vnf
        sfc = self.sfc
        previous_vnf = sfc.get_previous_vnf(vnf)
        previous_vnf_id = previous_vnf.id
        #if not self.node_info[substrate_node][previous_vnf_id]['flag']:
        #    # This substrate node cannot host precedent vnf, thus, no need to exam further.
        #    return False
        
        vnf_id = vnf.id
        # Get single source path from substrate node to all other substrate node
        (node_latency, node_path) = self.single_source_minimum_latency_path[substrate_node]
        
        _latency = self.node_info[substrate_node][previous_vnf_id]['latency']

        cpu_request = sfc.get_vnf_cpu_request(vnf)
        bandwidth_request = sfc.get_link_bandwidth_request(previous_vnf_id, vnf_id)
        cache_request = sfc.get_vnf_cache_request(vnf)

        for node, latency in list(node_latency.items()):
            if latency > 3:
                continue
            
            forbidden = False
            for vnf_f,node_f in self.forbidden_matches.items():
                if node_f == node and vnf_f == vnf_id:
                    forbidden =True
            if forbidden:
                continue

            if node == substrate_node:
               # Cannot use the current substrate node to host this vnf.
               continue
            if node in self.node_info[substrate_node][previous_vnf_id]['current_substrate_nodes']:
               # If node has been used, cannot host this vnf
               # Current_substrate_nodes contains the nodes that have been used
               continue
            if node == self.src_substrate_node or node == self.dst_substrate_node:
                # Ingress and egress cannot host this vnf
                continue

            # Check CPU and cache resources
            cpu_available = self.graph.nodes[node]['cpu_capacity'] - self.graph.nodes[node]['cpu_used']
            cache_available = self.graph.nodes[node]['cache_capacity'] - self.graph.nodes[node]['cache_used']
            

            if cpu_request > cpu_available or cache_request > cache_available:
                # if node has not sufficient cpu, check next node. 
                continue

            # Check bandwidth resources
            is_bandwidth_sufficient = True
            bandwidth_usage_info = copy.copy(
                self.node_info[substrate_node][previous_vnf_id]['bandwidth_usage_info'])

            path = node_path[node]
            if path[0] != substrate_node:
                path.reverse()
            length = len(path)
            for i in range(0, length - 1):
                edge_key = frozenset((path[i], path[i + 1]))
                residual_bandwidth = None
                if edge_key in bandwidth_usage_info:
                    residual_bandwidth = bandwidth_usage_info[edge_key] - bandwidth_request
                else:
                    
                    residual_bandwidth = get_link_bandwidth_free(self.graph,path[i], path[i + 1]) - bandwidth_request
                if residual_bandwidth < 0:
                    #logger.warning('Bandwidth resources is not sufficient')
                    is_bandwidth_sufficient = False
                    break
                bandwidth_usage_info[edge_key] = residual_bandwidth
            if not is_bandwidth_sufficient:
                continue
            self.node_info[node][vnf_id]['bandwidth_usage_info'] = bandwidth_usage_info
            if not self.node_info[node][vnf_id]['latency'] or (_latency + latency) <= self.node_info[node][vnf_id]['latency']:
                self.node_info[node][vnf_id]['latency'] = _latency + latency
                self.node_info[node][vnf_id]['path'] = node_path[node]
                self.node_info[node][vnf_id]['flag'] = True
                self.node_info[node][vnf_id]['previous_substrate_node'] = substrate_node
                self.node_info[node][vnf_id]['current_substrate_nodes'] = self.node_info[substrate_node][previous_vnf_id]['current_substrate_nodes'][:]
                self.node_info[node][vnf_id]['current_substrate_nodes'].append(node)
                self.node_info[node][vnf_id]['src_path'] = self.node_info[substrate_node][previous_vnf_id]['src_path'][:] + node_path[node][:-1]
        return True
