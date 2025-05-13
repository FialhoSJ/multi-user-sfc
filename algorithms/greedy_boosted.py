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
import random
import copy
import networkx as nx
from algorithms.networkUtils import get_shortest_path_length,get_shortest_path,pre_get_single_source_minimum_latency_path, get_link_latency

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
from algorithms.algorithm import Algorithm
from config import ROOT_PATH
ch = logging.FileHandler(ROOT_PATH + './logs/GreedyAlgorithm.log')
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')


class GreedyOptAlgorithm(Algorithm):
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
        self.route_info = None
        self.latency = None
        self.mono = False
        self.old_greedy = True
        self.forbidden_matches = {}
        self.is_backup = False
        self.graph = None
        self.single_source_minimum_latency_path = None

    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.graph = None
        self.route_info = None
        self.latency = None
        self.is_backup = False
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self, substrate_network, sfc_list, shareable_sfs=[]):
        self.substrate_network = substrate_network
        self.graph = copy.deepcopy(substrate_network.graph)
        self.add_mobile_user_to_graph(substrate_network,sfc_list)
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
        self.route_info = {}
        self.latency = None
        return self.sfc
        #is_backup = True if sfc.id.split("_")[2] == 'backup' else False
        # self.is_backup = is_backup
        # if is_backup:
        #     split = sfc.id.split("_")
        #     original_sfc_id = f"{split[0]}_{split[1]}_{split[3]}_{split[4]}" 
        #     route_info = self.substrate_network.sfc_route_info[original_sfc_id]
        #     for vnf, rf in route_info.items():
        #         if vnf not in ['src','dst']:
        #             node_used = route_info[vnf][0]
        #             correct_name =  vnf + "_b"
        #             self.forbidden_matches[correct_name] = node_used
        #return self.sfc

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
        
    def check_solution(self):
        if not isinstance(self.latency, (int, float)) or self.latency < 0 or self.latency > self.sfc.get_latency_request() or not self.route_info:
            return False
        if len(list(self.route_info.keys()))!=6:
            return False
        return True

        # for vnf, server_forbidden in self.forbidden_matches.items():
        #     try:
        #         server_used = self.route_info[vnf][0]
        #         if server_used == server_forbidden:
        #             self.route_info = False
        #             self.latency = None
        #             return False
        #     except:
        #         self.route_info = False
        #         self.latency = None
        #         return False

    def handle_failure(self):
        self.route_info = False
        self.latency = None
    
    # def install_SFC(self, sfc):
    #     self.sfc = sfc
    #     is_backup = True if sfc.id.split("_")[2] == 'backup' else False
    #     self.is_backup = is_backup
    #     if is_backup:
    #         split = sfc.id.split("_")
    #         original_sfc_id = f"{split[0]}_{split[1]}_{split[3]}_{split[4]}" 
    #         route_info = self.substrate_network.sfc_route_info[original_sfc_id]
    #         for vnf, rf in route_info.items():
    #             if vnf not in ['src','dst']:
    #                 node_used = route_info[vnf][0]
    #                 correct_name =  vnf + "_b"
    #                 self.forbidden_matches[correct_name] = node_used
    #     return self.sfc

    def start_algorithm(self):
        sfc = self.sfc
        logger.info("Start algorithm")
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

    def get_latency(self):
        return self.latency
    
    def get_route_info(self):
        return self.route_info

    def algorithm(self, sfc):
        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advance
        src_substrate_node = sfc.get_substrate_node(src_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)
        
        single_source_minimum_latency_path = pre_get_single_source_minimum_latency_path(self.graph)

        route_info = {}
        bandwidth_usage_info = {}

        latency = 0
        used_node = [dst_substrate_node, src_substrate_node]

        number_of_vnfs = sfc.get_number_of_vnfs()
        current_vnf = dst_vnf
        current_substrate_node = dst_substrate_node

        servers = list(self.graph.nodes())
        
        # Inicializa o dicionário de recursos dos servidores
        server_resources = {
            server: {
                'cpu_capacity': self.graph.nodes[server]['cpu_capacity'],
                'cache_capacity':self.graph.nodes[server]['cache_capacity'],
                'cpu_used': self.graph.nodes[server]['cpu_used'],
                'cache_used': self.graph.nodes[server]['cache_used'],

                'cpu_capacity': self.graph.nodes[server]['cpu_capacity'],
                'cache_capacity':self.graph.nodes[server]['cache_capacity'],
                'cpu_used': self.graph.nodes[server]['cpu_used'],
                'cache_used': self.graph.nodes[server]['cache_used'],
                'reuse': []
            } for server in servers if  self.graph.nodes[server]['cpu_capacity'] > 0 and self.graph.nodes[server]['cache_capacity']}
        first_vnf = True

        nodes_used = []
        servers_to_check = list(server_resources.keys())
        for i in range(number_of_vnfs - 1, -1, -1):
            prev_vnf = current_vnf.get_previous_vnf()
            vnf_id = prev_vnf.id
            cpu_request = sfc.get_vnf_cpu_request(prev_vnf)
            cache_request = sfc.get_vnf_cache_request(prev_vnf)
            bandwidth_request = sfc.get_link_bandwidth_request(prev_vnf.id, current_vnf.id)

            min_latency = float("inf")
            node = None
            random.shuffle(servers_to_check)
            for node_a in servers_to_check:
                #if not self.mono:
                if not first_vnf:
                    if node_a == current_substrate_node or node_a in nodes_used:
                        continue
                
                forbidden = False
                for vnf_f,node_f in self.forbidden_matches.items():
                    if node_f == node_a and vnf_f == vnf_id:
                        forbidden =True
                if forbidden:
                    continue

                cpu_used = server_resources[node_a]['cpu_used']
                cache_used = server_resources[node_a]['cache_used']

                cpu_cap = server_resources[node_a]['cpu_capacity']
                cache_cap = server_resources[node_a]['cache_capacity']

                # Agora buscamos valores diretamente em server_resources:
                cpu_available = round(cpu_cap - cpu_used,2)  
                cache_available = round(cache_cap - cache_used,2) 

                if cpu_cap <= 0 or cache_cap <= 0:
                    continue

                if cpu_available <= 0 or cache_available <= 0:
                    continue

                if cpu_used + cpu_request > cpu_cap or cache_used + cache_request > cache_cap:
                    continue

                if cpu_request > cpu_available:
                    logger.debug("Node %s não tem CPU suficiente para %s", node_a, cpu_request)
                    continue
                if cache_request > cache_available:
                    logger.debug("Node %s não tem CACHE suficiente para %s",node_a, cache_request)
                    continue

                # Verificando link (apenas se não for laço no mesmo nó)
                if node_a == node:
                    edge_latency = 0
                else:
                    # bandwidth_available = substrate_network.get_link_bandwidth_free(e[0], e[1])
                    # if bandwidth_request > bandwidth_available:
                    #     logger.debug("Aresta (%s, %s) sem banda suficiente", e[0], e[1])
                    #     continue
                    edge_latency = single_source_minimum_latency_path[current_substrate_node][0][node_a]

                # Verifica se a latência desse caminho é a menor
                if edge_latency < min_latency:
                    min_latency = edge_latency
                    node = node_a
            
            # Se encontrou um nó para alocar
            if node is not None:
                nodes_used.append(node)
                # Atualizamos o dicionário de recursos
                server_resources[node]['cpu_used'] += cpu_request
                server_resources[node]['cache_used'] += cache_request
                
                # Se for usar a banda, você também decrementa a banda do enlace
                # se node != current_substrate_node, por exemplo
                # Ajuste do route_info e soma de latência
                if node == current_substrate_node:
                    route_info[prev_vnf.id] = [node]
                else:
                    route_info[prev_vnf.id] = single_source_minimum_latency_path[node][1][current_substrate_node]
                used_node.append(node)
                latency += min_latency

            else:
                logger.debug("Não foi possível alocar VNF")
                self.route_info = {}
                self.latency = None
                return False

            current_substrate_node = node
            current_vnf = prev_vnf
            first_vnf = False
        try:
            path = get_shortest_path(self.graph,src_substrate_node, node)
            path_latency = get_shortest_path_length(self.graph,src_substrate_node, node)
        except:
            logger.warning('Não há caminho entre src e primeira VNF: %s - %s',
                        src_substrate_node, node)
            self.route_info = {}
            self.latency = None
            return False

        # Adicionamos esse path como 'src' no route_info
        route_info['src'] = path
        route_info['dst'] = []
        latency += path_latency

        # Define route_info e latency no objeto
        self.route_info = route_info
        self.latency = latency

        # Se você precisa fazer algum ajuste de latência baseado em edges do path:
        path = self.route_info['src']
        for i in range(len(path) - 1):
            edge_latency = get_link_latency(self.graph,path[i], path[i + 1])
            self.latency = self.latency - edge_latency


