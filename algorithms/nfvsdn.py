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
import copy
import logging
import numpy as np
import pandas as pd
from config import ROOT_PATH
import networkx as nx

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler(ROOT_PATH + './logs/MSF.log')
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

from algorithms.algorithm import Algorithm


class Goku(Algorithm):
    def __init__(self):
        self.name = "gr"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        self.latency_request = 0
        
        self.fail_for_band = 0
        self.fail_for_cache = 0
        self.fail_for_cpu = 0
        self.fail_resources = 0
        
        self.saved_band = 0
        self.saved_cpu = 0
        self.saved_cache = 0
        
        self.server_resources = 0
        self.services_requirements = 0
        self.G = 0 
        self.services = 0
        
        self.cpu_factor=2
        self.cache_factor=2
        self.band_factor=0.5
        self.boot_factor=0
        
        self.using_bit_rate = False
        self.bitrate_cut = 1.0
        self.latency_cumulative = 0  

    def clear_all(self):
        #logger.debug('clear all')
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None

    def install_substrate_network(self, substrate_network):
        self.substrate_network = substrate_network
        #self.single_source_minimum_latency_path = self.substrate_network.single_source_minimum_latency_path
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        self.using_bit_rate = False
        self.bitrate_cut = 1.0
        return self.sfc 

    def check_sfc(self,sfc):
        backup = False
        sfc_id = sfc.id
        is_backup = True if int(sfc_id.split("_")[-1]) % 2 == 0 else False

        new_sfc = 0
        if is_backup == True:
            prefixo, x = sfc_id.rsplit('_', 1)
            real_sfc_id = f"{prefixo}_{int(x) - 1}"
            route_info = self.substrate_network.sfc_route_info

            if real_sfc_id in list(route_info.keys()):
                route_info = route_info[real_sfc_id]
            else:
                return sfc
            servers_used = list(set([item for chave, valor in route_info.items() if chave not in ['src', 'dst'] for item in valor])) 
            return new_sfc 
        else:
            return sfc
    
    def set_costs(self,costs_parameters):
        self.cpu_factor=costs_parameters[0]
        self.cache_factor=costs_parameters[1]
        self.band_factor=costs_parameters[2]
        self.boot_factor=costs_parameters[3]
        return
    
    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def get_bit_rate_used(self):
        return self.bitrate_cut
    
    def is_using_bit_rate_cut(self):
        return self.using_bit_rate

    def start_algorithm(self, shareable_sfs=None, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc
        #logger.info('Algorithm start')
        if self.algorithm(substrate_network, sfc, shareable_sfs):
            #logger.info('Algorithm end, success')
            return True
        #logger.info('Algorithm end, failed')
        return False
        
    def algorithm(self, substrate_network, sfc, shareable_sfs=None):
        # Parte 1: Preparação dos dados de entrada
        sfs_dict = self.sfc.vnfs_dict
        net_info = substrate_network
        server_resources = net_info._node
        servers = list(server_resources.keys())
        new_server_resources = {}

        for server in servers:
            cpu_used = round(self.substrate_network.get_node_cpu_used(server),3)
            cache_used = round(self.substrate_network.get_node_cache_used(server),3)

            cpu_free =  round(self.substrate_network.get_node_cpu_free(server),3) 
            cache_free = round(self.substrate_network.get_node_cache_free(server),3)

            cpu_capacity =  round(self.substrate_network.get_node_cpu_capacity(server),3) 
            cache_capacity = round(self.substrate_network.get_node_cache_capacity(server),3)

            new_server_resources[server]= {'cpu_capacity':cpu_capacity,'cache_capacity':cache_capacity,
                                           'cpu_used':cpu_used,'cache_used':cache_used,
                                           'cpu_free':cpu_free,'cache_free':cache_free,'position':server_resources[server]['position']}

        server_resources = new_server_resources

        shareable_sfs = shareable_sfs if shareable_sfs is not None else {node_id: [] for node_id in server_resources.keys()}
        
        # Parte 2: Obtenção dos VNFs de origem e destino
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Parte 3: Obtenção dos nós de substrato para os VNFs de origem e destino
        src = sfc.get_substrate_node(src_vnf)
        dst = sfc.get_substrate_node(dst_vnf)
        
        # Parte 4: Inicialização da topologia da rede
        network_topology = net_info._adj

        # Parte 5: Criação do grafo da rede
        G = self.create_network_graph(network_topology)
            
        # Parte 6: Preparação dos requisitos de serviço
        services, service_requirements = self.prepare_service_requirements(sfs_dict)

        # Parte 7: Configuração do uso compartilhado de funções de serviço (SFs)
        self.configure_shareable_sfs(server_resources, shareable_sfs)

        # Parte 8: Encontrando a rota e calculando a latência
        bit_rate_trials = [1.0,0.95,0.75]
        is_success = False

        # Supondo que bit_rate_trials e outras variáveis estejam definidas
        for bitrate in bit_rate_trials:
            service_requirements_altered = copy.deepcopy(service_requirements)
            
            # Percorre o dicionário e altera o valor de out_bw para chaves que começam com "EC_TC"
            for chave in service_requirements_altered:
                if chave.startswith('EC_TC'):
                    service_requirements_altered[chave]['out_bw'] = service_requirements_altered[chave]['out_bw'] * bitrate

            route_info, latency = self.find_best_allocation_for_sfc(G, service_requirements_altered, server_resources, services, dst)

            # Parte 9: Avaliação do resultado com base na latência
            is_success = self.evaluate_result(latency, route_info)
            
            # if bitrate == 0.3:
            #     print(bitrate)

            if is_success and bitrate != 1.0:
                self.using_bit_rate = True
                self.bitrate_cut = bitrate
            # elif is_success and bitrate == 1.0:
            #     print(bitrate)

            if is_success:
                self.bitrate_cut = bitrate
                return is_success
            else:
                continue

        return is_success

    def create_network_graph(self, network_topology):
        import networkx as nx
        G = nx.Graph()
        for node, edges in network_topology.items():
            for target, edge_attr in edges.items():
                G.add_edge(node, target, bandwidth=edge_attr['bandwidth_free'], weight=1)
        return G

    def prepare_service_requirements(self, sfs_dict):
        service_requirements = {}
        services = []  # Lista para guardar os nomes dos serviços
        for item in sfs_dict:
            nome = item['name']
            services.append(nome)  # Adiciona o nome à lista
            service_requirements[nome] = {
                'CPU': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw'],
            }
        service_requirements['dst'] = {'CPU': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0}
        services = list(reversed(services))
        return services, service_requirements

    def configure_shareable_sfs(self, server_resources, shareable_sfs):
        for node_id, node_info in server_resources.items():
            node_info['reuse'] = []  # Inicializa o campo 'reuse'
            if node_id in shareable_sfs:
                for vnf in shareable_sfs[node_id]:
                    node_info['reuse'].append(vnf.id)

    def evaluate_result(self, latency, route_info):
        if latency > self.latency_request:
            self.latency = None
            self.route_info = False
            print("Serviço Negado por latência")
            return False
        else:
            self.latency = latency
            self.route_info = route_info
            return True

    # Modificando a função de alocação para usar a nova lógica de exploração
    def find_best_allocation_for_sfc(self, G, service_requirements, server_resources, services, dst):
        allocation_results = {'dst': {'allocated_server': dst, 'path': [], 'cost': 0}}
        current_location = dst # começa a alocação de trás pra frente 
        success = True

        for i, service in enumerate(services):
            # Verifica se há um próximo serviço na lista
            if i + 1 < len(services):
                next_service = services[i + 1]
            else:
                # Caso não exista, define o próximo serviço como None
                next_service = None
            
            best_server, best_path, cost_details = self.find_best_server_for_service_with_exploration(G, server_resources, service_requirements, current_location, service, next_service)

            allocation_results[service] = {
                'allocated_server': best_server,
                'path': best_path,
                'cost': cost_details
            }
            current_location = best_server
            # Se existe um servidor ótimo
            if best_server:  # Verifica se um servidor foi escolhido
                server_resources[best_server]['cpu_used'] += service_requirements[service]['CPU']
                server_resources[best_server]['cache_used'] += service_requirements[service]['cache']
                self.latency_cumulative += len(best_path) - 1
            else:
                #print(f"Falha.")
                success =  False
                break

        if success == False:
            return [],1000

        # ultima iteração para o src
        path_to_src = nx.dijkstra_path(G, current_location, 0, weight='weight')

        route_info = {key: list(reversed(value['path'])) for key, value in allocation_results.items()}
                    
        #calculo da latencia antes do src
        total_latency = sum(len(path) - 1 for path in route_info.values() if path)

        route_info['src'] = list(reversed(path_to_src))

        #colocando o dst no final
        first_key, first_value = next(iter(route_info.items()))
        del route_info[first_key]
        route_info[first_key] = first_value
        
        return route_info,total_latency

    def find_best_server_for_service_with_exploration(self, G, server_resources, service_requirements, current_location, service, next_service, exploration_margin=1):
        hops_allowed = self.latency_request - self.latency_cumulative
        best_cost, best_candidate, candidates = self.find_candidates_serves_for_sf(G, server_resources, service_requirements, current_location, service,hops_allowed)
        
        if best_cost == float('inf') or best_candidate == float('inf') or candidates == None:
            return False, False, False

        server_choose = best_candidate[0]
        path_to = best_candidate[1]
        min_cost = best_candidate[2]

        return server_choose, path_to, min_cost

    def find_candidates_serves_for_sf(self, G, server_resources, service_requirements, current_location, service,hops_allowed):
        cutoff = int((6-self.latency_cumulative))
        if cutoff < 0:
            return float('inf'), float('inf'), None
        if cutoff == 0:
            print()
        paths = dict(nx.single_source_shortest_path_length(G, current_location, cutoff=cutoff))

        paths[current_location] = 0  # Custo de 'mover' para o mesmo servidor é 0
        
        candidates = []
        best_cost = float('inf')

        # Função para verificar disponibilidade de largura de banda e recursos do servidor
        def check_resources(server, path, bandwidth_requirement, cpu_required, cache_required):
            if all(G[u][v]['bandwidth'] > bandwidth_requirement for u, v in zip(path, path[1:])):
                available_cpu =  server_resources[server]['cpu_capacity'] - server_resources[server]['cpu_used']
                available_cache = server_resources[server]['cache_capacity'] - server_resources[server]['cache_used']
                if available_cpu > cpu_required and available_cache > cache_required:
                    return True
            return False

        bandwidth_requirement = service_requirements[service]['out_bw']
        for server, num_hops in paths.items():
            # Exemplo de uso
            path = nx.shortest_path(G, current_location, server, weight='weight')
            reuse = service in server_resources[server]['reuse']
            cpu_required = 0.05 * service_requirements[service]['CPU'] if reuse else service_requirements[service]['CPU']
            cache_required = 0.05 * service_requirements[service]['CPU'] if reuse else service_requirements[service]['cache']
            boot_cost = 0
            available_cpu   = server_resources[server]['cpu_capacity'] - server_resources[server]['cpu_used']
            available_cache = server_resources[server]['cache_capacity'] - server_resources[server]['cache_used']

            if check_resources(server, path, bandwidth_requirement, cpu_required, cache_required):
                cpu_cost = self.calculate_cpu_cost(cpu_required, available_cpu)
                cache_cost = self.calculate_cache_cost(cache_required, available_cache)
                bandwidth_cost = self.calculate_bandwidth_cost(path,G, bandwidth_requirement)
                latency_cost = self.calculate_latency_cost(len(path) - 1)
                total_cost = self.calculate_total_cost(cpu_cost, cache_cost, bandwidth_cost, latency_cost)

                if total_cost < best_cost:
                    best_cost = total_cost

                candidates.append((server, path, total_cost, {
                    'cpu_cost': round(cpu_cost, 2),
                    'cache_cost': round(cache_cost, 2),
                    'bandwidth_cost': round(bandwidth_cost, 2),
                    'latency_cost': round(latency_cost, 2),
                    'total_cost': round(total_cost, 2)
                }))
            # else:
            #     # Sem recurso disponível
            #     pass

            # Se encontrou candidatos viáveis, não tenta com bitrate menor
            # if best_cost < float('inf'):
            #     break

        best_candidate = min(candidates, key=lambda x: x[2], default=(None, None, None, None))
        return best_cost, best_candidate, candidates

    # latency as restriction
    # def calculate_latency_cost(self,distance):
    #     if distance <= self.latency_request:
    #         return 0
    #     else:
    #         return 1000
        

    # def check_my_node_resources(server,server_resources, cpu_required, cache_required):
    #     """Verifica se o servidor e o caminho podem suportar os requisitos de recursos."""
    #     available_cpu = server_resources[server]['cpu_capacity'] - server_resources[server]['cpu_used']
    #     available_cache = server_resources[server]['cache_capacity'] - server_resources[server]['cache_used']

    #     # Verifique se os recursos de CPU estão disponíveis, considerando a condição especial
    #     if (available_cpu == 0 and cpu_required != 0) or (available_cpu != 0 and cpu_required > available_cpu):
    #         return False
        
    #     # Verifique se os recursos de cache estão disponíveis, considerando a condição especial
    #     if (available_cache == 0 and cache_required != 0) or (available_cache != 0 and cache_required > available_cache):
    #         return False

    #     return True

    def calculate_cpu_cost(self,cpu_required, available_cpu):
        """Calcula o custo do nó com base no uso de CPU."""
        if available_cpu == 0:
            return float('inf') if cpu_required != 0 else 0  # Considera infinito se CPU for necessário mas não disponível
        utilization_ratio = cpu_required / available_cpu
        cpu_cost = utilization_ratio * 10  # Custo como uma função linear da razão de utilização
        return cpu_cost

    def calculate_cache_cost(self,cache_required, available_cache):
        """Calcula o custo do nó com base no uso de cache."""
        if available_cache == 0:
            return float('inf') if cache_required != 0 else 0  # Considera infinito se cache for necessário mas não disponível
        utilization_ratio = cache_required / available_cache
        cache_cost = utilization_ratio * 10  # Custo como uma função linear da razão de utilização
        return cache_cost


    def calculate_bandwidth_cost(self,path,G, bandwidth_requirement):
        """Calcula o custo da largura de banda ao longo do caminho."""
        total_bandwidth_cost = 0
        for i in range(len(path) - 1):
            edge_weight = G[path[i]][path[i + 1]]['weight']
            total_bandwidth = G[path[i]][path[i + 1]]['bandwidth']/10  # Supondo que essa informação está disponível no grafo
            total_bandwidth_cost += (edge_weight * bandwidth_requirement/10) / total_bandwidth
        return (total_bandwidth_cost)*4

    def calculate_latency_cost(self,num_hops):
        """Calcula o custo de latência com base no número de saltos."""
        latency_cost = (num_hops/10)*1.5   # Exemplo: custo de latência aumenta linearmente com o número de saltos
        if self.latency_cumulative>6:
            print("Aqui")
            return 10000
        return latency_cost

    # Função para calcular o custo total
    def calculate_total_cost(self,cpu_cost, cache_cost, bandwidth_cost, latency_cost):
        boot_cost = 0
        return (cpu_cost * self.cpu_factor) + (cache_cost * self.cache_factor) + (bandwidth_cost * self.band_factor) + (latency_cost * 1)


    def check_resources(self,server,server_resources, cpu_required, cache_required):
        """Verifica se o servidor e o caminho podem suportar os requisitos de recursos."""
        available_cpu = server_resources[server]['cpu_capacity'] - server_resources[server]['cpu_used']
        available_cache = server_resources[server]['cache_capacity'] - server_resources[server]['cache_used']

        # Verifique se os recursos estão disponíveis
        if available_cpu >= cpu_required and available_cache >= cache_required:
            return True
        return False
