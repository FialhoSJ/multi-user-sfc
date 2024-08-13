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
import pandas as pd
from config import ROOT_PATH
import random
import deap
from deap import base, creator, tools, algorithms
import networkx as nx
import numpy


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


class Genetic(Algorithm):
    def __init__(self):
        self.name = "ga"
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.route_info = {}
        self.single_source_minimum_latency_path = None
        self.latency = None
        
        self.latency_request = 0
        self.min_latency = 0
        
        self.latency_minus_dst =  0
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

        self.cpu_factor=1
        self.cache_factor=1
        self.band_factor=1
        self.boot_factor=1

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
        return self.substrate_network

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        self.min_latency  = 0 
        return self.sfc

    def set_costs(self,costs_parameters):
        self.cpu_weight   =  costs_parameters[0]
        self.cache_weight =  costs_parameters[1]
        self.band_weight  =  costs_parameters[2]
        self.boot_weight  =  costs_parameters[3]

    def get_latency(self):
        return self.latency
    
    def get_route_info(self):
        return self.route_info

    def start_algorithm(self, shareable_sfs=None, **kwargs):
        substrate_network = self.substrate_network
        sfc = self.sfc
        #logger.info('Algorithm start')
        if self.algorithm(substrate_network, sfc, shareable_sfs):
            #logger.info('Algorithm end, success')
            return True
        #logger.info('Algorithm end, failed')
        return False
    

    def cut_topology(self,G_complex, complex_network_topology, server_resources_complex, node_reference, hops_cuff):
        nodes_within_hops_complex = nx.single_source_shortest_path_length(G_complex, node_reference, cutoff=hops_cuff)
        
        sub_graph_complex = G_complex.subgraph(nodes_within_hops_complex.keys())

        server_resources = {node: server_resources_complex[node] for node in sub_graph_complex.nodes if node != 0}

        new_network_topology = {}
        for node in sub_graph_complex.nodes:
            if node ==  0:  # Pula o nó '0' se presente
                continue
            new_network_topology[node] = {}
            for neighbor in sub_graph_complex.neighbors(node):
                if neighbor == 0:  # Pula o vizinho '0' se presente
                    continue
                if neighbor in sub_graph_complex.nodes:
                    new_network_topology[node][neighbor] = complex_network_topology[node][neighbor]

        G_non_complex = nx.Graph()
        
        for node, edges in new_network_topology.items():
            for target, edge_attr in edges.items():
                G_non_complex.add_edge(node, target, bandwidth=edge_attr['bandwidth_free'], weight=1)

        # Removendo a chave '0' após a criação para garantir que não esteja presente
        if 0 in G_non_complex:
            G_non_complex.remove_node(0)

        return G_non_complex, server_resources

    
    def algorithm(self,substrate_network, sfc, shareable_sfs=None):
        
        sfs_dict = self.sfc.vnfs_dict
        net_info = substrate_network

        server_resources = net_info._node

        shareable_sfs = shareable_sfs if shareable_sfs is not None else {node_id: [] for node_id in server_resources.keys()}

        # Get src and dst vnf
        src_vnf = sfc.get_src_vnf()
        dst_vnf = sfc.get_dst_vnf()

        # Get substrate network nodes that src and dst are assigned in advanced
        src = sfc.get_substrate_node(src_vnf)
        dst = sfc.get_substrate_node(dst_vnf)
        
        # Inicialização da topologia da rede
        network_topology = net_info._adj

        # Criação do grafo representando a rede com capacidade de banda
        A = nx.Graph()
        for node, edges in network_topology.items():
            for target, edge_attr in edges.items():
                A.add_edge(node, target, bandwidth=edge_attr['bandwidth_free'], weight=1)

        service_requirements = {} 
        services = []  # Lista para guardar os nomes

        for item in sfs_dict:
            nome = item['name']
            services.append(nome)  # Adiciona o nome à lista de nomes
            service_requirements[nome] = {
                'CPU': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw'],
            }

        services.append('dst')
        service_requirements['dst'] = {'CPU': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0}  

        # Adicionando o campo 'reuse' no node_table
        for node_id, node_info in server_resources.items():
            node_info['reuse'] = []
            if node_id in shareable_sfs:
                for vnf in shareable_sfs[node_id]:
                    node_info['reuse'].append(vnf.id)  # Acessando o atributo 'id' da VNF


        G,server_resources = self.cut_topology(G_complex=A,
                          complex_network_topology=network_topology,
                          server_resources_complex=server_resources,
                          node_reference=dst,hops_cuff=4)
        
        route_info, latency = self.genetic_alg(G,A,service_requirements,server_resources,services,dst)

        if latency > self.latency_request or route_info == False:
            self.latency = None
            self.route_info = False
            return False
        else:
            self.latency = latency
            self.route_info = route_info
            return True

    # Início da função fit_path modificado
    def genetic_alg(self, G,G_old, service_requirements, server_resources, services, dst):
        service_requirements_local = service_requirements
        server_resources_local = server_resources
        services_list = services

        servers_quantity = len(server_resources_local)
        services_q_real = len(service_requirements_local)-1

        available_servers  = list(server_resources.keys())

        # Verifica se a classe já existe e, em caso afirmativo, exclui-a
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # DEAP setup para minimizar o fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Pré-computação de caminhos mínimos e seus custos
        all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
        #all_pairs_shortest_path_length = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

        # Função para criar um indivíduo, garantindo que o último serviço seja alocado em last_service_server
        def create_individual():
            servers_chosen = random.sample(available_servers, services_q_real)
            # Construindo o indivíduo
            servers_chosen.append(dst)
            
            return creator.Individual(servers_chosen)

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def custom_crossover(ind1, ind2):
            tools.cxTwoPoint(ind1, ind2)
            ind1[-1] = dst
            ind2[-1] = dst
            return ind1, ind2

        # Função de mutação adaptada
        def custom_mutate(individual):
            # Para cada gene no indivíduo, exceto o último, aplica a mutação com uma certa probabilidade
            for i in range(len(individual) - 1):  # Ajustado para usar o comprimento do indivíduo
                if random.random() < 0.2:
                    individual[i] = random.choice(available_servers)
            return individual,

        def evaluate(individual):
                total_cost = 0

                latency_threshold = self.latency_request  # Limite de latência em ms

                # Rastrear o uso de recursos em cada servidor
                server_usage = {server_id: {'cpu_free': server['cpu_free'],
                                            'cache_free': server['cache_free'],
                                            'cpu_used': server['cache_used'],
                                            'cache_used': server['cache_used'],
                                            'reuse': server['reuse']
                                            } for server_id, server in server_resources.items()}
                
                # Rastrear o uso de largura de banda em cada link
                link_usage = {tuple(sorted((u, v))): 0 for u, v in G.edges()}

                total_latency = 0  # Inicializa a latência total acumulada

                for service_index, server_id in enumerate(individual):
                    if service_index >= len(services):  # Evita índices fora do alcance
                        continue
                    service = service_requirements[services[service_index]]
                    
                    # Verifica e calcula o custo com base em reuso
                    reuse = service in server_usage[server_id]['reuse']
                    node_resource_cost = 0 if reuse else 1
                    cpu_required = 0 if reuse else service['CPU']
                    cache_required = 0 if reuse else service['cache']

                    # Update boot_cost based on CPU requirements and potential reuse logic
                    boot_cost = 0
                    if server_usage[server_id]['cpu_used'] >= 17.27 or reuse:
                        boot_cost = 0
                    elif (server_usage[server_id]['cpu_used'] + cpu_required) >= 17.27:
                        boot_cost = 1

                    # Verifica e acumula o uso de CPU e cache
                    if server_usage[server_id]['cpu_free'] >= cpu_required and server_usage[server_id]['cache_free'] >= cache_required:
                        server_usage[server_id]['cpu_free'] -= cpu_required
                        server_usage[server_id]['cache_free'] -= cache_required
                        total_cost += (self.cpu_weight * node_resource_cost) + (self.cache_weight * node_resource_cost) + (self.boot_weight * boot_cost)
                    else:
                        return float('inf'),  # Solução inválida devido à exceder recursos

                    # Cálculo do custo de banda baseado em caminhos pré-computados e acumulação de latência
                    if service_index < len(individual) - 1:
                        next_server_id = individual[service_index + 1]
                        path = all_pairs_shortest_path[server_id][next_server_id]
                        path_cost = sum(self.band_weight for _ in range(len(path) - 1))
                        
                        total_cost += path_cost

                        # Acumula latência para a cadeia completa de serviços
                        total_latency += len(path) - 1  # Cada salto contribui com 1 ms

                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            edge_key = tuple(sorted((u, v)))

                            if edge_key in link_usage:
                                link_usage[edge_key] += service['out_bw']
                            else:
                                print(f"Aresta não encontrada: {edge_key}")

                            if link_usage[edge_key] > G[edge_key[0]][edge_key[1]]['bandwidth']:
                                return float('inf'),  # Solução inválida por exceder capacidade de banda

                # Verifica se a latência total excede o limite após calcular toda a cadeia de serviços
                if total_latency > latency_threshold:
                    return float('inf'),  # Custo altíssimo devido à latência excedente
                # if total_latency > 6 and total_latency <= 10:
                #     total_cost = total_cost + 2.0 

                return total_cost,

        # Registro das operações genéticas
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", custom_crossover)
        toolbox.register("mutate", custom_mutate)
        toolbox.register("select", tools.selTournament, tournsize=5)

        # Parâmetros do algoritmo genético
        population_size = 50
        crossover_probability = 0.7
        mutation_probability = 0.2
        number_of_generations = 100

        # Inicialização da população
        pop = toolbox.population(n=population_size)

        # Algoritmo genético
        result, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=number_of_generations, verbose=False)

        def display_paths_and_create_service_dict(best_individual):
            paths = []  # Lista para armazenar os caminhos ótimos
            service_to_server_dict = {}
            service_names = list(service_requirements_local.keys())  # Assumindo que você tem os nomes dos serviços
            
            for i, server_id in enumerate(best_individual):
                service_name = service_names[i]
                if i < len(best_individual) - 1:
                    next_server_id = best_individual[i + 1]
                    if server_id == next_server_id:
                        service_to_server_dict[service_name] = [server_id]
                    else:
                        path = nx.shortest_path(G, source=server_id, target=next_server_id, weight='weight')
                        service_to_server_dict[service_name] = path
                else:
                    service_to_server_dict[service_name] = [server_id]

            return service_to_server_dict

        # Após a execução do algoritmo genético
        best_ind = tools.selBest(pop, 1)[0]
        
        if best_ind.fitness.values[0] == float('inf'):
            return False, 100
        else:     
            service_to_server_dict = display_paths_and_create_service_dict(best_ind)

            service_to_server_dict.popitem()

            # Inverter a ordem dos itens no dicionário
            route_info = dict(reversed(list(service_to_server_dict.items())))

            src_node = next(reversed(route_info.values()))[0]

            path_to_src = list(reversed(nx.dijkstra_path(G_old, src_node, 0, weight='weight')))
        
            total_latency = sum(len(path) - 1 for path in route_info.values() if path) 

            route_info['src'] = path_to_src
            route_info['dst'] = []
            
            return route_info, total_latency

