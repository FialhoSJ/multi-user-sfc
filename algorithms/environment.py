import copy
import gymnasium as gym
import numpy as np
from algorithms.networkUtils import get_shortest_path, pre_get_single_source_minimum_latency_path, calcular_latencia_total, get_available_shortest_path
from algorithms.networkUtils import calculate_computational_latency, calculate_latency_betwen_nodes
from gymnasium import spaces
import time
import networkx as nx

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

class NetworkEnv(gym.Env):
    def __init__(self, graph='', services=[0], service_requirements=1, latency_request=1, dst=1, valid_nodes=1, pesos=None, sfc=None):
        super().__init__()

        # Armazenando o grafo e outros parâmetros
        self.G = graph
        self.valid_nodes = valid_nodes

        # Parâmetros dos serviços
        self.service_requirements = service_requirements
        self.services = services
        self.service = services[0]
        self.dst_node = dst
        self.current_location = dst
        self.latency_request = latency_request
        self.latency_used = 0

        # Fatores de custo
        self.cpu_factor = pesos["cpu"]
        self.cache_factor = pesos["cache"]
        self.band_factor = pesos["band"]
        self.latency_factor = pesos["latency"]

        # Variáveis de controle
        self.is_training = True
        self.success = False
        self.reuse = False
        self.servers_used = []  # Usar set para verificação mais rápida
        self.allocation_results = {}

        # Inicialização dos objetos sfc e dst_vnf
        self.sfc = sfc  # Certifique-se de que o objeto sfc é passado para o ambiente       
        # Espaços de observação e ação
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.valid_nodes) * 5,),  # Pode ser melhor modularizado no futuro
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.valid_nodes))
        self.cached_paths = {}

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente e reinicializa os parâmetros, restaurando o estado original do grafo.
        """

        self.G = copy.deepcopy(self.G_backup)
        # Reinicializa as variáveis de controle e custos
        self.latency_used = 0
        self.current_location = self.dst_node
        self.servers_used.clear()  # Limpa o set de servidores usados
        self.service = self.services[0]
        self.ac_total_cost = self.total_reward = self.reward = self.total_cost = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        self.path = None
        self.buscou_no_grafo = False
        self.bandwidth_required = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth() 
        # self.band_test = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth() 

        return self.get_normalized_state(), {}

    def step(self, action):

        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        
        done = False
        self.server = self.valid_nodes[int(action)]

        if (self.current_location, self.server) not in self.cached_paths:
            self.cached_paths[(self.current_location, self.server)] = get_available_shortest_path(self.G,
                                                                                                  self.current_location,
                                                                                                  self.server,
                                                                                                  self.bandwidth_required)
        
        self.path = self.cached_paths[(self.current_location, self.server)]

        self.reuse = self.verificar_reuso_de_servico(self.G.nodes[self.server], self.sfc.get_vnf_by_id(self.service))
        if not self.allocate_resources_on_node(self.G, self.server, self.session_number, self.reuse):
            return self._fail_step('resource')

        # A lógica de alocação de banda e latência está correta. O buffer de 3% (1.03) é uma boa prática.
        sucess, latency = self.allocate_bandwidth_along_path(self.G, self.path, self.bandwidth_required , self.service)
        if not sucess:
            new_path = get_available_shortest_path(self.G,self.current_location,self.server,self.bandwidth_required)

            if not new_path:
                return self._fail_step('bandwidth')
            self.cached_paths[(self.current_location, self.server)] = new_path
            self.path = self.cached_paths[(self.current_location, self.server)]
            sucess, latency = self.allocate_bandwidth_along_path(self.G, self.path, self.bandwidth_required , self.service)
        if not sucess:
            return self._fail_step('bandwidth')
        
        self.latency_used += latency
        
        if self.latency_used > self.latency_request:
            return self._fail_step('latency')
        
        self.total_cost = self.calculate_total_cost(self.G)
        
        self.ac_total_cost += self.total_cost
        self.reward = -(self.total_cost**1.5)
        self.total_reward += self.reward

        self.servers_used.append(self.server)  

        if not self.is_training:
            self.allocation_results[self.service] = {
                'allocated_server': self.server,
                'path': copy.deepcopy(self.path),
                'cost': self.total_cost
            }

        # Verifica se a cadeia de serviços foi concluída.
        if self.service == self.services[-1]:
            done = True
            self.success = True
        else:
            current_index = self.services.index(self.service)
            self.service = self.services[current_index + 1]
            self.update_bandwidth_required()

        self.current_location = self.server

        return self.get_normalized_state(), self.reward, done, False, {}    
    def set_graph(self,graph):
        self.G_backup = copy.deepcopy(graph)
        self.G = graph

    def update_bandwidth_required(self):
        if not self.service_requirements or not self.service:
            raise ValueError("Variaveis não instanciadas")
        self.bandwidth_required = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth() 
        # self.band_test = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth() 

    def set_dst_node(self,dst_node):
        if not dst_node:
            raise ValueError("dst_node None")
        self.dst_node = dst_node
        self.current_location = dst_node
    
    
    def _fail_step(self, reason):
        self.fail_reason = reason
        self.success = False
        state = self.get_normalized_state()
        self.total_cost = 2000
        self.reward = -self.total_cost
        done = True
        return state, self.reward, done, False, {}


    
    def allocate_resources_on_node(self,graph, node_id,session_id, is_shareable):

        vnf = self.sfc.get_vnf_by_id(self.service)
        service_id = vnf.id
        service_key = (service_id, session_id)
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()
        node = graph.nodes[node_id]

        if node['type'] not in ['server', 'mobile_device']:
            return False
        if node['cpu_used'] + cpu_required > node['cpu_capacity']:
            return False
        if node['cache_used'] + cache_required > node['cache_capacity']:
            return False

        if service_key in node['services']:
            node['services'][service_key]['copys'] += 1
            if not is_shareable:
                node['cpu_used'] += cpu_required
                node['cache_used'] += cache_required
        else:
            node['services'][service_key] = {
                'cpu': cpu_required,
                'cache': cache_required,
                'copys': 1
            }
            node['cpu_used'] += cpu_required
            node['cache_used'] += cache_required
            if is_shareable:
                node['reuse'].append(vnf)

        return True
    
    def _commit_bandwidth_on_link(self, graph, u, v, vnf, bandwidth_required, ms_name):
        """
        Aloca a banda e calcula a latência para um único enlace (u, v).
        Esta função é o "coração" da lógica de alocação, extraída para reutilização.
        """
        edge = graph.edges[u, v]
        
        # Calcula a latência para este enlace específico
        latency = calculate_latency_betwen_nodes(graph, u, v, vnf)

        # Atualiza o dicionário de serviços em trânsito
        if ms_name in edge['services_in_transit']:
            edge['services_in_transit'][ms_name]['copys'] += 1
        else:
            edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bandwidth_required}
        
        # Atualiza a banda total utilizada no enlace
        edge['bandwidth_used'] += bandwidth_required
        
        return latency
    

    def allocate_bandwidth_along_path(self, graph, path, bandwidth_required, ms_name):
        """
        Verifica e aloca banda em todos os enlaces de um caminho, reutilizando a lógica
        de alocação por enlace.
        """
        # FASE 1: Verificação (permanece inalterada, é a garantia de segurança)
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                return False, float("inf")

        # FASE 2: Alocação (agora muito mais limpa)
        total_latency = 0.0
        vnf = self.sfc.get_vnf_by_id(ms_name)

        for u, v in zip(path[:-1], path[1:]):
            # Delega a lógica de alocação para a função auxiliar
            link_latency = self._commit_bandwidth_on_link(
                graph, u, v, vnf, bandwidth_required, ms_name
            )
            total_latency += link_latency
                
        return True, total_latency




    def calculate_total_cost(self, graph):
        """
        Calcula o custo total da alocação de um serviço, considerando os recursos utilizados.
        A fórmula aplica um custo exponencial para penalizar fortemente a alta utilização de recursos.
        """
        node = graph.nodes[self.server]

        # --- Custo de Recursos Computacionais (CPU e Cache) ---
        # Se o serviço for reutilizado, o custo de alocação de recursos é zero.
        
        # Adicionada verificação para evitar divisão por zero.
        cpu_capacity = node["cpu_capacity"] if node["cpu_capacity"] > 0 else 1
        self.cpu_cost = (node["cpu_used"] / cpu_capacity + 1) ** self.cpu_factor if not self.reuse else 0
        
        cache_capacity = node["cache_capacity"] if node["cache_capacity"] > 0 else 1
        self.cache_cost = (node["cache_used"] / cache_capacity + 1) ** self.cache_factor if not self.reuse else 0

        if self.server in self.servers_used:
            self.cpu_cost = self.cpu_cost*1.1
            self.cache_cost = self.cache_cost*1.1
        # --- Custo de Rede (Latência e Largura de Banda) ---
        # Estes custos só se aplicam se houver um caminho de rede (comprimento >= 2).
        
        if len(self.path) >= 2:
            # Custo de Latência
            latency_request = self.latency_request if self.latency_request > 0 else 1
            self.latency_cost = ((self.latency_used / latency_request) + 1) ** self.latency_factor

            # # Custo de Largura de Banda
            # capacity_band, used_band = self.get_critical_link_info(self.G, self.path)
            # capacity_band = capacity_band if capacity_band > 0 else 1
            
            # # LÓGICA CORRIGIDA: A condição agora é baseada no comprimento do caminho, não no custo de latência.
            # self.bandwidth_cost = (used_band / capacity_band + 1) ** self.band_factor
            self.bandwidth_cost = 0

        else:
            # Se não há caminho, não há custo de rede.
            self.latency_cost = 0
            self.bandwidth_cost = 0



        return sum([
            self.cpu_cost,
            self.cache_cost,
            self.latency_cost,
            self.bandwidth_cost
        ])


    def get_normalized_state(self):
        """
        Gera o vetor de estado normalizado para o agente de RL.

        Para cada nó válido na rede, esta função calcula um conjunto de 6 métricas
        que representam o "custo" e a "viabilidade" de alocar o serviço atual
        naquele nó. O estado final é a concatenação desses vetores para todos os nós.

        O vetor de estado para CADA nó contém 6 elementos normalizados [0, 1]:
        1.  Custo de CPU projetado: Utilização de CPU se o serviço for alocado aqui.
        2.  Custo de Cache projetado: Utilização de cache se o serviço for alocado aqui.
        3.  Custo de Latência projetado: Latência total se o serviço for alocado aqui.
        4.  Custo de Banda projetado: Utilização de banda no link mais crítico do caminho.
        5.  Flag de Nó Usado: 1.0 se o nó já foi usado para outro serviço nesta requisição.
        6.  Flag de Impossibilidade: 1.0 se a alocação neste nó for impossível (excede 100% de algum recurso).
        
        Retorna:
            np.array: O vetor de estado completo, achatado e normalizado.
        """
        state_vectors = []

        for node_id in self.valid_nodes:
            node = self.G.nodes[node_id]

            # --- 1. Cálculo de Custos de Recursos (CPU & Cache) ---
            
            # Evita divisão por zero se a capacidade for 0
            cpu_capacity = node["cpu_capacity"] if node["cpu_capacity"] > 0 else 1
            cache_capacity = node["cache_capacity"] if node["cache_capacity"] > 0 else 1

            # Verifica se o serviço pode ser reutilizado neste nó
            is_reusable = self.verificar_reuso_de_servico(node, self.sfc.get_vnf_by_id(self.service))
            
            # Requerimento de recursos é zero se houver reutilização
            cpu_required = 0 if is_reusable else self.service_requirements[self.service]["cpu"]
            cache_required = 0 if is_reusable else self.service_requirements[self.service]["cache"]
            
            # Calcula o custo projetado (utilização se o serviço for alocado aqui)
            projected_cpu_cost = (node["cpu_used"] + cpu_required) / cpu_capacity
            projected_cache_cost = (node["cache_used"] + cache_required) / cache_capacity


            # --- 2. Cálculo de Custos de Rede (Latência & Banda) ---

            # Obtém o caminho mais curto (usando cache para otimização)
            if (self.current_location, node_id) not in self.cached_paths:
                path = get_available_shortest_path(self.G,
                                                   self.current_location,
                                                   node_id,
                                                   self.bandwidth_required*1.03)
                self.cached_paths[(self.current_location, node_id)] = path
            else:
                path = self.cached_paths[(self.current_location, node_id)]

            # Custo de Latência
            latency_request = self.latency_request if self.latency_request > 0 else 1
            path_latency = calcular_latencia_total(path, self.G)
            projected_latency_cost = (self.latency_used + path_latency) / latency_request

            # # Custo de Banda (baseado no link mais congestionado do caminho)
            # if len(path) >= 2:
            #     link_capacity, link_used = self.get_critical_link_info(self.G, path)
            #     link_capacity = link_capacity if link_capacity > 0 else 1
            #     projected_bandwidth_cost = (link_used + self.bandwidth_required) / link_capacity
            # elif path == []:
            #     projected_bandwidth_cost = 1 # Sem caminho, sem custo de banda
            # else: 
            #     projected_bandwidth_cost = 0 # Sem caminho, sem custo de banda

            # --- 3. Geração das Flags de Estado ---
            
            # Flag que indica se o nó já foi escolhido nesta SFC
            is_server_used = 1.0 if node_id in self.servers_used else 0.0
            
            # Flag que indica se a alocação é impossível (excede >100% de algum recurso)
            # Usa os valores não cortados para uma verificação precisa da impossibilidade.
            cant_allocate = 1.0 if (projected_cpu_cost > 1.0 or
                                    projected_cache_cost > 1.0 or
                                    projected_latency_cost > 1.0 ) else 0.0

            
            # --- 4. Montagem do Vetor de Estado para este Nó ---
            # Os valores de custo são cortados em 1.0 para se manterem dentro do espaço de observação [0, 1].
            node_state = [
                min(projected_cpu_cost, 1.0),
                min(projected_cache_cost, 1.0),
                min(projected_latency_cost, 1.0),
                # min(projected_bandwidth_cost, 1.0),
                is_server_used,
                cant_allocate
            ]
            state_vectors.extend(node_state)

        return np.array(state_vectors, dtype=np.float32)    
    
    def get_critical_link_bandwidth(self, graph, path):
        min_available_bandwidth = float('inf')  # Inicializa com um valor muito grande para comparar
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            if graph.has_edge(u, v):
                edge_data = graph[u][v]
                # Calcula a capacidade de banda disponível (capacidade total - já usada)
                available_bandwidth = edge_data.get('bandwidth_capacity', 0) - edge_data.get('bandwidth_used', 0)
                
                # Atualiza a largura de banda disponível mínima se encontrar um link com menor capacidade
                min_available_bandwidth = min(min_available_bandwidth, available_bandwidth)
            else:
                raise Exception(f"O link entre {u} e {v} não existe")
        
        return min_available_bandwidth  # Retorna o menor valor de banda disponível
    

    def get_critical_link_info(self, graph, path):
        min_available_bandwidth = float('inf')  # Inicializa com valor muito alto para comparar a banda disponível
        critical_link_capacity = 0  # Variável para armazenar a capacidade do link crítico
        critical_link_used = 0  # Variável para armazenar a banda usada no link crítico

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            if graph.has_edge(u, v):
                edge_data = graph[u][v]
                # Calcula a banda disponível no link (capacidade total - banda usada)
                available_bandwidth = edge_data.get('bandwidth_capacity', 0) - edge_data.get('bandwidth_used', 0)
                
                # Verifica se o link atual é o mais crítico (com a menor banda disponível)
                if available_bandwidth < min_available_bandwidth:
                    min_available_bandwidth = available_bandwidth
                    # Atualiza a capacidade total e a banda usada do link crítico
                    critical_link_capacity = edge_data.get('bandwidth_capacity', 0)
                    critical_link_used = edge_data.get('bandwidth_used', 0)
            else:
                raise Exception(f"O link entre {u} e {v} não existe")
        
        return critical_link_capacity, critical_link_used




    def verificar_reuso_de_servico(self,servidor: dict, servico: object) -> bool:
        
        service_id = servico.id
        is_shareable = service_id.startswith(SHAREABLE_PREFIXES)
        
        if not is_shareable:
            return False

        if 'services' not in servidor or not servidor['services']:
            return False
            
        for existing_service_id, _ in servidor['services'].keys():
            if existing_service_id == service_id:
                return True

        return False

# def verificar_chave(dicionario, chave, prefixo):
#     string1, string2 = chave  # Desempacotando a tupla (string1, string2)
    
#     # Verificar se a chave existe no dicionário com as condições
#     for (string3, string4), valor in dicionario.items():
#         if string3.startswith(prefixo) and string4 == string2:
#             # Se string3 tem o prefixo e string4 é igual a string2
#             if string1.startswith(prefixo):
#                 return True  # Chave encontrada
#     return False  # Chave não encontrada




