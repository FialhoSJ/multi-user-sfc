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
            shape=(len(self.valid_nodes) * 6,),  # Pode ser melhor modularizado no futuro
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
        self.bandwidth_required = self.service_requirements[self.service]["in_bw"]

        return self.get_normalized_state(), {}

    def step(self, action):

        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        done = False
        self.server = self.valid_nodes[int(action)]

        if (self.current_location, self.server) not in self.cached_paths:
            self.cached_paths[(self.current_location, self.server)] = nx.shortest_path(self.G, self.current_location, self.server, weight='weight')
        
        self.path = self.cached_paths[(self.current_location, self.server)]

        # Verifica se há recursos suficientes para alocar o serviço
        self.reuse = self.check_reuse(self.service, self.server)
        if not self.allocate_resources_on_node(self.G, self.server,self.session_number, self.reuse):
            return self._fail_step('resource')

        sucess, latency = self.allocate_bandwidth_along_path(self.G, self.path, self.bandwidth_required*1.03, self.service)
        if not sucess:
            return self._fail_step('bandwidth')
        
        self.latency_used+=latency
        
        # Verifica se os limites de latência ou largura de banda são atingidos
        if self.latency_used > self.latency_request:
            return self._fail_step('latency')
        
        self.total_cost = self.calculate_total_cost(self.G)
        if self.total_cost>2000:
            epa=3
        self.ac_total_cost += self.total_cost
        self.reward = -self.total_cost
        self.total_reward += self.reward
        if self.server in self.servers_used:
            self._fail_step("resource")
        self.servers_used.append(self.server)  

        if not self.is_training:
            self.allocation_results[self.service] = {
                'allocated_server': self.server,
                'path': self.path,
                'cost': self.total_cost
            }

        done = self.service == self.services[-1]
        if not done:
            self.service = self.services[self.services.index(self.service) + 1]
            self.update_bandwidth_required()
        else:
            self.success=True
        self.current_location = self.server

        return self.get_normalized_state(), self.reward, done, False, {}
    
    def set_graph(self,graph):
        self.G_backup = copy.deepcopy(graph)
        self.G = graph

    def update_bandwidth_required(self):
        if not self.service_requirements or not self.service:
            raise ValueError("Variaveis não instanciadas")
        self.bandwidth_required = self.service_requirements[self.service]["in_bw"]

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
    

    def allocate_bandwidth_along_path(self,graph, path, bandwidth_required, ms_name):
        """
        Verifica e aloca banda em todos os enlaces de um caminho.
        
        Parâmetros:
        - graph: grafo com os nós e enlaces (com atributos 'bandwidth_capacity' e 'bandwidth_used')
        - path: lista de nós representando o caminho
        - bandwidth_required: banda necessária para a transmissão
        - ms_name: nome do microsserviço ou identificador da transmissão
        
        Retorna:
        - soma das latências de comunicação entre os nós
        """
        total_latency = 0.0

        # Primeira verificação: checa se há banda em todos os enlaces
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                return False, float("inf")

        # Segunda etapa: aloca efetivamente a banda e acumula latência
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edges[u, v]
            vnf = self.sfc.get_vnf_by_id(ms_name)
            latency = calculate_latency_betwen_nodes(graph, u, v, vnf)  # None se não houver VNF necessário
            total_latency += latency

            if ms_name in edge['services_in_transit']:
                edge['services_in_transit'][ms_name]['copys'] += 1
                edge['bandwidth_used'] += bandwidth_required
            else:
                edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bandwidth_required}
                edge['bandwidth_used'] += bandwidth_required
        return True, total_latency



    def calculate_total_cost(self, graph):
        """
        Calcula o custo total considerando recursos, latência e largura de banda.
        Os valores de CPU, cache e tipo do nó são extraídos diretamente do grafo.
        """
        node = graph.nodes[self.server]


        self.cpu_cost = (node["cpu_used"] /node["cpu_capacity"] + 1) ** self.cpu_factor if not self.reuse else 0
        self.cache_cost = (node["cache_used"] /node["cache_capacity"] + 1) ** self.cache_factor if not self.reuse else 0

        self.latency_cost = ((self.latency_used / self.latency_request) + 1) ** self.latency_factor if len(self.path) >=2 else 0

        capacity_band, used_band=self.get_critical_link_info(self.G,self.path)
        self.bandwidth_cost = (used_band/capacity_band+1)**self.band_factor if self.latency_cost != 0 else 0

        self.boot_cost = 0

        if self.server in self.servers_used:
            self.cpu_cost *= self.cpu_factor
            self.cache_cost *= self.cache_factor

        return sum([
            self.cpu_cost,
            self.cache_cost,
            self.latency_cost,
            self.bandwidth_cost,
            self.boot_cost
        ])

    def check_reuse(self, service, server):
        """
        Verifica se o serviço pode ser reutilizado no servidor.
        """
        for sf_shareable in SHAREABLE_PREFIXES:
            if service.startswith(sf_shareable):
                if verificar_chave(self.G.nodes[server]['services'], (service, self.session_number), sf_shareable):
                    return True
        return False


    def get_normalized_state(self):
        """
        Retorna o estado normalizado do ambiente com uma única iteração sobre os nós válidos.
        """
        state = []

        for node_id in self.valid_nodes:
            node = self.G.nodes[node_id]
            cpu_used_norm = (node["cpu_used"])/100
            cache_used_norm = node["cache_used"]/100

            cpu_capacity_norm = (node["cpu_capacity"])/100
            cache_capacity_norm = node["cache_capacity"]/100

            reuse = self.check_reuse(self.service, node_id)
            
            if not reuse:
                cpu_request_norm = self.service_requirements[self.service]["cpu"] / 100 
                cache_request_norm = self.service_requirements[self.service]["cache"] / 100
            else:
                cpu_request_norm = 0
                cache_request_norm = 0

            cost_cpu = (cpu_request_norm+cpu_used_norm)/cpu_capacity_norm if not reuse else 0
            cost_cache = (cache_request_norm+cache_used_norm)/cache_capacity_norm if not reuse else 0

            cost_cpu = min(cost_cpu,1)
            cost_cache= min(cost_cpu,1)

            # Parte 1: estado de recursos e flags
            state.extend([
                cost_cpu,
                cost_cache,
                0 if node_id not in self.servers_used else 1
            ])


            # Parte 3: latência e viabilidade de alocação
            if (self.current_location, node_id) not in self.cached_paths:

                path = self.cached_paths[(self.current_location, node_id)] = nx.shortest_path(self.G, self.current_location, node_id, weight='weight')
                self.cached_paths[(self.current_location, node_id)] = path
            else:
                path = self.cached_paths[(self.current_location, node_id)]

            cost_latency = (calcular_latencia_total(path, self.G) + self.latency_used) / self.latency_request 
            state.append(min(cost_latency, 1))

            min_capacity, min_used = self.get_critical_link_info(self.G,path)
            band_cost = (min_used+self.bandwidth_required)/min_capacity if len(path)>=2 else 0
            state.append(min(band_cost, 1))

            cant_allocate = 1 if cost_cpu>=1 or cost_cache >= 1 or cost_latency>=1 or band_cost>=1 else 0

            state.append(cant_allocate)

        return np.array(state, dtype=np.float32)
    
    
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





def verificar_chave(dicionario, chave, prefixo):
    string1, string2 = chave  # Desempacotando a tupla (string1, string2)
    
    # Verificar se a chave existe no dicionário com as condições
    for (string3, string4), valor in dicionario.items():
        if string3.startswith(prefixo) and string4 == string2:
            # Se string3 tem o prefixo e string4 é igual a string2
            if string1.startswith(prefixo):
                return True  # Chave encontrada
    return False  # Chave não encontrada




