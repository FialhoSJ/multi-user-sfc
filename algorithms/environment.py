import copy
import gymnasium as gym
import numpy as np
from algorithms.networkUtils import get_shortest_path, pre_get_single_source_minimum_latency_path, calcular_latencia_total, get_available_shortest_path
from algorithms.networkUtils import calculate_computational_latency, calculate_latency_betwen_nodes
from gymnasium import spaces
import time

# Definindo cores para logs
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

class NetworkEnv(gym.Env):
    def __init__(self, graph='',services=[0], service_requirements=1, latency_request=1, dst=1, valid_nodes=1, pesos=None, sfc=None):
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
        # self.sfc = sfc  # Certifique-se de que o objeto sfc é passado para o ambiente
       
        # Espaços de observação e ação
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.valid_nodes) * 6 + 3,),  # Pode ser melhor modularizado no futuro
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.valid_nodes))

        # Cache de paths para otimizar cálculos repetitivos
        self.cached_paths = {}
        
        # self.cached_all_shortest_paths = {}

    # def set_dst_vnf(self,dst_vnf):
    #     self.dst_vnf = dst_vnf
    #     self.current_vnf = self.dst_vnf
    #     self.prev_vnf = self.current_vnf.get_previous_vnf()

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente e reinicializa os parâmetros, restaurando o estado original do grafo.
        """
        # start_time = time.time()  # Início da medição de tempo
        # Recarrega os nós do grafo a partir do backup
        # print(f"Tempo para deep copy do grafo: {time.time() - start_time:.4f} segundos")  # Fim da medição de tempo
        self.G._adj = copy.deepcopy(self.network_links_backup)
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
        self.server_resources = copy.deepcopy(self.server_resources_backup)
        # self.set_dst_vnf(self.dst_vnf)
        # self.bandwidth_request = self.sfc.get_link_bandwidth_request(self.prev_vnf.id, self.current_vnf.id)
        self.bandwidth_request = self.service_requirements[self.service]["in_bw"]

        return self.get_normalized_state(), {}

    def step(self, action):
        """
        Realiza uma ação no ambiente, avaliando o sucesso ou falha.
        """

        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        done = False
        self.server = self.valid_nodes[int(action)]

        # self.prev_vnf = self.current_vnf.get_previous_vnf()
        # self.bandwidth_request = self.service_requirements[self.services[self.services.index(self.service)+1]]["out_bw"]

        # Verifica se o caminho já foi calculado e está em cache
        if (self.current_location, self.server) in self.cached_paths:
            self.path = self.cached_paths[(self.current_location, self.server)]
            if not self.is_bandwidth_sufficient(self.path, self.bandwidth_request):
                # Se a largura de banda não for suficiente, calcula o caminho com largura de banda disponível
                self.path = get_available_shortest_path(self.G, self.current_location, self.server, self.bandwidth_request)
                self.cached_paths[(self.current_location, self.server)] = self.path
        else:
            # Se não, calcula o menor caminho sem considerar a largura de banda
            self.path = get_available_shortest_path(self.G, self.current_location, self.server, self.bandwidth_request)
            # Armazena o caminho no cache
            self.cached_paths[(self.current_location, self.server)] = self.path

        self.reuse = self.check_reuse(self.service, self.server)

        # Atualiza a latência usada
        self.latency_used += calcular_latencia_total(self.path, self.G)

        # Verifica se os limites de latência ou largura de banda são atingidos
        if self.latency_used > self.latency_request:
            self.total_cost = self.calculate_total_cost()
            return self._fail_step('latency')
        elif not self.path:
            self.total_cost = self.calculate_total_cost()
            return self._fail_step('bandwidth')
        else:
            # Verifica se há recursos suficientes
            if not self.reuse and not self._has_resources(self.server, self.service):
                self.total_cost = self.calculate_total_cost()
                return self._fail_step('resource')

            self.total_cost = self.calculate_total_cost()
            self.ac_total_cost += self.total_cost
            self.reward = -self.total_cost
            self.total_reward += self.reward
            self.servers_used.append(self.server)  # Usar set para adicionar o servidor

        # Aloca recursos se necessário
        if not self.reuse:
            self._allocate_resources(self.server, self.service)

        # Atualiza a largura de banda utilizada ao longo do caminho
        for i in range(len(self.path) - 1):
            self.G._adj[self.path[i]][self.path[i + 1]]['bandwidth_used'] += self.bandwidth_request

        # Atualiza resultados da alocação se não estiver em treinamento
        if not self.is_training:
            self.allocation_results[self.service] = {
                'allocated_server': self.server,
                'path': self.path,
                'cost': self.total_cost
            }

        done = self.service == self.services[-1]
        if not done:
            self.service = self.services[self.services.index(self.service) + 1]
            self.update_bandwidth_request()
        else:
            aux = 1
            self.success=True
        self.current_location = self.server
        # self.current_vnf = self.prev_vnf

        return self.get_normalized_state(), self.reward, done, False, {}
    
    def set_graph(self,graph):
        self.network_links_backup = graph._adj
        self.G = graph

    def set_server_resources(self,server_resources):
        self.server_resources_backup = server_resources
        self.server_resources = copy.deepcopy(server_resources)

    def update_bandwidth_request(self):
        if not self.service_requirements or not self.service:
            raise ValueError("Variaveis não instanciadas")
        self.bandwidth_request = self.service_requirements[self.service]["in_bw"]

    def set_dst_node(self,dst_node):
        if not dst_node:
            raise ValueError("dst_node None")
        self.dst_node = dst_node
        self.current_location = dst_node

    def _has_resources(self,server, service):
        cpu_required = self.service_requirements[service]["cpu"]
        cache_required = self.service_requirements[service]["cpu"]
        cpu_aval = self.server_resources[server]["cpu_free"]
        cache_aval = self.server_resources[server]["cache_free"]
        return cpu_aval >= cpu_required+1 and cache_aval>= cache_required+1
    
    def _fail_step(self, reason):
        self.fail_reason = reason
        self.success = False
        state = self.get_normalized_state()
        self.total_cost = 2000
        self.reward = -self.total_cost
        # self.ac_total_cost += self.total_cost
        done = True

        # if done and  self.min_cost > self.ac_total_cost:
        #     self.min_cost = self.total_cost
        # self.print_info_of_allocation()
        return state, self.reward, done, False, {}


    
    def _allocate_resources(self,server, service):
        self.server_resources[server]["cpu_used"]+=self.service_requirements[service]["cpu"]
        self.server_resources[server]["cache_used"]+=self.service_requirements[service]["cache"]

        self.server_resources[server]["cpu_free"]-=self.service_requirements[service]["cpu"]
        self.server_resources[server]["cache_free"]-=self.service_requirements[service]["cache"]





    def is_bandwidth_sufficient(self, path, bandwidth_request):
        """
        Verifica se há largura de banda suficiente ao longo do caminho.
        """
        min_bandwidth = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_attr = self.G._adj[u][v]
            bw_available = edge_attr.get('bandwidth_capacity', 0) - edge_attr.get('bandwidth_used', 0)
            min_bandwidth = min(min_bandwidth, bw_available)
        return min_bandwidth >= bandwidth_request

    def calculate_total_cost(self):
        """
        Calcula o custo total considerando recursos, latência e largura de banda.
        """
        cpu_req = self.service_requirements[self.service]["cpu"] if not self.reuse else 0
        cache_req = self.service_requirements[self.service]["cache"] if not self.reuse else 0

        cpu_avail = self.server_resources[self.server]["cpu_free"]
        cache_avail = self.server_resources[self.server]["cache_free"]
        e = 10 ** -6

        self.cpu_cost = (cpu_req / (cpu_avail + e) + 1) ** self.cpu_factor
        self.cache_cost = (cache_req / (cache_avail + e) + 1) ** self.cache_factor

        self.latency_cost = (self.latency_used / self.latency_request + 1) ** self.latency_factor
        self.bandwidth_cost = 0
        self.boot_cost = 0

        if self.server in self.servers_used:
            self.cpu_cost *= self.cpu_factor
            self.cache_cost *= self.cache_factor

        return sum([self.cpu_cost, self.cache_cost, self.latency_cost, self.bandwidth_cost, self.boot_cost])

    def check_reuse(self, service, server):
        """
        Verifica se o serviço pode ser reutilizado no servidor.
        """
        for sf_shareable in SHAREABLE_PREFIXES:
            if service.startswith(sf_shareable):
                if verificar_chave(self.G.nodes[server]['services'], (service, self.session_number), sf_shareable):
                    return True
        return False

    # def min_bandwidth_on_shortest_path(self, graph, source, target):
    #     """
    #     Calcula a menor largura de banda disponível em um caminho mais curto.
    #     """
    #     path = get_available_shortest_path(self.G, target, source, self.bandwidth_request)

    #     min_bandwidth = float('inf')
    #     for i in range(len(path) - 1):
    #         u, v = path[i], path[i + 1]
    #         edge_attr = graph._adj[u][v]
    #         bw_available = edge_attr.get('bandwidth_capacity', 0) - edge_attr.get('bandwidth_used', 0)
    #         min_bandwidth = min(min_bandwidth, bw_available)

    #     return min_bandwidth

    def get_normalized_state(self):
        """
        Retorna o estado normalizado do ambiente.
        """
        # if self.current_location not in self.cached_all_shortest_paths:
        #     try:
        #         self.cached_all_shortest_paths[self.current_location] = nx.single_source_dijkstra_path(self.G, self.current_location, weight='weight')
        #     except nx.NetworkXNoPath:
        #         self.cached_all_shortest_paths[self.current_location] = {}

        state = []
        # for node_id, res in self.server_resources:
        #     if node_id in self.valid_nodes:
        #         state.extend([
        #             (res['cpu_capacity'] - res['cpu_used']) / 100,
        #             (res['cache_capacity'] - res['cache_used']) / 100,
        #             1 if self.check_reuse(self.service, node_id) else 0
        #         ])
        #         state.append(1 if node_id in self.servers_used else 0)

        for node_id in self.valid_nodes:
            state.extend([
                (self.server_resources[node_id]["cpu_free"]/100),
                (self.server_resources[node_id]["cache_free"]/100),
                1 if self.check_reuse(self.service, node_id) else 0
            ])
            state.append(1 if node_id in self.servers_used else 0)

        # Adicionando dados do serviço
        state.extend([
            self.service_requirements[self.service]["cpu"] / 100,
            self.service_requirements[self.service]["cache"] / 100
        ])

        aux = min(self.latency_used / self.latency_request, 1)
        state.append(aux)

        # Adicionando dados da latência
        for node_id in self.valid_nodes:
            if (self.current_location,node_id) in self.cached_paths:
                path_to_node = self.cached_paths[(self.current_location, node_id)]
                if not self.is_bandwidth_sufficient(path_to_node,self.bandwidth_request):
                    path_to_node = get_available_shortest_path(self.G,self.current_location, node_id, self.bandwidth_request)
                    self.cached_paths[(self.current_location, node_id)] = path_to_node
            else:
                path_to_node= get_available_shortest_path(self.G, self.current_location, node_id, self.bandwidth_request)
                self.cached_paths[(self.current_location, node_id)] = path_to_node

            cost_latency = calcular_latencia_total(path_to_node, self.G) / self.latency_request if path_to_node else 1
            state.extend([min(cost_latency, 1)])

            # Verifica se o servidor tem recursos suficientes
            if (not self._has_resources(node_id, self.service) and not self.check_reuse(self.service, node_id))  or cost_latency >= 1:
                state.append(0)
            else:
                state.append(1)

        return np.array(state, dtype=np.float32)

def verificar_chave(dicionario, chave, prefixo):
    string1, string2 = chave  # Desempacotando a tupla (string1, string2)
    
    # Verificar se a chave existe no dicionário com as condições
    for (string3, string4), valor in dicionario.items():
        if string3.startswith(prefixo) and string4 == string2:
            # Se string3 tem o prefixo e string4 é igual a string2
            if string1.startswith(prefixo):
                return True  # Chave encontrada
    return False  # Chave não encontrada
