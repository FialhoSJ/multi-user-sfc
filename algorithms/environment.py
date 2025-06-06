import gymnasium as gym
import networkx as nx
import numpy as np
import copy
from algorithms.networkUtils import get_shortest_path_length,get_shortest_path,pre_get_single_source_minimum_latency_path, get_link_latency, get_available_shortest_path
from algorithms.networkUtils import calculate_computational_latency,calculate_latency_betwen_nodes
from gymnasium import spaces

# Definindo cores para logs
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')


class NetworkEnv(gym.Env):
    def __init__(self, graph='', server_resources=1, services=[0], service_requirements=1, latency_request=1, dst=1, valid_nodes=1, pesos=None):
        super().__init__()

        # Armazenando o grafo e outros parâmetros
        self.G = graph
        self.nodes_r = copy.deepcopy(self.G.nodes)  # Backup profundo do grafo
        self.links_values = copy.deepcopy(self.G._adj)
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
        self.success = True
        self.reuse = False
        self.servers_used = []
        self.allocation_results = {}

        # Espaços de observação e ação
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.valid_nodes) * 7 + 3,),  # Pode ser melhor modularizado no futuro
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.valid_nodes))

        # Cache de paths para otimizar cálculos repetitivos
        self.cached_paths = {}
        self.cached_all_shortest_paths = {}

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente e reinicializa os parâmetros, restaurando o estado original do grafo.
        """
        # Recarrega os nós do grafo a partir do backup
        self.G.nodes = copy.deepcopy(self.nodes_r)  # Restaura o backup do grafo
        self.G._adj = copy.deepcopy(self.links_values)

        # Reinicializa as variáveis de controle e custos
        self.latency_used = 0
        self.current_location = self.dst_node
        self.servers_used = []
        self.service = self.services[0]
        self._reset_costs()
        self.ac_total_cost = self.total_reward = self.reward = self.total_cost = 0
        self.success = True
        self.fail_reason = None
        self.allocation_results = {}
        self.path = None
        self.current_vnf = self.dst_vnf

        return self.get_normalized_state(), {}

    def step(self, action):
        """
        Realiza uma ação no ambiente, avaliando o sucesso ou falha.
        """
        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}. Deve estar entre 0 e {len(self.valid_nodes) - 1}.")
        done = False
        self.server = self.valid_nodes[int(action)]

        self.prev_vnf = self.current_vnf.get_previous_vnf()
        self.bandwidth_request = self.sfc.get_link_bandwidth_request(self.prev_vnf.id, self.current_vnf.id)
        self.path = get_available_shortest_path(self.G,self.server,self.current_location,self.bandwidth_request)

        self.reuse = self.check_reuse(self.current_vnf.id, self.server)

        # Atualiza a latência usada
        self.latency_used += calculate_latency_betwen_nodes(self.G,self.current_location,self.server,self.prev_vnf)

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
            self.servers_used.append(self.server)

        # Aloca recursos se necessário
        if not self.reuse:
            self._allocate_resources(self.server, self.service)

        # Atualiza a largura de banda utilizada ao longo do caminho
        for i in range(len(self.path) - 1):
            self.G._adj[self.path[i]][self.path[i + 1]]['bandwidth_used'] += self.service_requirements[self.service]['out_bw']

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
        else:
            aux = 1
        self.current_location = self.server

        return self.get_normalized_state(), self.reward, done, False, {}

    def calculate_total_cost(self):
        """
        Calcula o custo total considerando recursos, latência e largura de banda.
        """
        cpu_req = self.service_requirements[self.service]['CPU'] if not self.reuse else 0
        cache_req = self.service_requirements[self.service]['cache'] if not self.reuse else 0

        cpu_avail = self.G.nodes[self.server]["cpu_capacity"] - self.G.nodes[self.server]["cpu_used"]
        cache_avail = self.G.nodes[self.server]["cache_capacity"] - self.G.nodes[self.server]["cache_used"]
        e = 10 ** -6

        self.cpu_cost = (cpu_req / (cpu_avail + e) + 1) ** self.cpu_factor
        self.cache_cost = (cache_req / (cache_avail + e) + 1) ** self.cache_factor

        self.latency_cost = (self.latency_used/self.latency_request + 1) ** self.latency_factor
        min_band_avail = self.min_bandwidth_on_shortest_path(self.G, self.current_location, self.server)
        self.bandwidth_cost = (self.service_requirements[self.service]['out_bw'] / min_band_avail + 1) ** self.band_factor if min_band_avail != 0 else 0
        self.boot_cost = 0

        if self.server in self.servers_used:
            self.cpu_cost = self.cpu_cost**self.cpu_factor
            self.cache_cost *= self.cache_cost**self.cache_factor

        return sum([self.cpu_cost, self.cache_cost, self.latency_cost, self.bandwidth_cost, self.boot_cost])

    def check_reuse(self, service, server):
        """
        Verifica se o serviço pode ser reutilizado no servidor.
        """
        return service.startswith(SHAREABLE_PREFIXES) and (service, self.session_number) in self.G.nodes[server]['services']

    def min_bandwidth_on_shortest_path(self, graph, source, target):
        """
        Calcula a menor largura de banda disponível em um caminho mais curto.
        """
        path = get_available_shortest_path(self.G,self.server,self.current_location,self.bandwidth_request)

        min_bandwidth = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_attr = graph._adj[u][v]
            bw_available = edge_attr.get('bandwidth_capacity', 0) - edge_attr.get('bandwidth_used', 0)
            min_bandwidth = min(min_bandwidth, bw_available)

        return min_bandwidth

    def _has_resources(self, server, service):
        """
        Verifica se o servidor tem recursos suficientes para alocar o serviço.
        """

        node = self.G.nodes[server]
        cpu_required = self.service_requirements[service]['CPU']
        cache_required = self.service_requirements[service]['cache']
        return node['cpu_used'] + cpu_required < node['cpu_capacity'] and node['cache_used'] + cache_required < node['cache_capacity']

    def _allocate_resources(self, server, service):
        """
        Aloca os recursos necessários para o serviço no servidor.
        """
        self.G.nodes[server]['cpu_used'] += self.service_requirements[service]['CPU']
        self.G.nodes[server]['cache_used'] += self.service_requirements[service]['cache']

    def _reset_costs(self):
        """
        Reseta os custos relacionados aos recursos.
        """
        self.cpu_cost = self.cache_cost = self.latency_cost = self.bandwidth_cost = self.boot_cost = 0

    def _fail_step(self, reason):
        """
        Define o estado de falha quando uma condição não é atendida.
        """
        self.fail_reason = reason
        self.success = False
        state = self.get_normalized_state()
        self.total_cost = 1000  # Custo elevado para indicar falha
        self.reward = -self.total_cost
        done = True
        return state, self.reward, done, False, {}

    def get_normalized_state(self):
        """
        Retorna o estado normalizado do ambiente.
        """
        if self.current_location not in self.cached_all_shortest_paths:
            try:
                self.cached_all_shortest_paths[self.current_location] = nx.single_source_dijkstra_path(self.G, self.current_location, weight='weight')
            except nx.NetworkXNoPath:
                self.cached_all_shortest_paths[self.current_location] = {}

        state = []
        for node_id, res in self.G.nodes.items():
            if node_id in self.valid_nodes:
                state.extend([
                    (res['cpu_capacity'] - res['cpu_used']) / 100,
                    (res['cache_capacity'] - res['cache_used']) / 100,
                    1 if self.check_reuse(self.service, node_id) else 0
                ])
                state.append(1 if node_id in self.servers_used else 0)

        # Adicionando dados do serviço
        state.extend([
            self.service_requirements[self.service]["CPU"] / 100,
            self.service_requirements[self.service]["cache"] / 100
        ])

        aux = min(self.latency_used / self.latency_request, 1)
        state.append(aux)

        # Adicionando dados de largura de banda e latência
        for node_id in self.valid_nodes:
            if node_id in self.cached_all_shortest_paths[self.current_location]:
                path_to_node = self.cached_all_shortest_paths[self.current_location][node_id]
                cost_latency = (len(path_to_node) - 1 + self.latency_used) / self.latency_request
            else:
                cost_latency = 2  # custo elevado em caso de erro
            min_band = self.min_bandwidth_on_shortest_path(self.G, self.current_location, node_id)
            cost_band = self.service_requirements[self.service]['out_bw'] / min_band if min_band and min_band != 0 else float('inf')
            state.extend([min(cost_latency, 1), min(cost_band, 1)])

            # Verifica se o servidor tem recursos suficientes
            if not self._has_resources(node_id, self.service) or cost_latency > 1:
                state.append(0)
            else:
                state.append(1)

        return np.array(state, dtype=np.float32)
