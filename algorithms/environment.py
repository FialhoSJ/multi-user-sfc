import gymnasium as gym
import networkx as nx
import numpy as np
import copy
import time
from gymnasium import spaces

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

class NetworkEnv(gym.Env):
    def __init__(self, graph='', substrate_network=1, server_resources=1, services=[0], service_requirements=1, latency_request=1, dst=1, valid_nodes=1, pesos=None):
        super().__init__()

        self.G = graph
        # MUDANÇA AQUI =============================
        # Evita deepcopy pesado na inicialização; copia rasa (shallow copy)
        self.G_backup = graph.copy() if hasattr(graph, 'copy') else graph
        # MUDANÇA AQUI =============================

        self.server_resources_backup = server_resources
        self.server_resources = copy.deepcopy(self.server_resources_backup)
        self.show_allocation = True
        self.valid_nodes = valid_nodes
        self.session_number = ''

        # Parâmetros dos serviços
        self.service_requirements = service_requirements
        self.services = services
        self.service = services[0]
        self.dst_node = dst
        self.current_location = dst
        self.latency_request = latency_request
        self.latency_used = 0

        # Parâmetros de custo
        self.cpu_factor = pesos["cpu"]
        self.cache_factor = pesos["cache"]
        self.band_factor = pesos["band"]
        self.latency_factor = pesos["latency"]
        self.boot_factor = 0

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
            shape=(len(self.valid_nodes) * 7 + 3,),  # pode ser melhor modularizado no futuro
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.valid_nodes))

        # Cache de paths para evitar recalculá-los frequentemente
        self.cached_paths = {}

        # MUDANÇA AQUI =============================
        # Cache para caminhos mínimos a partir da localização atual para todos os nós válidos
        self.cached_all_shortest_paths = {}
        # MUDANÇA AQUI =============================

    def set_server_resources(self, server_resources):
        self.server_resources_backup = server_resources
        self.server_resources = copy.deepcopy(self.server_resources_backup)

    def reset(self, seed=None, options=None):
        self.server_resources = copy.deepcopy(self.server_resources_backup)
        # MUDANÇA AQUI =============================
        # Usa copy superficial para evitar overhead pesado
        self.G = self.G_backup.copy() if hasattr(self.G_backup, 'copy') else self.G_backup
        # Limpa caches de caminhos
        self.cached_paths.clear()
        self.cached_all_shortest_paths.clear()
        # MUDANÇA AQUI =============================

        self.latency_used = 0
        self.current_location = copy.copy(self.dst_node)
        self.servers_used = []
        self.service = self.services[0]
        self._reset_costs()
        self.ac_total_cost = 0
        self.total_reward = 0
        self.reward = 0
        self.total_cost = 0
        self.success = True
        self.fail_reason = None
        self.allocation_results = {}
        self.path = None

        return self.get_normalized_state(), {}

    def step(self, action):
        if not 0 <= action < (len(self.valid_nodes)):
            raise ValueError(f"Ação inválida: {action}. Deve estar entre 0 e {len(self.valid_nodes) - 1}.")
        done = False

        self.server = self.valid_nodes[int(action)]

        # Verificar se o caminho já está calculado
        if (self.current_location, self.server) not in self.cached_paths:
            self.cached_paths[(self.current_location, self.server)] = nx.shortest_path(self.G, self.current_location, self.server, weight='weight')

        self.path = self.cached_paths[(self.current_location, self.server)]

        if (self.service, self.session_number) in self.server_resources[self.server]['reuse']:
            self.reuse = True
        else:
            self.reuse = False

        self.latency_used += len(self.path) - 1

        if self.latency_used > self.latency_request:
            self.total_cost = self.calculate_total_cost()
            return self._fail_step('latency')
        elif self.min_bandwidth_on_shortest_path(self.G, self.current_location, self.server) < self.service_requirements[self.service]["out_bw"]:
            self.total_cost = self.calculate_total_cost()
            return self._fail_step('bandwith')
        else:
            if not self.reuse:
                if not self._has_resources(self.server, self.service):
                    self.total_cost = self.calculate_total_cost()
                    return self._fail_step('resource')

            self.total_cost = self.calculate_total_cost()
            self.ac_total_cost += self.total_cost
            self.reward = -self.total_cost
            self.total_reward += self.reward

            self.servers_used.append(self.server)

        if not self.reuse:
            self._allocate_resources(self.server, self.service)

        for i in range(len(self.path) - 1):
            self.G._adj[self.path[i]][self.path[i + 1]]['bandwidth_used'] += self.service_requirements[self.service]['out_bw']

        if not self.is_training:
            self.allocation_results[self.service] = {
                'allocated_server': self.server,
                'path': self.path,
                'cost': self.total_cost
            }

        done = self.service == self.services[-1]
        if not done:
            self.service = self.services[self.services.index(self.service) + 1]

        self.current_location = self.server

        return self.get_normalized_state(), self.reward, done, False, {}

    # ===================
    # Custo e recursos
    # ===================

    def calculate_total_cost(self):
        cpu_req = 0 if self.reuse else self.service_requirements[self.service]['CPU']
        cache_req = 0 if self.reuse else self.service_requirements[self.service]['cache']
        if cpu_req == 0:
            cpu_req = self.service_requirements[self.service]['CPU'] * 0.3
            cache_req = self.service_requirements[self.service]['cache'] * 0.3

        cpu_avail = self.server_resources[self.server]["cpu_free"]
        cache_avail = self.server_resources[self.server]["cache_free"]
        e = 10 ** -6
        self.cpu_cost = (cpu_req / (cpu_avail + e) + 1) ** self.cpu_factor
        self.cache_cost = (cache_req / (cache_avail + e) + 1) ** self.cache_factor

        self.latency_cost = (len(self.path) - 1) ** self.latency_factor
        min_band_avail = self.min_bandwidth_on_shortest_path(self.G, self.current_location, self.server)
        self.bandwidth_cost = (self.service_requirements[self.service]['out_bw'] / min_band_avail + 1) ** self.band_factor if min_band_avail != 0 else 0
        self.boot_cost = 0

        if self.server in self.servers_used:
            self.cpu_cost *= self.cpu_factor
            self.cache_cost *= self.cache_factor

        return sum([self.cpu_cost, self.cache_cost, self.latency_cost, self.bandwidth_cost, self.boot_cost])

    # Otimizado para usar cache pré-calculada na maior parte do tempo
    def min_bandwidth_on_shortest_path(self, graph, source, target):
        # MUDANÇA AQUI =============================
        # Reutiliza o caminho já cacheado ou calcula se não existir
        if (source, target) in self.cached_paths:
            path = self.cached_paths[(source, target)]
        else:
            try:
                path = nx.shortest_path(graph, source=source, target=target, weight='latency')
                self.cached_paths[(source, target)] = path
            except nx.NetworkXNoPath:
                print(f"Nenhum caminho entre {source} e {target}")
                return None
        # MUDANÇA AQUI =============================

        min_bandwidth = float('inf')
        adj = graph._adj

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_attr = adj[u][v]
            bw_available = edge_attr.get('bandwidth_capacity', 0) - edge_attr.get('bandwidth_used', 0)
            if bw_available < min_bandwidth:
                min_bandwidth = bw_available

        return min_bandwidth

    def _has_resources(self, server, service):
        cpu_ok = self.server_resources[server]['cpu_free'] >= (self.service_requirements[service]['CPU'] + 1)
        cache_ok = self.server_resources[server]['cache_free'] >= (self.service_requirements[service]['cache'] + 1)
        return cpu_ok and cache_ok

    def _allocate_resources(self, server, service):
        self.server_resources[server]['cpu_used'] += self.service_requirements[service]['CPU']
        self.server_resources[server]['cache_used'] += self.service_requirements[service]['cache']
        self.server_resources[server]['cpu_free'] -= self.service_requirements[service]['CPU']
        self.server_resources[server]['cache_free'] -= self.service_requirements[service]['cache']

    def _reset_costs(self):
        self.cpu_cost = self.cache_cost = self.latency_cost = self.bandwidth_cost = self.boot_cost = 0

    def _fail_step(self, reason):
        self.fail_reason = reason
        self.success = False
        state = self.get_normalized_state()
        self.total_cost = 1000
        self.reward = -self.total_cost
        done = True
        return state, self.reward, done, False, {}

    # ===================
    # Estado do agente
    # ===================

    def get_normalized_state(self):
        # MUDANÇA AQUI =============================
        # Pré-calcular todos os caminhos mínimos de current_location para todos valid_nodes para evitar calcular vários nx.shortest_path
        if self.current_location not in self.cached_all_shortest_paths:
            try:
                self.cached_all_shortest_paths[self.current_location] = nx.single_source_dijkstra_path(self.G, self.current_location, weight='weight')
            except nx.NetworkXNoPath:
                self.cached_all_shortest_paths[self.current_location] = {}
        # MUDANÇA AQUI =============================

        state = []

        for node_id, res in self.server_resources.items():
            if node_id in self.valid_nodes:
                state.extend([
                    res['cpu_free'] / 100 if res['cpu_free'] >= 0 else 0,
                    res['cache_free'] / 100 if res['cache_free'] >= 0 else 0,
                    1 if self.service in res['reuse'] else 0
                ])
                if node_id in self.servers_used:
                    state.append(1)
                else:
                    state.append(0)

        state.extend([
            self.service_requirements[self.service]["CPU"] / 100,
            self.service_requirements[self.service]["cache"] / 100
        ])

        aux = min(self.latency_used / self.latency_request, 1)
        state.append(aux)

        for node_id in self.valid_nodes:
            # MUDANÇA AQUI =============================
            if node_id in self.cached_all_shortest_paths[self.current_location]:
                path_to_node = self.cached_all_shortest_paths[self.current_location][node_id]
                cost_latency = (len(path_to_node) - 1 + self.latency_used) / self.latency_request
            else:
                # Caso não tenha caminho, considera custo alto
                cost_latency = 2  # maior que 1 para invalidar
            
            min_band = self.min_bandwidth_on_shortest_path(self.G, self.current_location, node_id)
            cost_band = self.service_requirements[self.service]['out_bw'] / min_band if min_band and min_band != 0 else float('inf')
            cost_band = min(cost_band, 1)

            state.append(min(cost_latency, 1))
            state.append(cost_band)
            # MUDANÇA AQUI =============================

            if (not self._has_resources(node_id, self.service) and not (self.service, self.session_number) in self.server_resources[node_id]['reuse']) or cost_latency > 1:
                state.append(0)
            else:
                state.append(1)

        return np.array(state, dtype=np.float32)
