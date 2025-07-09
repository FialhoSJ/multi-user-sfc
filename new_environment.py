import copy
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

from algorithms.networkUtils import get_available_shortest_path, calculate_computational_latency, calculate_latency_betwen_nodes

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

def latency_rounded(u, v, data):
    return int(round(data.get('latency', 0), 3) * 1000)

class NetworkEnv(gym.Env):
    def __init__(self, graph, valid_nodes, pesos):
        super().__init__()

        self.G = graph
        self.initial_resource_snapshot = {}
        self.valid_nodes = valid_nodes
        self.cached_paths = {}

        ### MODIFICADO: Atributos para gerenciar a lista de SFCs
        self.lista_SFCs = []
        self.current_sfc_idx = 0

        self.sfc = None
        self.services = None
        self.latency_request = None
        self.dst_node = None
        self.current_location = None
        self.service = None
        self.session_number = None

        self.latency_used = 0
        self.is_training = True
        self.servers_used = []
        self.allocation_results = {}
        self.success = False
        self.was_reused_in_step = False 
        
        self.cpu_factor = pesos.get("cpu", 1)
        self.cache_factor = pesos.get("cache", 1)
        self.band_factor = pesos.get("band", 1)
        self.latency_factor = pesos.get("latency", 1)
        
        ### CORRIGIDO: O espaço de observação volta a ter 6 métricas por nó.
        self.action_space = spaces.Discrete(len(self.valid_nodes))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            # shape=(len(self.valid_nodes) * 6,), # Corrigido de 5 para 6
            shape=(13*6,),
            dtype=np.float32
        )

    def set_sfcs_list(self, sfcs: list):
        """Define a lista de SFCs a serem alocadas no episódio."""
        if not sfcs:
            raise ValueError("A lista de SFCs não pode ser vazia.")
        self.lista_SFCs = sfcs

    ### NOVO: Método auxiliar para carregar uma SFC específica da lista.
    def _load_sfc(self, sfc_index):
        """Carrega os dados da SFC no índice fornecido e reinicia as métricas."""
        self.current_sfc_idx = sfc_index
        self.sfc = self.lista_SFCs[self.current_sfc_idx]
        
        # Carrega os dados específicos da nova SFC
        services = [item['name'] for item in self.sfc.vnfs_dict]
        self.services = list(reversed(services))
        self.latency_request = 9
        self.dst_node = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        self.session_number = self.sfc.id.split("_")[-1] # Usa o ID da SFC como número de sessão
        self.service = self.services[0]
        
        # Reinicia o estado para a nova SFC
        self.current_location = self.dst_node
        self.latency_used = 0
        self.servers_used.clear()
        mobile_device_id = self.sfc.dst_node
        self.valid_nodes[-1] = mobile_device_id
        self.update_bandwidth_required()

    def reset(self, seed=None, options=None):
        ### MODIFICADO: O reset agora prepara o ambiente para a lista de SFCs.
        if not self.lista_SFCs:
            raise ValueError("Chame set_sfcs_list() antes de resetar o ambiente.")
            
        # 1. Restaura o grafo ao seu estado inicial limpo
        if not self.initial_resource_snapshot:
            self._capture_initial_snapshot()

        snapshot_nodes = self.initial_resource_snapshot['nodes']
        for node_id, initial_state in snapshot_nodes.items():
            node = self.G.nodes[node_id]
            node['cpu_used'] = initial_state['cpu_used']
            node['cache_used'] = initial_state['cache_used']
            node['services']= copy.deepcopy(initial_state.get('services', {})),
            node['reuse']= copy.deepcopy(initial_state.get('reuse', []))

        snapshot_edges = self.initial_resource_snapshot['edges']
        for (u, v), initial_state in snapshot_edges.items():
            edge = self.G.edges[u, v]
            edge['bandwidth_used'] = initial_state['bandwidth_used']
            edge['services_in_transit'] = copy.deepcopy(initial_state.get('services_in_transit', {}))
        
        # 2. Reinicia métricas globais do episódio
        self.reward = 0
        self.total_reward = 0
        self.total_cost = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        
        # 3. Carrega a PRIMEIRA SFC da lista
        self._load_sfc(0)
        
        return self.get_normalized_state(), {}

    def step(self, action):
        mobile_device_id = self.sfc.dst_node
        self.valid_nodes[-1] = mobile_device_id
        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        
        server_to_allocate = self.valid_nodes[int(action)]
        self.was_reused_in_step = False

        success_alloc = self.allocate_resources_on_node(server_to_allocate)
        if not success_alloc:
            return self._fail_step('resource')
        

        if self.latency_used > self.latency_request:
            return self._fail_step('latency_comp')

        self.path = self._get_path_with_fallback(self.current_location, server_to_allocate)
        if not self.path:
            return self._fail_step('bandwidth_path')

        success_band, path_latency = self.allocate_bandwidth_along_path(self.path, self.bandwidth_required, self.service)
        if not success_band:
            return self._fail_step('bandwidth_alloc')

        self.latency_used += path_latency
        
        if self.latency_used > self.latency_request:
            return self._fail_step('latency_total')

        self.servers_used.append(server_to_allocate)
        # print(f"SFC {self.sfc.id}: {self.servers_used}")
        self.total_cost = self.calculate_total_cost(server_to_allocate, self.path)
        self.reward = -self.total_cost
        self.total_reward += self.reward

        self.current_location = server_to_allocate
        if not self.is_training:
            if self.sfc.id not in self.allocation_results:
                self.allocation_results[self.sfc.id] = {}
            self.allocation_results[self.sfc.id][self.service] = {'allocated_server': server_to_allocate, 'path': self.path, 'cost': self.total_cost}
        
        ### MUDANÇA PRINCIPAL: Lógica de fim de episódio e transição de SFC.
        done = False
        # Verifica se o serviço atual é o último da SFC ATUAL
        if self.service == self.services[-1]:
            # Se for, verifica se a SFC atual é a última da LISTA
            
            if self.current_sfc_idx == len(self.lista_SFCs) - 1:
                # Se for, o episódio inteiro terminou com sucesso
                done = True
                self.success = True
            else:
                # Se não for, carrega a PRÓXIMA SFC e o episódio continua
                self._load_sfc(self.current_sfc_idx + 1)
        else:
            # Se não for o último serviço, apenas avança para o próximo na mesma SFC
            current_index = self.services.index(self.service)
            self.service = self.services[current_index + 1]
            self.update_bandwidth_required()
        
        return self.get_normalized_state(), self.reward, done, False, {}

    # ... (O resto dos métodos auxiliares como _fail_step, allocate_*, calculate_*, etc., permanecem os mesmos da versão anterior)
    
    def _fail_step(self, reason):
        self.fail_reason = reason
        self.success = False
        self.reward = -2000 
        done = True
        return self.get_normalized_state(), self.reward, done, False, {}

    def _get_path_with_fallback(self, source, target):
        if source == target:
            return [source]
            
        cache_key = (source, target)
        if cache_key not in self.cached_paths:
            try:
                path = nx.dijkstra_path(self.G, source, target, weight=latency_rounded)
                self.cached_paths[cache_key] = path
            except nx.NetworkXNoPath:
                return None
        else:
            path = self.cached_paths[cache_key]
        
        if self.get_critical_link_bandwidth(path) >= self.bandwidth_required:
            return path
        
        path = get_available_shortest_path(self.G, source, target, self.bandwidth_required, rounded=True)
        if not path:
            return self.cached_paths.get(cache_key)
        else:
            self.cached_paths[cache_key] = path
            return path

    def allocate_resources_on_node(self, node_id):
        vnf = self.sfc.get_vnf_by_id(self.service)
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        node = self.G.nodes[node_id]

        if 'services' not in node: node['services'] = {}
        if 'reuse' not in node: node['reuse'] = []
        
        if (node['cpu_used'] + cpu_req > node['cpu_capacity']) or \
           (node['cache_used'] + cache_req > node['cache_capacity']):
            return False

        service_key = (self.service, self.session_number)
        is_shareable_service = self.is_shareable(self.service)
        


        if service_key in node['services']:
            node['services'][service_key]['copys'] += 1
            if not is_shareable_service:
                node['cpu_used'] += cpu_req
                node['cache_used'] += cache_req
            else:
                 self.was_reused_in_step = True
        else:
            node['services'][service_key] = {'cpu': cpu_req, 'cache': cache_req, 'copys': 1}
            node['cpu_used'] += cpu_req
            node['cache_used'] += cache_req
            
            if is_shareable_service:
                node['reuse'].append(vnf)
        
        return True

    def allocate_bandwidth_along_path(self, path, bandwidth_required, ms_name):
        if not path or len(path) < 2:
            return True, 0.0

        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                return False, 0.0

        total_latency = 0.0
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            latency = self._commit_bandwidth_on_link(u, v, vnf, bandwidth_required)
            total_latency += latency

        return True, total_latency
        
    def _commit_bandwidth_on_link(self, u, v, vnf, bandwidth_required):
        edge = self.G.edges[u, v]
        ms_name = vnf.id

        if 'services_in_transit' not in edge:
            edge['services_in_transit'] = {}

        if ms_name in edge['services_in_transit']:
            edge['services_in_transit'][ms_name]['copys'] += 1
        else:
            edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bandwidth_required}
        
        edge['bandwidth_used'] += bandwidth_required
        
        latency = calculate_latency_betwen_nodes(self.G, u, v, vnf)
        return latency

    def calculate_total_cost(self, server_id, path):
        node = self.G.nodes[server_id]
        cpu_capacity = node["cpu_capacity"] or 1
        cpu_cost = (node["cpu_used"] / cpu_capacity + 1) ** self.cpu_factor if not self.was_reused_in_step else 0
        cache_capacity = node["cache_capacity"] or 1
        cache_cost = (node["cache_used"] / cache_capacity + 1) ** self.cache_factor if not self.was_reused_in_step else 0
        self.latency_cost = 0
        bandwidth_cost = 0
        if len(path) >= 2:
            latency_request = self.latency_request or 1
            self.latency_cost = ((self.latency_used / latency_request) + 1) ** self.latency_factor
        return sum([cpu_cost, cache_cost, self.latency_cost, bandwidth_cost])

    def _capture_initial_snapshot(self):
        self.initial_resource_snapshot['nodes'] = {}
        for node_id, data in self.G.nodes(data=True):
            self.initial_resource_snapshot['nodes'][node_id] = {
                'cpu_used': data.get('cpu_used', 0),
                'cache_used': data.get('cache_used', 0),
                'services': copy.deepcopy(data.get('services', {})),
                'reuse': copy.deepcopy(data.get('reuse', []))
            }

        self.initial_resource_snapshot['edges'] = {}
        for u, v, data in self.G.edges(data=True):
            self.initial_resource_snapshot['edges'][(u, v)] = {
                'bandwidth_used': data.get('bandwidth_used', 0),
                'services_in_transit': copy.deepcopy(data.get('services_in_transit', {}))
            }

    def can_reuse_vnf_on_node(self, node_id, service_id):
        if not self.is_shareable(service_id):
            return False
        node = self.G.nodes[node_id]
        if 'reuse' not in node or not node['reuse']:
            return False
        for running_vnf in node['reuse']:
            if running_vnf.id == service_id:
                return True
        return False
    
    ### CORRIGIDO: get_normalized_state com 6 métricas, incluindo o flag de reuso.
    def get_normalized_state(self):
        if self.sfc is None: # Adiciona um guarda para o caso de falha antes do primeiro step
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        state_vectors = []
        current_vnf = self.sfc.get_vnf_by_id(self.service)
        base_cpu_req = current_vnf.get_cpu_request()
        base_cache_req = current_vnf.get_cache_request()

        for node_id in self.valid_nodes:
            node = self.G.nodes[node_id]
            can_reuse = self.can_reuse_vnf_on_node(node_id, self.service)
            
            effective_cpu_req = 0 if can_reuse else base_cpu_req
            effective_cache_req = 0 if can_reuse else base_cache_req
            
            cpu_capacity = node["cpu_capacity"] or 1
            cache_capacity = node["cache_capacity"] or 1
            
            if (base_cpu_req + node["cpu_used"]) > cpu_capacity or \
               (base_cache_req + node["cache_used"]) > cache_capacity:
                proj_cpu_cost = 1.0
                proj_cache_cost = 1.0
            else:
                proj_cpu_cost = (node["cpu_used"] + effective_cpu_req) / cpu_capacity
                proj_cache_cost = (node["cache_used"] + effective_cache_req) / cache_capacity
            
            path = self._get_path_with_fallback(self.current_location, node_id)
            path_latency = self.calculate_path_latency(path, self.service) if path else float('inf')
            latency_request = self.latency_request or 1
            proj_latency_cost = (self.latency_used + path_latency) / latency_request

            is_server_used = 1.0 if node_id in self.servers_used else 0.0
            cant_allocate = 1.0 if (proj_cpu_cost >= 1.0 or proj_cache_cost >= 1.0 or proj_latency_cost >= 1.0) else 0.0
            offers_reuse_flag = 1.0 if can_reuse else 0.0

            node_state = [
                min(proj_cpu_cost, 1.0), min(proj_cache_cost, 1.0),
                min(proj_latency_cost, 1.0), is_server_used,
                cant_allocate, offers_reuse_flag 
            ]
            state_vectors.extend(node_state)

        return np.array(state_vectors, dtype=np.float32)

    def is_shareable(self, service_name):
        return service_name.startswith(SHAREABLE_PREFIXES)
        
    def set_graph(self, graph):
        self.G = graph
        self._capture_initial_snapshot()

    def update_bandwidth_required(self):
        if not self.service: raise ValueError("Serviço atual não definido.")
        self.bandwidth_required = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth()

    def calculate_path_latency(self, path, ms_name):
        if not path or len(path) < 2: return 0.0
        total_latency = 0.0
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            total_latency += calculate_latency_betwen_nodes(self.G, u, v, vnf)
        return total_latency

    def get_critical_link_bandwidth(self, path):
        if not path or len(path) < 2: return float('inf')
        min_available = float('inf')
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            available = edge.get('bandwidth_capacity', 0) - edge.get('bandwidth_used', 0)
            min_available = min(min_available, available)
        return min_available