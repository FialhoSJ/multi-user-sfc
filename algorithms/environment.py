import copy # Importamos copy no topo
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

from algorithms.networkUtils import get_available_shortest_path, calculate_computational_latency, calculate_latency_betwen_nodes

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')
VALOR_LATENCIA_SALTO = 1

# ... (funções auxiliares como latency_rounded permanecem iguais) ...
def latency_rounded(u, v, data):
    return int(round(data.get('latency', 0), 3) * 1000)

class NetworkEnv(gym.Env):
    def __init__(self, graph, valid_nodes, pesos):
        super().__init__()

        # SOLUÇÃO CORRETA: Usamos deepcopy no grafo inteiro, apenas uma vez.
        self.initial_G = copy.deepcopy(graph) 
        self.G = copy.deepcopy(graph)
        
        self.valid_nodes = valid_nodes
        self.cached_paths = {}
        # ... o resto do __init__ permanece igual ...
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
        self.action_space = spaces.Discrete(len(self.valid_nodes))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(valid_nodes)*6,), dtype=np.float32
        )


    def reset(self, seed=None, options=None):
        """
        CORRIGIDO E OTIMIZADO: Restaura o estado usando uma cópia profunda
        do grafo mestre. É correto e mais rápido que a implementação original.
        """
        if not self.lista_SFCs:
            raise ValueError("Chame set_sfcs_list() antes de resetar o ambiente.")
            
        # SOLUÇÃO CORRETA: A linha abaixo garante uma cópia totalmente independente.
        self.G = copy.deepcopy(self.initial_G)
        
        self.cached_paths.clear()

        # Reseta as variáveis de controle do episódio
        self.reward = 0
        self.total_reward = 0
        self.total_cost = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        
        self._load_sfc(0)
        
        return self.get_normalized_state(), {}

    def set_graph(self, graph):
        """
        Define o grafo de trabalho e guarda uma cópia profunda como mestre.
        """
        # SOLUÇÃO CORRETA
        self.initial_G = copy.deepcopy(graph)
        self.G = copy.deepcopy(graph)
        self.cached_paths.clear()

    # Nenhuma outra alteração é necessária. O restante do código da classe
    # (step, _release_resources_for_sfc, etc.) permanece o mesmo da versão anterior.
    # ... (cole o resto dos métodos da classe aqui) ...
    def set_sfcs_list(self, sfcs: list):
        """Define a lista de SFCs a serem alocadas no episódio."""
        if not sfcs:
            raise ValueError("A lista de SFCs não pode ser vazia.")
        self.lista_SFCs = sfcs

    def _release_resources_for_sfc(self, sfc_id_to_release):
        if not sfc_id_to_release:
            return
        for node_id, node_data in self.G.nodes(data=True):
            # sfc_object_to_remove = next((sfc for sfc in node_data.get('sfcs_list', []) if sfc.id == sfc_id_to_release), None)
            sfc_object_to_remove = self.sfc
            if "sfcs_list" not in node_data or not node_data["sfcs_list"]:
                continue
            if node_data["type"] == 'router' or sfc_object_to_remove not in node_data["sfcs_list"] :
                continue
            vnf_names_in_sfc = {item['name'] for item in sfc_object_to_remove.vnfs_dict}
            session_to_release = sfc_object_to_remove.id.split("_")[-1]
            service_keys_to_remove = [k for k in node_data.get('services', {}) if k[1] == session_to_release and k[0] in vnf_names_in_sfc]
            for service_key in service_keys_to_remove:
                service_name = service_key[0]
                service_info = node_data['services'][service_key]
                release_physical_resources = True
                if self.is_shareable(service_name):
                    if sum(1 for k in node_data['services'] if k[0] == service_name and k[1] != session_to_release) > 0:
                        release_physical_resources = False
                if release_physical_resources:
                    # print(f"removeu {self.sfc.id} do nó {node_id}")
                    node_data['cpu_used'] -= service_info.get('cpu', 0)
                    node_data['cache_used'] -= service_info.get('cache', 0)
                    if self.is_shareable(service_name) and 'reuse' in node_data:
                        vnf_to_remove = next((vnf for vnf in node_data['reuse'] if vnf.id.startswith(service_name.split('_')[0])), None)
                        if vnf_to_remove:
                            node_data['reuse'].remove(vnf_to_remove)
                del node_data['services'][service_key]
            node_data['sfcs_list'].remove(sfc_object_to_remove)
        for u, v, edge_data in self.G.edges(data=True):
            if 'services_in_transit' in edge_data:
                traffic_to_remove = [k for k in edge_data['services_in_transit'] if sfc_id_to_release in k]
                for key in traffic_to_remove:
                    transit_info = edge_data['services_in_transit'][key]
                    edge_data['bandwidth_used'] -= transit_info.get('bw_used', 0) * transit_info.get('copys', 1)
                    del edge_data['services_in_transit'][key]

    def _load_sfc(self, sfc_index):
        sfc_to_load = self.lista_SFCs[sfc_index]
        self._release_resources_for_sfc(sfc_to_load.id)
        self.current_sfc_idx = sfc_index
        self.sfc = sfc_to_load
        services = [item['name'] for item in self.sfc.vnfs_dict]
        self.services = list(reversed(services))
        self.latency_request = 10
        self.dst_node = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        self.session_number = self.sfc.id.split("_")[-1]
        self.service = self.services[0]
        self.current_location = self.dst_node
        self.latency_used = 0
        self.servers_used.clear()
        mobile_device_id = self.sfc.dst_node
        self.valid_nodes[-1] = mobile_device_id
        self.update_bandwidth_required()

    def step(self, action):
        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        server_to_allocate = self.valid_nodes[int(action)]
        self.was_reused_in_step = False
        # print(f"Nó {server_to_allocate} || cpu_used: {self.G.nodes[server_to_allocate]["cpu_used"]} || cache_used: {self.G.nodes[server_to_allocate]["cache_used"]}")
        # vnf = self.sfc.get_vnf_by_id(self.service)
        # cpu_req = vnf.get_cpu_request()
        # cache_req = vnf.get_cache_request()
        # print(f"Sevirço {vnf.id} || cpu_request: {cpu_req} || cache_request: {cache_req}")
        if not self.allocate_resources_on_node(server_to_allocate):
            vnf = self.sfc.get_vnf_by_id(self.service)
            print(f"""Falha: Resource || Node {server_to_allocate} || CPU_used : {self.G.nodes[server_to_allocate]["cpu_used"]} CPU_required : {vnf.get_cpu_request()}
            cache_used : {self.G.nodes[server_to_allocate]["cache_used"]} cache_required : {vnf.get_cache_request()}""")
            return self._fail_step('resource')
        path = self._get_path_with_fallback(self.current_location, server_to_allocate)
        if not path:
            return self._fail_step('bandwidth')
        success_band, path_latency = self.allocate_bandwidth_along_path(path, self.bandwidth_required, self.service)
        if not success_band:
            return self._fail_step('bandwidth')
        self.latency_used += path_latency
        if self.latency_used > self.latency_request:
            return self._fail_step('latency')
        self.servers_used.append(server_to_allocate)
        self.total_cost = self.calculate_total_cost(server_to_allocate, path)
        # if self.total_cost == 0:
        #     self.total_cost = self.calculate_total_cost(server_to_allocate, path)
        self.reward = -self.total_cost
        self.total_reward += self.reward
        self.current_location = server_to_allocate
        if not self.is_training:
            self.allocation_results[self.service] = {'allocated_server': server_to_allocate, 'path': path, 'cost': self.total_cost}
        done = False
        if self.service == self.services[-1]:
            if self.current_sfc_idx == len(self.lista_SFCs) - 1:
                # print("Alocou tudo com sucesso.")
                done = True
                self.success = True
            else:
                self._load_sfc(self.current_sfc_idx + 1)
        else:
            current_index = self.services.index(self.service)
            self.service = self.services[current_index + 1]
            self.update_bandwidth_required()
        return self.get_normalized_state(), self.reward, done, False, {}

    def allocate_resources_on_node(self, node_id):
        vnf = self.sfc.get_vnf_by_id(self.service)
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        node = self.G.nodes[node_id]
        if 'services' not in node: node['services'] = {}
        if 'reuse' not in node: node['reuse'] = []
        if 'sfcs_list' not in node: node['sfcs_list'] = []
        is_reusable = self.can_reuse_vnf_on_node(node_id, self.service)
        effective_cpu_req = 0 if is_reusable else cpu_req
        effective_cache_req = 0 if is_reusable else cache_req
        if (node['cpu_used'] + cpu_req > node['cpu_capacity']) or \
           (node['cache_used'] + cache_req > node['cache_capacity']):
            return False
        
        effective_cpu_req = 0 if is_reusable else cpu_req
        effective_cache_req = 0 if is_reusable else cache_req
        
        service_key = (self.service, self.session_number)
        if not is_reusable:
            node['cpu_used'] += effective_cpu_req
            node['cache_used'] += effective_cache_req
            if self.is_shareable(self.service):
                node['reuse'].append(vnf)
        self.was_reused_in_step = is_reusable
        if service_key not in node['services']:
            node['services'][service_key] = {'cpu': cpu_req, 'cache': cache_req, 'copys': 1}
        else:
            node['services'][service_key]['copys'] += 1
        if self.sfc not in node['sfcs_list']:
            node['sfcs_list'].append(self.sfc)
        return True

    def _fail_step(self, reason):
        # print(f"Falha na alocação da SFC {self.sfc.id}: {reason}")
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
                path = nx.shortest_path(self.G, source, target)
                self.cached_paths[cache_key] = path
            except nx.NetworkXNoPath:
                return None
        else:
            path = self.cached_paths[cache_key]
        if self.get_critical_link_bandwidth(path) >= self.bandwidth_required:
            return path
        path_alternative = get_available_shortest_path(self.G, source, target, self.bandwidth_required, rounded=True)
        if not path_alternative:
            return self.cached_paths.get(cache_key)
        else:
            self.cached_paths[cache_key] = path_alternative
            return path_alternative

    def allocate_bandwidth_along_path(self, path, bandwidth_required, ms_name):
        if not path or len(path) < 2:
            return True, 0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                return False, 0
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            self._commit_bandwidth_on_link(u, v, vnf, bandwidth_required)
        return True, (len(path) - 1) * VALOR_LATENCIA_SALTO
        
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

    def can_reuse_vnf_on_node(self, node_id, service_id):
        if not self.is_shareable(service_id):
            return False
        node = self.G.nodes[node_id]
        if 'reuse' not in node or not node['reuse']:
            return False
        return any(vnf.id.startswith(service_id.split('_')[0]) for vnf in node['reuse'])
    
    def get_normalized_state(self):
        if self.sfc is None:
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
            if (node["cpu_used"] + base_cpu_req) > cpu_capacity or (node["cache_used"] + base_cache_req) > cache_capacity:
                proj_cpu_cost = 1.0
                proj_cache_cost = 1.0
            else:
                proj_cpu_cost = (node["cpu_used"] + effective_cpu_req) / cpu_capacity
                proj_cache_cost = (node["cache_used"] + effective_cache_req) / cache_capacity
            path_latency = (len(nx.shortest_path(self.G, source=self.current_location, target=node_id)) - 1) * VALOR_LATENCIA_SALTO
            latency_request = self.latency_request or 1
            proj_latency_cost = (self.latency_used + path_latency) / latency_request
            is_server_used = 1.0 if node_id in self.servers_used else 0.0
            cant_allocate = 1.0 if (proj_cpu_cost >= 1.0 or proj_cache_cost >= 1.0 or proj_latency_cost > 1.0) else 0.0
            offers_reuse_flag = 1.0 if can_reuse else 0.0
            node_state = [min(proj_cpu_cost, 1.0), min(proj_cache_cost, 1.0), min(proj_latency_cost, 1.0), is_server_used, cant_allocate, offers_reuse_flag]
            state_vectors.extend(node_state)
        return np.array(state_vectors, dtype=np.float32)

    def is_shareable(self, service_name):
        return service_name.startswith(SHAREABLE_PREFIXES)

    def update_bandwidth_required(self):
        if not self.service: raise ValueError("Serviço atual não definido.")
        self.bandwidth_required = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth()

    def get_critical_link_bandwidth(self, path):
        if not path or len(path) < 2: return float('inf')
        min_available = float('inf')
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            available = edge.get('bandwidth_capacity', 0) - edge.get('bandwidth_used', 0)
            min_available = min(min_available, available)
        return min_available