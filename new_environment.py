import copy 
import gymnasium as gym
import numpy as np
import networkx as nx
import random 

from gymnasium import spaces
from utils.david.utils_rl import add_mobile_user_to_graph
from utils.david.utils_rl import subtrair_valor_padrao
from algorithms.networkUtils import get_available_shortest_path, calculate_computational_latency, calculate_latency_betwen_nodes

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')
VALOR_LATENCIA_SALTO = 1

def latency_rounded(u, v, data):
    return int(round(data.get('latency', 0), 3) * 1000)

class NetworkEnv(gym.Env):
    def __init__(self, graph, valid_nodes, pesos, all_sfc_lists: list):
        super().__init__()

        self.initial_G = copy.deepcopy(graph) 
        self.G = copy.deepcopy(graph)
        
        # ARMAZENA A COLEÇÃO COMPLETA DE LISTAS DE SFCs
        if not all_sfc_lists:
            raise ValueError("A coleção de listas de SFCs (all_sfc_lists) não pode ser vazia.")
        self.all_sfc_lists = all_sfc_lists
        
        self.lista_SFCs = [] # Esta será a lista do episódio ATUAL, preenchida no reset
        
        self.valid_nodes = valid_nodes
        self.cached_paths = {}
        self.current_sfc_idx = 0
        self.verbose = 0
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
            low=0.0, high=1.0, shape=(len(valid_nodes)*7,), dtype=np.float32
        )


    def reset(self, seed=None, options=None):
        """
        A cada reset (início de um novo episódio), seleciona aleatoriamente
        uma lista de SFCs da coleção para o treinamento.
        """
        # --- LÓGICA DE GENERALIZAÇÃO ---
        # Seleciona aleatoriamente uma das listas de SFCs disponíveis
        self.lista_SFCs = random.choice(self.all_sfc_lists) #
        # --------------------------------

        # O resto do reset continua como antes, mas agora operando na lista recém-selecionada
        self.G = copy.deepcopy(self.initial_G) #
        self.cached_paths.clear() #

        self.reward = 0
        self.total_reward = 0
        self.total_cost = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        
        self._load_sfc(0) #
        
        return self.get_normalized_state(), {} #

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
        """
        Libera os recursos (CPU, cache, banda) alocados para uma SFC específica em todo o grafo.

        Args:
            sfc_id_to_release (str): O ID da SFC cujos recursos devem ser liberados.
        """
        if not sfc_id_to_release:
            return

        # 1. Encontrar o objeto SFC correto a partir do ID fornecido.
        # Esta é a principal correção: buscar em todas as listas de SFCs conhecidas pelo ambiente.
        sfc_object_to_remove = None
        # A busca pode ser otimizada se você souber que a SFC estará sempre na lista do episódio atual.
        # Para robustez, buscamos em todas as listas.
        for sfc_list in self.all_sfc_lists:
            found_sfc = next((sfc for sfc in sfc_list if sfc.id == sfc_id_to_release), None)
            if found_sfc:
                sfc_object_to_remove = found_sfc
                break
        
        # Se a SFC não for encontrada em nenhuma lista, não há o que liberar.
        if not sfc_object_to_remove:
            # print(f"Aviso: SFC com ID {sfc_id_to_release} não encontrada para liberação.")
            return

        # 2. Liberar recursos nos nós
        for node_id, node_data in self.G.nodes(data=True):
            if "sfcs_list" not in node_data or not node_data["sfcs_list"]:
                continue
            
            # Verifica se o objeto SFC a ser removido está de fato alocado neste nó
            if node_data["type"] == 'router' or sfc_object_to_remove not in node_data["sfcs_list"]:
                continue
                
            vnf_names_in_sfc = {item['name'] for item in sfc_object_to_remove.vnfs_dict}
            session_to_release = sfc_object_to_remove.id.split("_")[-1]
            
            service_keys_to_remove = [k for k in node_data.get('services', {}) if k[1] == session_to_release and k[0] in vnf_names_in_sfc]
            
            for service_key in service_keys_to_remove:
                service_name = service_key[0]
                service_info = node_data['services'][service_key]
                
                release_physical_resources = True
                # Se o serviço for compartilhável, só libera recursos se for a última instância
                if self.is_shareable(service_name):
                    # Verifica se existe outro serviço com o mesmo nome, mas de uma sessão diferente
                    if sum(1 for k in node_data['services'] if k[0] == service_name and k[1] != session_to_release) > 0:
                        release_physical_resources = False
                
                if release_physical_resources:
                    node_data['cpu_used'] -= service_info.get('cpu', 0)
                    node_data['cache_used'] -= service_info.get('cache', 0)
                    
                    # Remove da lista de reuso se for o caso
                    if self.is_shareable(service_name) and 'reuse' in node_data:
                        vnf_to_remove_from_reuse = next((vnf for vnf in node_data['reuse'] if vnf.id.startswith(service_name.split('_')[0])), None)
                        if vnf_to_remove_from_reuse:
                            node_data['reuse'].remove(vnf_to_remove_from_reuse)
                
                del node_data['services'][service_key]
                
            # Remove a SFC da lista de SFCs do nó
            node_data['sfcs_list'].remove(sfc_object_to_remove)

        # 3. Liberar recursos de banda nos links
        for u, v, edge_data in self.G.edges(data=True):
            if 'services_in_transit' in edge_data and edge_data['services_in_transit']:
                # CORREÇÃO: Encontra todo o tráfego cuja chave (tupla) corresponde ao sfc_id_to_release.
                # A chave é uma tupla no formato (sfc_id, service_name).
                traffic_to_remove = [k for k in edge_data['services_in_transit'] if k[0] == sfc_id_to_release]
                
                for key in traffic_to_remove:
                    transit_info = edge_data['services_in_transit'][key]
                    # Subtrai a banda que foi usada por esta entrada específica.
                    edge_data['bandwidth_used'] -= transit_info.get('bw_used', 0)
                    del edge_data['services_in_transit'][key]

                    # Garante que o valor não fique negativo por problemas de ponto flutuante
                    if edge_data['bandwidth_used'] < 0:
                        edge_data['bandwidth_used'] = 0

    def _load_sfc(self, sfc_index):
        sfc_to_load = self.lista_SFCs[sfc_index]
        session_number = sfc_to_load.id.split("_")[-1]

        self._release_resources_for_sfc(sfc_to_load.id)
        if self.is_training:
            if int(session_number) >=9:
                id_antigo=subtrair_valor_padrao(sfc_to_load.id, 8)
                self._release_resources_for_sfc(id_antigo)
                if "unique" in id_antigo:
                    sfc_antiga = next((sfc for sfc in self.lista_SFCs if sfc.id == id_antigo), None)
                    self.G.remove_node(sfc_antiga.dst_node)
        self.current_sfc_idx = sfc_index
        self.sfc = sfc_to_load
        services = [item['name'] for item in self.sfc.vnfs_dict]
        self.services = list(reversed(services))
        self.latency_request = 12
        self.dst_node = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        self.service = self.services[0]
        self.current_location = self.dst_node
        self.latency_used = 0
        self.servers_used.clear()
        self.session_number = session_number
        mobile_device_id = self.sfc.dst_node
        self.valid_nodes[-1] = mobile_device_id
        self.update_bandwidth_required()
        if mobile_device_id not in self.G.nodes:
            add_mobile_user_to_graph(self.G, [self.sfc])

    def step(self, action):
        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        server_to_allocate = self.valid_nodes[int(action)]
        self.was_reused_in_step = False
        
        if not self.allocate_resources_on_node(server_to_allocate):
            return self._fail_step('resource')

        # --- LÓGICA DE ROTEAMENTO CENTRALIZADA AQUI ---
        # Chamamos nossa nova função para obter o caminho e sua latência real
        path, path_latency = self._find_viable_path_and_latency(self.current_location, server_to_allocate)

        if not path:
            # Se não há caminho viável (retorno foi None), falha por banda/roteamento.
            return self._fail_step('bandwidth')

        # A função abaixo agora apenas aloca a banda, não calcula mais a latência
        success_band = self.allocate_bandwidth_along_path(path, self.bandwidth_required, self.service)
        
        if not success_band:
            return self._fail_step('bandwidth')

        # Usamos a latência que foi retornada junto com o caminho
        self.latency_used += path_latency
        
        if self.latency_used > self.latency_request:
            return self._fail_step('latency')

        self.servers_used.append(server_to_allocate)
        if self.verbose:
            print(self.servers_used)
        self.total_cost = self.calculate_total_cost(server_to_allocate, path)
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

        # if self.current_sfc_idx == len(self.lista_SFCs) - 1:
        #     # print("Alocou tudo com sucesso.")
        #     done = True
        #     self.success = True
        # else:
        #     self._load_sfc(self.current_sfc_idx + 1)

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
            # Não há caminho, mas não há falha de banda pois nada é alocado. Latência é 0.
            return True
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                # Esta verificação é redundante se _find_viable_path_and_latency funciona bem,
                # mas é uma boa salvaguarda.
                return False
        
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            self._commit_bandwidth_on_link(u, v, vnf, bandwidth_required)
        
        # Não retorna mais a latência aqui
        return True
        
    def _commit_bandwidth_on_link(self, u, v, vnf, bandwidth_required):
        edge = self.G.edges[u, v]
        ms_name = vnf.id
        sfc_id = self.sfc.id # Obtém o ID da SFC atual

        if 'services_in_transit' not in edge:
            edge['services_in_transit'] = {}

        # CORREÇÃO: Usar uma tupla (sfc_id, ms_name) como chave única para o tráfego.
        # Isso evita colisões e permite a liberação precisa dos recursos.
        traffic_key = (sfc_id, ms_name)
        
        # Como a chave agora é única por SFC e serviço, não precisamos mais da lógica de 'copys'.
        # Cada alocação é registrada individualmente.
        if traffic_key not in edge['services_in_transit']:
            edge['services_in_transit'][traffic_key] = {'bw_used': bandwidth_required}
            edge['bandwidth_used'] += bandwidth_required
        # else:
            # A lógica 'else' pode ser removida, pois um mesmo serviço de uma mesma SFC
            # não deveria passar pelo mesmo link duas vezes em uma rota simples.

    def calculate_total_cost(self, server_id, path):
        node = self.G.nodes[server_id]
        
        # Custo de CPU e Cache (lógica inalterada)
        cpu_capacity = node["cpu_capacity"] or 1
        cpu_cost = (node["cpu_used"] / cpu_capacity + 1) ** self.cpu_factor if not self.was_reused_in_step else 0
        cache_capacity = node["cache_capacity"] or 1
        cache_cost = (node["cache_used"] / cache_capacity + 1) ** self.cache_factor if not self.was_reused_in_step else 0
        
        self.latency_cost = 0
        bandwidth_cost = 0

        if path and len(path) >= 2:
            # Custo de Latência (lógica inalterada)
            latency_request = self.latency_request or 1
            self.latency_cost = ((self.latency_used / latency_request) + 1) ** self.latency_factor
            
            # CÁLCULO DE CUSTO DE BANDA CORRIGIDO
            for u, v in zip(path[:-1], path[1:]):
                edge = self.G.edges[u, v]
                bw_used = edge.get('bandwidth_used', 0)
                bw_capacity = edge.get('bandwidth_capacity', 1)

                if bw_capacity > 0:
                    # CORREÇÃO: A banda ('bw_used') já foi alocada na etapa anterior do 'step'.
                    # Portanto, calculamos o custo com base na utilização ATUAL, sem somar o requisito novamente.
                    current_utilization = bw_used / bw_capacity
                    link_cost = (current_utilization + 1) ** self.band_factor
                    bandwidth_cost += link_cost
            # FIM DA CORREÇÃO

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
            # Garante que o shape do array zerado corresponda ao espaço de observação
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        state_vectors = []
        current_vnf = self.sfc.get_vnf_by_id(self.service)
        base_cpu_req = current_vnf.get_cpu_request()
        base_cache_req = current_vnf.get_cache_request()
        bw_req = self.bandwidth_required

        # Otimização para buscar o caminho apenas uma vez por nó candidato
        paths_from_current = {
            node_id: self._find_viable_path_and_latency(self.current_location, node_id)
            for node_id in self.valid_nodes
        }

        for node_id in self.valid_nodes:
            node = self.G.nodes[node_id]
            
            # --- Cálculos de Custo de CPU e Cache ---
            can_reuse = self.can_reuse_vnf_on_node(node_id, self.service)
            effective_cpu_req = 0 if can_reuse else base_cpu_req
            effective_cache_req = 0 if can_reuse else base_cache_req
            
            cpu_capacity = node["cpu_capacity"] or 1
            proj_cpu_util = (node["cpu_used"] + effective_cpu_req) / cpu_capacity


            
            cache_capacity = node["cache_capacity"] or 1
            proj_cache_util = (node["cache_used"] + effective_cache_req) / cache_capacity

            if node["cache_used"] + base_cache_req >= node["cache_capacity"] or node["cpu_used"] + base_cpu_req >= node["cpu_capacity"]:
                proj_cpu_util, proj_cache_util = 1, 1
                
            # --- Cálculo de Custo de Latência e Banda ---
            path, path_latency = paths_from_current[node_id]
            latency_request = self.latency_request or 1
            
            is_path_viable = path_latency != float('inf')
            
            proj_latency_util = (self.latency_used + path_latency) / latency_request if is_path_viable else 1.0
            
            proj_bandwidth_cost = 0.0
            if is_path_viable and path and len(path) >= 2:
                num_links = len(path) - 1
                total_link_cost = 0
                for u, v in zip(path[:-1], path[1:]):
                    edge = self.G.edges[u, v]
                    bw_used = edge.get('bandwidth_used', 0)
                    bw_capacity = edge.get('bandwidth_capacity', 1) # Evita divisão por zero

                    projected_utilization = (bw_used + bw_req) / bw_capacity
                    link_cost = (projected_utilization + 1) ** self.band_factor
                    total_link_cost += link_cost
                
                # Normaliza o custo pelo número de links para ter uma média
                proj_bandwidth_cost = total_link_cost / num_links if num_links > 0 else 0
            elif not is_path_viable:
                # Se o caminho não for viável, o custo de banda é máximo.
                # O valor 1.0 é usado para consistência com os outros custos no estado normalizado.
                proj_bandwidth_cost = 1.0

            # --- Flags de Estado ---
            is_server_used = 1.0 if node_id in self.servers_used else 0.0
            offers_reuse_flag = 1.0 if can_reuse else 0.0
            
            # --- CORREÇÃO PRINCIPAL APLICADA AQUI ---
            # A impossibilidade de alocação é verificada pela utilização de recursos ou pela inviabilidade do caminho.
            cant_allocate = 1.0 if (proj_cpu_util > 1.0 or 
                                    proj_cache_util > 1.0 or 
                                    proj_latency_util > 1.0 or
                                    not is_path_viable) else 0.0
            
            # O vetor de estado do nó, com todos os valores clipados entre 0.0 e 1.0
            node_state = [
                min(proj_cpu_util, 1.0), 
                min(proj_cache_util, 1.0), 
                min(proj_latency_util, 1.0), 
                min(proj_bandwidth_cost, 1.0), # Custo de banda também é clipado para normalização
                is_server_used, 
                cant_allocate, 
                offers_reuse_flag
            ]
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
    

    def _find_viable_path_and_latency(self, source, target):
        """
        Encontra um caminho viável de source para target e retorna o caminho e sua latência em saltos.
        A "fonte da verdade" para o roteamento.
        
        1. Tenta o caminho mais curto em saltos.
        2. Se não tiver banda, busca um caminho alternativo com banda suficiente.
        
        Retorna:
            (list, int): Uma tupla contendo o caminho (lista de nós) e a latência (número de saltos).
            (None, float('inf')): Se nenhum caminho viável for encontrado.
        """
        if source == target:
            return [source], 0

        # Tenta primeiro o caminho mais curto (cached ou calculado)
        cache_key = (source, target)
        path = self.cached_paths.get(cache_key)
        if not path:
            try:
                path = nx.shortest_path(self.G, source, target)
                self.cached_paths[cache_key] = path
            except nx.NetworkXNoPath:
                return None, float('inf')

        # Verifica se o caminho mais curto tem banda
        if self.get_critical_link_bandwidth(path) >= self.bandwidth_required:
            return path, len(path) - 1

        # Se não tiver, busca um alternativo (aqui a lógica de fallback é crucial)
        # Esta função externa deve ser otimizada para encontrar o caminho mais curto que satisfaça a banda
        path_alternative = get_available_shortest_path(self.G, source, target, self.bandwidth_required, latencia_saltos=True)
        
        if path_alternative:
            # Atualiza o cache com o novo caminho viável
            self.cached_paths[cache_key] = path_alternative
            return path_alternative, len(path_alternative) - 1
        
        # Se nenhum caminho (nem o original, nem o alternativo) for viável
        return None, float('inf')