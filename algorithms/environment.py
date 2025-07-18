import copy
import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

# Supondo que essas funções existem em um módulo `algorithms.networkUtils`
from algorithms.networkUtils import get_available_shortest_path, calculate_computational_latency, calculate_latency_betwen_nodes

# --- CONSTANTES ---
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

# --- FUNÇÕES AUXILIARES EXTERNAS ---
def latency_rounded(u, v, data):
    """Função de peso para o NetworkX, usada para arredondar a latência."""
    return int(round(data.get('latency', 0), 3) * 1000)

# --- CLASSE DO AMBIENTE ---
class NetworkEnv(gym.Env):
    """
    Ambiente customizado do Gymnasium para simular a alocação de Service Function Chains (SFCs)
    em uma rede de substrato. O objetivo de um agente de RL neste ambiente é aprender a
    alocar uma cadeia de serviços sequencialmente, minimizando um custo combinado de
    recursos computacionais (CPU, cache), latência e largura de banda.
    """
    def __init__(self, graph, valid_nodes, pesos):
        super().__init__()

        # --- Topologia e Configuração da Rede ---
        # self.G_backup = copy.deepcopy(graph)  # Backup para o reset
        self.G = graph
        self.initial_resource_snapshot = {}
        self.valid_nodes = valid_nodes
        # self.menor_reward = -float("inf")
        self.cached_paths = {}

        # --- Parâmetros da SFC e Estado Atual ---
        self.sfc = None
        self.services = None
        self.service_requirements = None
        self.latency_request = None
        self.dst_node = None
        self.current_location = None
        self.service = None
        self.session_number = None

        # --- Métricas de Desempenho e Controle ---
        self.latency_used = 0
        self.is_training = True
        self.servers_used = []
        self.allocation_results = {}
        self.success = False
        self.reuse = False
        
        # --- Fatores de Custo para a Recompensa ---
        self.cpu_factor = pesos.get("cpu", 1)
        self.cache_factor = pesos.get("cache", 1)
        self.band_factor = pesos.get("band", 1)
        self.latency_factor = pesos.get("latency", 1)
        
        # --- Espaços de Ação e Observação ---
        self.action_space = spaces.Discrete(len(self.valid_nodes))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.valid_nodes) * 5,),  # 5 métricas por nó válido
            dtype=np.float32
        )

    # --------------------------------------------------------------------------
    # --- MÉTODOS PRINCIPAIS DO GYMNASIUM (CORE API) ---
    # --------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente restaurando o grafo ao seu 'snapshot' inicial,
        o que é muito mais rápido que um deepcopy.
        """
        # self.G = copy.deepcopy(self.G_backup)

        snapshot_nodes = self.initial_resource_snapshot['nodes']

        for node_id, initial_state in snapshot_nodes.items():
            node = self.G.nodes[node_id]
            node['cpu_used'] = initial_state['cpu_used']
            node['cache_used'] = initial_state['cache_used']

        snapshot_edges = self.initial_resource_snapshot['edges']
        for (u, v), initial_state in snapshot_edges.items():
            edge = self.G.edges[u, v]
            edge['bandwidth_used'] = initial_state['bandwidth_used']
            
        # Reinicializa estado da alocação
        self.latency_used = 0
        self.current_location = self.dst_node
        self.servers_used.clear()
        self.service = self.services[0]
        self.update_bandwidth_required()
        
        # Reinicializa métricas de recompensa e resultado
        self.reward = 0
        self.total_reward = 0
        self.total_cost = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        
        return self.get_normalized_state(), {}

    def step(self, action):
        """
        Executa um passo no ambiente a partir de uma ação do agente.
        A ação corresponde a escolher um nó para alocar o serviço atual da SFC.
        """
        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        
        self.server = self.valid_nodes[int(action)]


        self.reuse = self.is_reusable_at_node(self.server, self.service, self.sfc.id.split("_")[-1])

        # 1. Alocar recursos no nó (CPU/Cache)
        if not self.allocate_resources_on_node(self.server, self.reuse):
            return self._fail_step('resource')

        # 2. Encontrar caminho e alocar banda
        # self.path = self._get_path_with_fallback(self.current_location, self.server)
        self.path = self._get_path_with_fallback(self.current_location, self.server)
        if not self.path:
            return self._fail_step('bandwidth')

        # 3. Alocar banda e calcular latência do caminho
        success_band, path_latency = self.allocate_bandwidth_along_path(self.path, self.bandwidth_required, self.service)
        if not success_band:
            # Esta verificação é uma dupla segurança, o fallback já deveria ter resolvido.
            return self._fail_step('bandwidth')

        self.latency_used += path_latency
        
        # # 4. Verificar restrição de latência
        # if self.latency_used > self.latency_request:
        #     return self._fail_step('latency')

        # 5. Calcular custo e recompensa
        self.servers_used.append(self.server)
        # if self.servers_used[0] and isinstance(self.servers_used[0],str): 
        #     if len(self.servers_used) > 1 :
        #         if self.servers_used[0] == self.servers_used[1]:
        #             aqui = 1
        self.total_cost = self.calculate_total_cost(self.server, self.path)
        self.reward = -self.total_cost
        self.total_reward += self.reward

        # 6. Atualizar estado para o próximo passo
        
        self.current_location = self.server

        if not self.is_training:
            self.allocation_results[self.service] = {'allocated_server': self.server, 'path': self.path, 'cost': self.total_cost}

        # 7. Verificar se o episódio terminou
        done = False
        if self.service == self.services[-1]:
            done = True
            self.success = True
            # if self.total_reward > self.menor_reward:
            #     print("Solucao ",self.servers_used, f" reward: {self.total_reward}")
            #     self.menor_reward = self.total_reward
        else:
            current_index = self.services.index(self.service)
            self.service = self.services[current_index + 1]
            self.update_bandwidth_required()
        
        return self.get_normalized_state(), self.reward, done, False, {}
    
    # --------------------------------------------------------------------------
    # --- MÉTODOS DE CONFIGURAÇÃO DO AMBIENTE ---
    # --------------------------------------------------------------------------

    def set_graph(self, graph):
        """Define e faz backup do grafo da rede."""
        # self.G_backup = copy.deepcopy(graph)
        self.G = graph
        self._capture_initial_snapshot()

    def set_dst_node(self, dst_node):
        """Define o nó de destino inicial da SFC."""
        if not dst_node:
            raise ValueError("Nó de destino (dst_node) não pode ser None.")
        self.dst_node = dst_node
        self.current_location = dst_node

    

    # --------------------------------------------------------------------------
    # --- MÉTODOS AUXILIARES DE LÓGICA INTERNA ---
    # --------------------------------------------------------------------------

    def _fail_step(self, reason):
        """Padroniza a finalização de um episódio por falha."""
        self.fail_reason = reason
        self.success = False
        self.reward = -2000  # Penalidade fixa e alta por falha
        done = True
        return self.get_normalized_state(), self.reward, done, False, {}

    def _get_path_with_fallback(self, source, target):
        """Tenta obter o caminho do cache, senão busca o caminho mais curto com banda disponível."""
        # Tenta o caminho mais rápido (Dijkstra puro)

        if (source, target) not in self.cached_paths:
            path = get_available_shortest_path(self.G, source, target, self.bandwidth_required, rounded=True)
            self.cached_paths[(source, target)] = path
        else:
            path = self.cached_paths[(source, target)]
        
        # Verifica se este caminho rápido tem banda
        if self.get_critical_link_bandwidth(path) >= self.bandwidth_required:
            return path
        
        # Fallback: Se não tem banda, busca um caminho viável (pode ser mais lento)
        path = get_available_shortest_path(self.G, source, target, self.bandwidth_required, rounded=True)
        if not path:
            return self.cached_paths[(source, target)]
        else:
            self.cached_paths[(source, target)] = path
            return path
        

    # environment.py

    def allocate_resources_on_node(self, node_id, is_reusable):
        """Aloca CPU e Cache em um nó, considerando a possibilidade de reuso."""
        
        vnf = self.sfc.get_vnf_by_id(self.service)
        
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        can_reuse = self.is_reusable_at_node(node_id,vnf.id,self.sfc.id.split("_")[-1])
        # Recurso necessário é zero se for reutilizável
        effective_cpu_req = 0 if can_reuse else cpu_req
        effective_cache_req = 0 if can_reuse else cache_req

        # Esta verificação de segurança agora usa o valor efetivo (corrigindo o Erro 2)
        if (self.G.nodes[node_id]['cpu_used'] + cpu_req >= self.G.nodes[node_id]['cpu_capacity']) or \
            (self.G.nodes[node_id]['cache_used'] + cache_req) >= self.G.nodes[node_id]['cache_capacity']:
            return False
        
        # Aloca os recursos efetivos
        self.G.nodes[node_id]['cpu_used'] += effective_cpu_req
        self.G.nodes[node_id]['cache_used'] += effective_cache_req

        # --- INÍCIO DA MODIFICAÇÃO PRINCIPAL ---
        # Registra a alocação no dicionário 'services' do nó dentro do grafo da simulação.
        # Isso imita o comportamento do sfcs_instantiator.
        # if 'services' not in self.G.nodes[node_id]:
        #     self.G.nodes[node_id]['services'] = {}
        
        # # A sessão é obtida de 'self.session_number', que é definido em kuririn.py
        # service_key = (self.service, self.sfc.id.split("_")[-1]) 
        
        # if service_key not in self.G.nodes[node_id]['services']:
        #      self.G.nodes[node_id]['services'][service_key] = {'copys': 1}
        # else:
        #      self.G.nodes[node_id]['services'][service_key]['copys'] += 1
        # # --- FIM DA MODIFICAÇÃO PRINCIPAL ---

        # if can_reuse and not self.is_training:
        #     print("reuso no kuririn no nó", node_id)
        #     aux = node_id
        
        return True
    
    def allocate_bandwidth_along_path(self, path, bandwidth_required, ms_name):
        """
        Aloca banda ao longo de um caminho usando uma abordagem de duas passagens (verificar, depois alocar).
        Retorna (True, latência_total) em sucesso, ou (False, 0.0) em falha.
        """
        if not path or len(path) < 2:
            return True, 0.0

        # --- 1ª Passagem: VERIFICAÇÃO ---
        # Percorre todo o caminho para garantir que cada enlace tem capacidade suficiente.
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                # Se qualquer enlace no caminho falhar na verificação, a operação inteira é abortada.
                return False, 0.0

        # --- 2ª Passagem: ALOCAÇÃO (COMMIT) ---
        # Se o código chegou a este ponto, o caminho inteiro é válido.
        # Agora, percorremos o caminho novamente para efetivamente alocar os recursos.
        total_latency = 0.0
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            # A chamada para _commit_bandwidth_on_link agora é garantida de não exceder a capacidade.
            latency = self._commit_bandwidth_on_link(u, v, vnf, bandwidth_required)
            total_latency += latency

        return True, total_latency
        
    def _commit_bandwidth_on_link(self, u, v, vnf, bandwidth_required):
        """Aloca banda e calcula latência para um único enlace (u, v)."""
        edge = self.G.edges[u, v]
        edge['bandwidth_used'] += bandwidth_required
        latency = calculate_latency_betwen_nodes(self.G, u, v, vnf)
        return latency

    def calculate_total_cost(self, server_id, path):
        """Calcula o custo total da alocação de um serviço."""
        node = self.G.nodes[server_id]

        # Custo de CPU e Cache (zero se houver reuso)
        cpu_capacity = node["cpu_capacity"] or 1
        cpu_cost = (node["cpu_used"] / cpu_capacity + 1) ** self.cpu_factor if not self.reuse else 0
        
        cache_capacity = node["cache_capacity"] or 1
        cache_cost = (node["cache_used"] / cache_capacity + 1) ** self.cache_factor if not self.reuse else 0
        

        # # Penalidade se o nó já foi usado na mesma SFC
        # if server_id in self.servers_used:
        #     cpu_cost *= 1.5 # Aumenta o custo em 50%
        #     cache_cost *= 1.5

        # Custo de Rede (Latência e Banda)
        self.latency_cost = 0
        bandwidth_cost = 0
        if len(path) >= 2:
            latency_request = self.latency_request or 1
            self.latency_cost = ((self.latency_used / latency_request) + 1) ** self.latency_factor
            
            # # Custo de banda baseado no link crítico do caminho
            # cap_band, used_band = self.get_critical_link_info(path)
            # cap_band = cap_band or 1
            # bandwidth_cost = (used_band / cap_band + 1) ** self.band_factor
 

        return sum([cpu_cost, cache_cost, self.latency_cost, bandwidth_cost])

    # --------------------------------------------------------------------------
    # --- MÉTODOS DE GERAÇÃO DE ESTADO ---
    # --------------------------------------------------------------------------

    def get_normalized_state(self):
        """Gera o vetor de estado normalizado para o agente de RL."""
        state_vectors = []
        for node_id in self.valid_nodes:
            node = self.G.nodes[node_id]

            # 1. Custos de Recursos (CPU & Cache)
            cpu_capacity = node["cpu_capacity"] or 1
            cache_capacity = node["cache_capacity"] or 1
            is_reusable = int(self.is_reusable_at_node(node_id, self.service, self.sfc.id.split("_")[-1]))
            

                
            cpu_req = 0 if is_reusable else self.service_requirements[self.service]["cpu"]
            cache_req = 0 if is_reusable else self.service_requirements[self.service]["cache"]

            if (self.service_requirements[self.service]["cpu"] + node["cpu_used"]) > node["cpu_capacity"] or \
            (self.service_requirements[self.service]["cache"] + node["cache_used"]) > node["cache_capacity"]:
                proj_cpu_cost = 1
                proj_cache_cost = 1
            
            else:
                proj_cpu_cost = (node["cpu_used"] + cpu_req) / cpu_capacity
                proj_cache_cost = (node["cache_used"] + cache_req) / cache_capacity

            
            
            
            # 2. Custos de Rede (Latência)
            # if (self.current_location, node_id) not in self.cached_paths:
            #     self.cached_paths[(self.current_location, node_id)] = ge
           
            path =  self._get_path_with_fallback(self.current_location, node_id)
            path_latency = self.calculate_path_latency(path, self.service)
            latency_request = self.latency_request or 1
            proj_latency_cost = (self.latency_used + path_latency) / latency_request

            # 3. Flags de Estado
           
            
            cant_allocate = 1.0 if (proj_cpu_cost > 1.0 or proj_cache_cost > 1.0) else 0.0

            # 4. Montagem do Vetor de Estado do Nó
            node_state = [
                min(proj_cpu_cost, 1.0),
                min(proj_cache_cost, 1.0),
                min(proj_latency_cost, 1.0),
                is_reusable,
                cant_allocate
            ]
            state_vectors.extend(node_state)

        return np.array(state_vectors, dtype=np.float32)

    # --------------------------------------------------------------------------
    # --- MÉTODOS UTILITÁRIOS E DE CONSULTA ---
    # --------------------------------------------------------------------------

    def update_bandwidth_required(self):
        """Atualiza a necessidade de banda para o serviço atual na SFC."""
        if not self.service:
            raise ValueError("Serviço atual não definido.")
        self.bandwidth_required = self.sfc.get_vnf_by_id(self.service).get_outcome_interface_bandwidth()

    def calculate_path_latency(self, path, ms_name):
        """Calcula a latência total de um caminho sem modificar o grafo."""
        if not path or len(path) < 2:
            return 0.0
        
        total_latency = 0.0
        vnf = self.sfc.get_vnf_by_id(ms_name)
        for u, v in zip(path[:-1], path[1:]):
            total_latency += calculate_latency_betwen_nodes(self.G, u, v, vnf)
        return total_latency

    def get_critical_link_bandwidth(self, path):
        """Retorna a menor banda disponível em um caminho."""
        if not path or len(path) < 2:
            return float('inf')
        
        min_available = float('inf')
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            available = edge.get('bandwidth_capacity', 0) - edge.get('bandwidth_used', 0)
            min_available = min(min_available, available)
        return min_available

    def get_critical_link_info(self, path):
        """Retorna a capacidade e o uso do link mais crítico (menor banda livre) em um caminho."""
        if not path or len(path) < 2:
            return 0, 0

        min_available = float('inf')
        crit_cap, crit_used = 0, 0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G.edges[u, v]
            cap = edge.get('bandwidth_capacity', 0)
            used = edge.get('bandwidth_used', 0)
            available = cap - used
            if available < min_available:
                min_available = available
                crit_cap, crit_used = cap, used
        return crit_cap, crit_used
    
    def is_shareable(self,service_name):
        # TODO Mudar para a informação de compartilháveis estar em uma variável separável.
        #if self.shareable_node:
        if True:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False

    # environment.py

    def is_reusable_at_node(self, node_id, service_name, session_id):

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False
        service_key = (service_name,session_id)
        if service_key in self.G.nodes[node_id]['services']:
                return True
        return False

    def _capture_initial_snapshot(self):
        """
        Captura o estado de uso de recursos (CPU, cache, banda) do grafo atual.
        Deve ser chamado sempre que um novo grafo é definido no ambiente.
        """
        self.initial_resource_snapshot['nodes'] = {}
        for node_id, data in self.G.nodes(data=True):
            self.initial_resource_snapshot['nodes'][node_id] = {
                'cpu_used': data.get('cpu_used', 0),
                'cache_used': data.get('cache_used', 0),
            }

         # Captura estado das arestas
        self.initial_resource_snapshot['edges'] = {}
        for u, v, data in self.G.edges(data=True):
            self.initial_resource_snapshot['edges'][(u, v)] = {
                'bandwidth_used': data.get('bandwidth_used', 0),
            }