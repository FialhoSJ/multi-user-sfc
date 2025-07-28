import copy
import gymnasium
from gymnasium import spaces
import numpy as np
from networkx import Graph
from typing import Union, List, Dict

from core.sfc import SFC, VNF
from algorithms.networkUtils import get_available_shortest_path_optimized, calcular_latencia_total

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')


class SFC_AllocationEnv(gymnasium.Env):
    """
    Ambiente do Gymnasium para o problema de alocação de Service Function Chains (SFCs).
    
    Este ambiente simula a alocação de Virtual Network Functions (VNFs) de uma SFC
    em nós de uma infraestrutura de rede, considerando restrições de CPU, cache,
    latência e largura de banda.
    """

    # =================================================================================
    # 1. Métodos Principais da Interface do Gymnasium
    # =================================================================================

    def __init__(self,
                 valid_nodes: List[Union[int, str]],
                 list_graph_per_session: List[Graph],
                 list_sfcs_per_session: List[SFC],
                 pesos_fatores: Dict[str, float] = None,
                 reward_config: Dict[str, float] = None):
        """
        Inicializa o ambiente de alocação de SFC.
        """
        super().__init__()

        if len(list_graph_per_session) != len(list_sfcs_per_session):
            raise ValueError("A lista de grafos por sessão deve ter o mesmo tamanho da lista de SFCs.")

        # --- Parâmetros de Configuração ---
        self.valid_nodes = valid_nodes
        self.list_graph_per_session = list_graph_per_session
        self.list_sfcs_per_session = list_sfcs_per_session
        self.pesos_fatores = pesos_fatores if pesos_fatores is not None else \
                             {"cpu": 1.0, "cache": 1.0, "lat": 1.0, "band": 1.0}
        

        if reward_config is None:
            self.reward_config = {"success_bonus": 1000.0, "failure_penalty": -1000.0}
        else:
            self.reward_config = reward_config
        
        self._initialize_snapshots()

        # --- Estado do Episódio ---
        self.graph: Graph = None
        self.list_sfcs: List[SFC] = None
        self.current_sfc: SFC = None
        self.current_vnf: VNF = None
        self.current_session = None
        self.current_location: Union[int, str] = None
        self.band_request = None
        self.latency_used = 0
        self.latency_request = None
        self.is_training = True

        # --- Espaços de Ação e Observação ---
        self.action_space = spaces.Discrete(len(self.valid_nodes))
        self.observation_space = spaces.Dict({
            "valid_node_resource": spaces.Box(low=0, high=1, shape=(2, len(self.valid_nodes)), dtype=np.float32),
            "valid_node_lat_band": spaces.Box(low=0, high=1, shape=(2, len(self.valid_nodes)), dtype=np.float32),
            "valid_node_reuse": spaces.Box(low=0, high=1, shape=(len(self.valid_nodes),), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(len(self.valid_nodes),), dtype=np.int8),
        })

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o início de um novo episódio.
        Seleciona aleatoriamente um grafo e uma lista de SFCs da sessão,
        restaura o estado inicial dos recursos e prepara a primeira SFC para alocação.
        """
        super().reset(seed=seed)

        # Seleciona uma sessão aleatória para o episódio
        idx = np.random.randint(len(self.list_graph_per_session))
        self.graph = self.list_graph_per_session[idx]
        self.list_sfcs = self.list_sfcs_per_session[idx]

        # Restaura o estado do grafo a partir do snapshot inicial

        # IMPLEMENTAR A RESTAURAÇÂO DOS SERVIÇOS ALOCADOS EM CADA NÓ ===================================
        snapshot_nodes = self.initial_resource_snapshot[idx]['nodes']
        for node_id, initial_state in snapshot_nodes.items():
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id]['cpu_used'] = initial_state['cpu_used']
                self.graph.nodes[node_id]['cache_used'] = initial_state['cache_used']
                self.graph.nodes[node_id]['services'] = copy.deepcopy(initial_state['services'])

        #==============================================================================================

        snapshot_edges = self.initial_resource_snapshot[idx]['edges']
        for (u, v), initial_state in snapshot_edges.items():
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v]['bandwidth_used'] = initial_state['bandwidth_used']

        # Configura a primeira SFC e reinicia as variáveis de estado do episódio
        self.set_current_sfc(self.list_sfcs[0])
        self.latency_used = 0
        self.servers_used = []
        self.reward = 0
        self.total_reward = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Executa um passo no ambiente a partir de uma ação do agente.
        A ação corresponde à escolha de um nó para alocar a VNF atual.
        """
        # 1. Traduzir a ação para um nó do grafo
        chosen_server = self.current_sfc.dst_node if action == len(self.valid_nodes) - 1 else self.valid_nodes[action]
        self.server = chosen_server

        # 2. Tentar alocar recursos (CPU/cache) no nó escolhido
        if not self.allocate_resources_on_node(self.server):
            return self._fail_step('resource')

        # 3. Encontrar e alocar recursos no caminho (banda)
        self.path = get_available_shortest_path_optimized(
            self.graph, self.current_location, self.server,
            self.band_request, rounded=True
        )
        if not self.path or not self.allocate_bandwidth_along_path(self.path, self.band_request):
            return self._fail_step('bandwidth')

        # 4. Calcular latência e verificar restrição
        path_latency = calcular_latencia_total(self.graph, self.path, self.current_vnf)
        self.latency_used += path_latency
        if self.latency_used > self.latency_request:
            return self._fail_step('latency')

        # 5. Calcular custo e recompensa
        self.servers_used.append(self.server)
        total_cost = self.calculate_total_cost(self.server, self.path)
        reward = -total_cost
        self.total_reward += reward

        # 6. Atualizar estado para o próximo passo
        self.current_location = self.server
        if not self.is_training:
            self.allocation_results[self.current_vnf.id] = {'allocated_server': self.server, 'path': self.path, 'cost': total_cost}

        # 7. Verificar conclusão e avançar para a próxima VNF/SFC
        done = False
        if not self.current_vnf.previous_vnf:
            if self.current_sfc.id == self.list_sfcs[-1].id:
                done = True
                self.success = True
                
                # --- USA O BÔNUS DE SUCESSO CONFIGURADO ---
                reward += self.reward_config['success_bonus']
            else:
                idx = self.list_sfcs.index(self.current_sfc)
                self.set_current_sfc(self.list_sfcs[idx + 1])
                # Note que a linha 'self.update_band_request()' é chamada dentro de 'set_current_sfc'
        else:
            self.current_vnf = self.current_sfc.get_previous_vnf(self.current_vnf)
            self.update_band_request()

        return self._get_obs(), reward, done, False, {}

    def render(self):
        """Método para renderização (não implementado)."""
        pass

    def close(self):
        """Método para fechar o ambiente e liberar recursos (não implementado)."""
        pass

    # =================================================================================
    # 2. Lógica Central da Simulação e Estado
    # =================================================================================

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Monta e retorna a observação atual do ambiente, incluindo a máscara de ações válidas.
        A observação normaliza o estado dos recursos da rede em relação à requisição atual.
        """
        cpu_req = self.current_vnf.get_cpu_request()
        cache_req = self.current_vnf.get_cache_request()
        num_valid_nodes = len(self.valid_nodes)

        # Inicializa arrays de observação
        node_resource_obs = np.zeros((2, num_valid_nodes), dtype=np.float32)
        node_lat_band_obs = np.zeros((2, num_valid_nodes), dtype=np.float32)
        node_reuse_obs = np.zeros(num_valid_nodes, dtype=np.float32)
        action_mask = np.ones(num_valid_nodes, dtype=np.int8)

        for i, node_id in enumerate(self.valid_nodes):
            is_action_valid = True
            node_id = self.current_sfc.dst_node if i == len(self.valid_nodes) - 1 else node_id
            
            # --- Normalização de Recursos do Nó (CPU/Cache) ---
            cpu_cap, cpu_used = self.graph.nodes[node_id]['cpu_capacity'], self.graph.nodes[node_id]['cpu_used']
            new_cpu_used = min((cpu_req + cpu_used) / cpu_cap, 1.0)
            
            cache_cap, cache_used = self.graph.nodes[node_id]['cache_capacity'], self.graph.nodes[node_id]['cache_used']
            new_cache_used = min((cache_req + cache_used) / cache_cap, 1.0)

            node_resource_obs[0, i] = new_cpu_used
            node_resource_obs[1, i] = new_cache_used
            if new_cpu_used >= 1.0 or new_cache_used >= 1.0:
                is_action_valid = False

            # --- Normalização de Recursos do Caminho (Latência/Banda) ---
            path = get_available_shortest_path_optimized(self.graph, self.current_location, node_id, self.band_request, rounded=True)
            if path:
                latencia_path = calcular_latencia_total(self.graph, path, self.current_vnf)
                latencia_normalizada = min((self.latency_used + latencia_path) / self.latency_request, 1.0)
                
                band_cap, band_used = self.get_critical_link_info(path)
                band_normalizada = min((band_used + self.band_request) / band_cap, 1.0)
            else:
                latencia_normalizada, band_normalizada = 1.0, 1.0
                is_action_valid = False
            
            node_lat_band_obs[0, i] = latencia_normalizada
            node_lat_band_obs[1, i] = band_normalizada
            if latencia_normalizada >= 1.0 or band_normalizada >= 1.0:
                is_action_valid = False

            # --- Observação de Reutilização ---
            reusable = self.is_reusable_at_node(node_id, self.current_vnf) if is_action_valid else 0
            node_reuse_obs[i] = reusable
            
            # --- Máscara de Ação ---
            if not is_action_valid:
                action_mask[i] = 0

        return {
            "valid_node_resource": node_resource_obs,
            "valid_node_lat_band": node_lat_band_obs,
            "valid_node_reuse": node_reuse_obs,
            "action_mask": action_mask,
        }
    
    def allocate_resources_on_node(self, node_id: Union[int, str]) -> bool:
        """
        Aloca CPU e Cache em um nó, considerando o reuso de serviços.
        Retorna True se a alocação for bem-sucedida, False caso contrário.
        """
        node = self.graph.nodes[node_id]
        vnf = self.current_vnf
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        
        # Se o serviço for reutilizável, o custo efetivo de recursos é zero
        can_reuse = self.is_reusable_at_node(node_id, vnf)
        effective_cpu_req = 0 if can_reuse else cpu_req
        effective_cache_req = 0 if can_reuse else cache_req
        
        # Verifica se há capacidade disponível para a alocação
        if (node['cpu_used'] + effective_cpu_req > node['cpu_capacity']) or \
           (node['cache_used'] + effective_cache_req > node['cache_capacity']):
            return False

        # Aloca os recursos e atualiza os metadados do serviço
        node['cpu_used'] += effective_cpu_req
        node['cache_used'] += effective_cache_req
        
        if 'services' not in node:
            node['services'] = {}
            
        session_id = self.current_sfc.id.split("_")[-1]
        service_key = (vnf.id, session_id)
        
        if service_key in node['services']:
            node['services'][service_key]['copys'] += 1
        else:
            node['services'][service_key] = {'cpu': cpu_req, 'cache': cache_req, 'copys': 1}
            
        return True

    def allocate_bandwidth_along_path(self, path: List, bandwidth_required: float) -> bool:
        """
        Aloca largura de banda ao longo de um caminho de forma atômica.
        Verifica todos os links primeiro e, se todos tiverem capacidade, aloca a banda.
        Retorna True em caso de sucesso, False caso contrário.
        """
        # 1. Verificar se todos os links no caminho têm capacidade suficiente
        for u, v in zip(path[:-1], path[1:]):
            edge = self.graph.edges[u, v]
            available_bw = edge.get('bandwidth_capacity', 0) - edge.get('bandwidth_used', 0)
            if available_bw < bandwidth_required - 1e-9: # Tolerância para ponto flutuante
                return False

        # 2. Se a verificação passou, alocar a banda em todos os links
        for u, v in zip(path[:-1], path[1:]):
            self.graph.edges[u, v]['bandwidth_used'] += bandwidth_required
        
        return True

    def _fail_step(self, reason: str):
        """
        Finaliza um episódio com falha, aplicando uma penalidade alta.
        """
        self.fail_reason = reason
        self.success = False
        
        # --- USA A PENALIDADE CONFIGURADA ---
        reward = self.reward_config['failure_penalty']
        
        done = True
        return self._get_obs(), reward, done, False, {}
    
    def _initialize_snapshots(self):
        """
        Cria um snapshot do estado inicial dos recursos de todos os grafos
        para garantir um reset consistente dos episódios.
        """
        self.initial_resource_snapshot = {}
        for idx, graph in enumerate(self.list_graph_per_session):
            nodes = {n_id: {'cpu_used': data.get('cpu_used', 0),
                            'cache_used': data.get('cache_used', 0),
                            'services': copy.deepcopy(data.get('services', {}))}
                     for n_id, data in graph.nodes(data=True)}
            edges = {(u, v): {'bandwidth_used': data.get('bandwidth_used', 0)}
                     for u, v, data in graph.edges(data=True)}
            self.initial_resource_snapshot[idx] = {'nodes': nodes, 'edges': edges}

    # =================================================================================
    # 3. Funções Utilitárias e de Suporte
    # =================================================================================

    def set_current_sfc(self, sfc: SFC):
        """Define a SFC atual para alocação e inicializa seus parâmetros."""
        if not sfc:
            raise ValueError("SFC não pode ser None.")
        self.current_sfc = sfc
        self.current_vnf = self.current_sfc.get_dst_vnf()
        self.latency_request = 10  # TODO: Considerar tornar dinâmico
        self.current_location = self.current_sfc.dst_node
        self.latency_used = 0
        self.servers_used = []
        self.update_band_request()
        self.current_session = self.current_sfc.id.split("_")[-1]

    def update_band_request(self):
        """Atualiza a banda necessária para a VNF atual."""
        self.band_request = self.current_vnf.get_outcome_interface_bandwidth()

    def is_reusable_at_node(self, node_id: Union[int, str], vnf: VNF) -> bool:
        """Verifica se uma VNF compartilhável já está alocada em um nó."""
        service_name = vnf.id
        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False
        
        session_id = self.current_sfc.id.split("_")[-1]
        service_key = (service_name, session_id)
        
        return service_key in self.graph.nodes[node_id].get('services', {})

    def get_critical_link_info(self, path: List):
        """Retorna a capacidade e o uso do link mais sobrecarregado em um caminho."""
        if not path or len(path) < 2:
            return 0, 0

        min_available = float('inf')
        crit_cap, crit_used = 0, 0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.graph.edges[u, v]
            cap = edge.get('bandwidth_capacity', 0)
            used = edge.get('bandwidth_used', 0)
            available = cap - used
            if available < min_available:
                min_available = available
                crit_cap, crit_used = cap, used
        return crit_cap, crit_used
    


    def calculate_total_cost(self, server_id, path):
        """Calcula o custo total da alocação de um serviço."""
        node = self.graph.nodes[server_id]

        reusable = self.is_reusable_at_node(server_id, self.current_vnf)
        # Custo de CPU e Cache (zero se houver reuso)
        cpu_capacity = node["cpu_capacity"] or 1
        cpu_cost = (node["cpu_used"] / cpu_capacity + 1) ** self.pesos_fatores['cpu'] if not reusable else 0
        
        cache_capacity = node["cache_capacity"] or 1
        cache_cost = (node["cache_used"] / cache_capacity + 1) ** self.pesos_fatores['cache'] if not reusable else 0
        
        # Custo de Rede (Latência e Banda)
        latency_cost = 0
        bandwidth_cost = 0
        if len(path) >= 2:
            latency_request = self.latency_request or 1
            latency_cost = ((self.latency_used / latency_request) + 1) ** self.pesos_fatores['lat']
            
            # Custo de banda baseado no link crítico do caminho (LINHAS DESCOMENTADAS)
            cap_band, used_band = self.get_critical_link_info(path)
            cap_band = cap_band or 1  # Evita divisão por zero
            bandwidth_cost = ((used_band / cap_band) + 1) ** self.pesos_fatores['band']
 
        return sum([cpu_cost, cache_cost, latency_cost, bandwidth_cost])