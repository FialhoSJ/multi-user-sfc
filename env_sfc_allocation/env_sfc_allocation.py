import copy
import gymnasium
from gymnasium import spaces
import numpy as np
from networkx import Graph
from typing import Tuple, Union, List, Dict
from core.sfc import SFC, VNF
from algorithms.networkUtils import get_available_shortest_path_optimized, calculate_computational_latency, calculate_latency_betwen_nodes

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
                 list_graph: List[Graph],
                 list_sfc: List[SFC],
                 pesos_fatores: Dict[str, float] = None,
                 reward_config: Dict[str, float] = None):
        """
        Inicializa o ambiente de alocação de SFC.
        """
        super().__init__()

        if len(list_graph) != len(list_sfc):
            raise ValueError("A lista de grafos deve ter o mesmo tamanho da lista de SFCs.")

        # --- Parâmetros de Configuração ---
        self.valid_nodes = valid_nodes
        self.list_graph = list_graph
        self.list_sfc = list_sfc
        self.pesos_fatores = pesos_fatores if pesos_fatores is not None else \
                             {"cpu": 1.0, "cache": 1.0, "lat": 1.0, "band": 1.0}
        

        if reward_config is None:
            self.reward_config = {"success_bonus": 1000.0, "failure_penalty": -1000.0}
        else:
            self.reward_config = reward_config
        
        self.initial_resource_snapshot = self._initialize_snapshots(list_graph)

        # --- Estado do Episódio ---
        self.graph: Graph = None
        self.current_sfc: SFC = None
        self.current_vnf: VNF = None
        self.current_location: Union[int, str] = None
        self.latency_used = 0
        self.latency_request = None
        self.is_training = True

        # --- Espaços de Ação e Observação ---
        self.action_space = spaces.Discrete(len(self.valid_nodes))
        num_nodes = len(valid_nodes)

        self.observation_space = spaces.Dict({
        "sfc_recursos_requeridos": spaces.Box(low=0, high=1.5, shape=(4, 3), dtype=np.float32),
        "vnf_atual": spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
        "latencia_ja_usada": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "ultimo_no_escolhido": spaces.Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32),
        "recursos_nos_validos": spaces.Box(low=0, high=1, shape=(num_nodes, 5), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o início de um novo episódio.
        Seleciona aleatoriamente um grafo e uma sfc de um caso real,
        restaura o estado inicial dos recursos e prepara a primeira SFC para alocação.
        """
        super().reset(seed=seed)

        idx = np.random.randint(len(self.list_graph))
        self.graph = self.list_graph[idx]
        snapshot_nodes = self.initial_resource_snapshot[idx]['nodes']
        for node_id, initial_state in snapshot_nodes.items():
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id]['cpu_used'] = initial_state['cpu_used']
                self.graph.nodes[node_id]['cache_used'] = initial_state['cache_used']

        snapshot_edges = self.initial_resource_snapshot[idx]['edges']
        for (u, v), initial_state in snapshot_edges.items():
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v]['bandwidth_used'] = initial_state['bandwidth_used']

        # Configura a primeira SFC e reinicia as variáveis de estado do episódio
        sfc_sorteada = self.list_sfc[idx]
        self.set_current_sfc(sfc_sorteada)
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        obs = self._get_obs()
        
        return obs, {}

    def step(self, action: int):
        """
        Executa um passo no ambiente a partir de uma ação do agente.
        A ação corresponde à escolha de um nó para alocar a VNF atual.
        """
        # 1. Traduzir a ação para um nó do grafo
        chosen_server = self.current_sfc.dst_node if action == len(self.valid_nodes) - 1 else self.valid_nodes[action]
        band_req = self.get_band_req(self.current_vnf)

        # 2. Tentar alocar recursos (CPU/cache) no nó escolhido
        if not self.allocate_resources_on_node(chosen_server, self.current_vnf):
            return self._fail_step('resource')

        # 3. Encontrar e alocar recursos no caminho (banda)
        path = get_available_shortest_path_optimized(
            self.graph, self.current_location, chosen_server,
            band_req, rounded=True
        )
        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            return self._fail_step('bandwidth')

        # 4. Calcular latência e verificar restrição
        latency_total = calcular_latencia_total(path, self.graph, self.current_vnf)
        self.latency_used += latency_total
        if self.latency_used > self.latency_request:
            return self._fail_step('latency')

        # 5. Calcular custo e recompensa
        self.servers_used.append(chosen_server)
        total_cost = self.calculate_total_cost(chosen_server, path)
        reward = -total_cost

        # 6. Atualizar estado para o próximo passo
        self.current_location = chosen_server
        if not self.is_training:
            self.allocation_results[self.current_vnf.id] = {'allocated_server': chosen_server, 'path': path, 'cost': total_cost}

        # 7. Verificar conclusão e avançar para a próxima VNF/SFC
        done = False
        if self.current_vnf == self.reverse_vnf_list[-1]:
            done = True
            self.success = True
            reward += self.reward_config['success_bonus']

        else:
            idx=self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]
        obs = self._get_obs()
        
        # 🚀 CORREÇÃO: Retorne todos os valores, incluindo o dicionário 'info'.
        return obs, reward, done, False, {}

    def render(self):
        pass

    def close(self):
        pass

    # =================================================================================
    # 2. Lógica Central da Simulação e Estado
    # =================================================================================

# Substitua seu método _get_obs por este

    def _get_node_features(self, node_id: any, current_location: any, current_band_req: float) -> tuple[np.ndarray, float]:
        """Calcula o vetor de features para um único nó e retorna a capacidade de banda do caminho."""
        # Features: [cpu_used, cache_used, latency, band_used, reusable]
        features = np.zeros(5, dtype=np.float32)
        
        # 1. Recursos do nó
        node_data = self.graph.nodes[node_id]
        features[0] = node_data["cpu_used"] / node_data["cpu_capacity"]
        features[1] = node_data["cache_used"] / node_data["cache_capacity"]
        
        # 2. Reusabilidade
        features[4] = float(self.is_reusable_at_node(self.current_sfc, self.graph, node_id, self.current_vnf))

        # 3. Features de Rede (Latência e Banda)
        path = get_available_shortest_path_optimized(self.graph, current_location, node_id, current_band_req, rounded=True)
        
        if not path:
            features[2] = 1.0  # Latência "infinita" normalizada
            features[3] = 1.0  # Uso de banda "infinito" normalizado
            return features, 0.0 # Retorna 0 para a capacidade de banda do caminho
        
        # Se existe um caminho...
        features[2] = calcular_latencia_total(path, self.graph, self.current_vnf) / self.latency_request
        
        band_cap, band_used = self.get_critical_link_info(path)
        if band_cap > 0:
            features[3] = band_used / band_cap
        else:
            features[3] = 1.0  # CORREÇÃO: Se capacidade é 0, o uso é efetivamente 100%
            
        return features, band_cap

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Monta a observação do ambiente de forma estruturada e legível."""
        valid_nodes = self.valid_nodes
        num_valid_nodes = len(valid_nodes)
        num_vnfs = len(self.reverse_vnf_list)

        # --- 1. Preparação dos Vetores de Estado ---
        # Normaliza a latência já consumida
        latencia_ja_usada = np.array([self.latency_used / self.latency_request], dtype=np.float32)

        # One-hot encode do último nó escolhido
        ultimo_no_escolhido = np.zeros(num_valid_nodes, dtype=np.float32)
        current_loc = self.current_location if not isinstance(self.current_location, str) else 'M'
        idx_loc = valid_nodes.index(current_loc)
        ultimo_no_escolhido[idx_loc] = 1.0

        # --- 2. Coleta de Features dos Nós Válidos ---
        current_band_req = self.get_band_req(self.current_vnf)
        
        node_features_list = []
        total_band_cap = 0.0
        
        # Usa a função auxiliar para obter features de cada nó
        for i, node_id in enumerate(valid_nodes):
            # Caso especial: o último nó "válido" é sempre o destino do SFC
            if i == num_valid_nodes - 1:
                node_id = self.current_sfc.dst_node
            
            features, band_cap = self._get_node_features(node_id, self.current_location, current_band_req)
            node_features_list.append(features)
            total_band_cap += band_cap
        
        recursos_nodes = np.array(node_features_list, dtype=np.float32)

        # --- 3. Coleta de Features do SFC ---
        sfc_recursos_requeridos = np.zeros((num_vnfs, 3), dtype=np.float32)
        lista_vnf_atual = np.zeros(num_vnfs, dtype=np.float32)
        
        # Evita divisão por zero
        mean_band_cap = (total_band_cap / num_valid_nodes) if num_valid_nodes > 0 else 1.0

        for idx, vnf in enumerate(self.reverse_vnf_list):
            sfc_recursos_requeridos[idx, 0] = vnf.get_cpu_request() / 100.0
            sfc_recursos_requeridos[idx, 1] = vnf.get_cache_request() / 100.0
            # CORREÇÃO: Usar a requisição de banda, não de cache
            sfc_recursos_requeridos[idx, 2] = self.get_band_req(vnf) / mean_band_cap if mean_band_cap > 0 else 1.0
            
            if vnf == self.current_vnf:
                lista_vnf_atual[idx] = 1.0
                
        # --- 4. Retorno do Dicionário de Observação ---
        return {
            "sfc_recursos_requeridos": sfc_recursos_requeridos,
            "vnf_atual": lista_vnf_atual,
            "latencia_ja_usada": latencia_ja_usada,
            "ultimo_no_escolhido": ultimo_no_escolhido,
            "recursos_nos_validos": recursos_nodes,
        }

    def _is_node_valid_for_placement(self, node_id: any, cpu_req: float, cache_req: float, band_req: float) -> bool:
        """
        Verifica se um único nó é um alvo válido para alocação da VNF atual.
        
        Retorna True se todos os requisitos (recursos e rede) forem atendidos, False caso contrário.
        """
        # 1. Verificação de recursos do nó (CPU e Cache)
        # Pula a verificação de recursos se a VNF puder ser reusada no nó.
        if not self.is_reusable_at_node(self.current_sfc, self.graph, node_id, self.current_vnf):
            node_data = self.graph.nodes[node_id]
            if (node_data["cpu_capacity"] - node_data["cpu_used"]) < cpu_req:
                return False
            if (node_data["cache_capacity"] - node_data["cache_used"]) < cache_req:
                return False

        # 2. Verificação de Rede (Banda e Latência)
        # Procura por um caminho que suporte a banda necessária
        path = get_available_shortest_path_optimized(self.graph, self.current_location, node_id, band_req, rounded=True)
        if not path:
            return False

        # Verifica se a latência acumulada não estoura o limite do SFC
        path_latency = calcular_latencia_total(path, self.graph, self.current_vnf)
        if (self.latency_used + path_latency) > self.latency_request:
            return False

        # Se todas as verificações passaram, o nó é válido
        return True


    def action_masks(self) -> np.ndarray:
        """
        Cria uma máscara de ações válidas para a decisão atual de forma declarativa.

        Returns:
            np.ndarray: Um array binário (máscara) de ações válidas [1, 0, 0, 1, ...].
        """
        # Se não houver VNF para alocar, nenhuma ação é possível.
        if self.current_vnf is None:
            return np.zeros(len(self.valid_nodes), dtype=np.int8)
        
        # Obtém os requisitos da VNF atual uma única vez.
        cpu_req = self.current_vnf.get_cpu_request()
        cache_req = self.current_vnf.get_cache_request()
        band_req = self.get_band_req(self.current_vnf)

        # Usa List Comprehension para construir a máscara de forma declarativa.
        # Para cada nó, chama a função de validação e o resultado (True/False)
        # é usado para construir a lista, que é então convertida para um array numpy.
        mask = [
            self._is_node_valid_for_placement(
                node_id=(self.current_sfc.dst_node if i == len(self.valid_nodes) - 1 else node_id),
                cpu_req=cpu_req,
                cache_req=cache_req,
                band_req=band_req
            )
            for i, node_id in enumerate(self.valid_nodes)
        ]
        
        return np.array(mask, dtype=np.int8)
    
    def allocate_resources_on_node(self, node_id: Union[int, str], vnf: VNF) -> bool:
        """
        Aloca CPU e Cache em um nó, considerando o reuso de serviços.
        Retorna True se a alocação for bem-sucedida, False caso contrário.
        """
        node = self.graph.nodes[node_id]
        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        
        # Se o serviço for reutilizável, o custo efetivo de recursos é zero
        can_reuse = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
        effective_cpu_req = 0 if can_reuse else cpu_req
        effective_cache_req = 0 if can_reuse else cache_req
        
        # Verifica se há capacidade disponível para a alocação
        if (node['cpu_used'] + effective_cpu_req > node['cpu_capacity']) or \
           (node['cache_used'] + effective_cache_req > node['cache_capacity']):
            return False

        # Aloca os recursos e atualiza os metadados do serviço
        node['cpu_used'] += effective_cpu_req
        node['cache_used'] += effective_cache_req
        
        # if 'services' not in node:
        #     node['services'] = {}
            
        # session_id = self.current_sfc.id.split("_")[-1]
        # service_key = (vnf.id, session_id)
        
        # if service_key in node['services']:
        #     node['services'][service_key]['copys'] += 1
        # else:
        #     node['services'][service_key] = {'cpu': cpu_req, 'cache': cache_req, 'copys': 1}
            
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
            if available_bw < bandwidth_required + 1e-9: # Tolerância para ponto flutuante
                return False

        # 2. Se a verificação passou, alocar a banda em todos os links
        for u, v in zip(path[:-1], path[1:]):
            self.graph.edges[u, v]['bandwidth_used'] += bandwidth_required
        
        return True
    

    def _set_list_graph_sfcs(self, list_graph: List[Graph] ,list_sfc: List[SFC]):
        if len(list_graph) != len(list_sfc):
            raise Exception("O tamanho da lista de grafos deve ser igual ao de SFCs para correspondência")
        else: 
            self.list_graph = list_graph
            self.list_sfc = list_sfc
            self.reset()


    def _fail_step(self, reason: str):
        """
        Finaliza um episódio com falha, aplicando uma penalidade alta.
        """
        self.fail_reason = reason
        self.success = False
        
        # --- USA A PENALIDADE CONFIGURADA ---
        reward = self.reward_config['failure_penalty']
        
        done = True
        # 🚀 CORREÇÃO: Garanta que mesmo em falha, a última observação e info sejam retornados.
        obs = self._get_obs()
        
        return obs, reward, done, False, {}
    
    def _initialize_snapshots(self, list_graph: List[Graph] = None):
        """
        Cria um snapshot do estado inicial dos recursos de todos os grafos
        para garantir um reset consistente dos episódios.
        """
        initial_resource_snapshot = {}
        for idx, graph in enumerate(list_graph):
            nodes = {n_id: {'cpu_used': data.get('cpu_used', 0),
                            'cache_used': data.get('cache_used', 0)
                            }
                     for n_id, data in graph.nodes(data=True)}
            edges = {(u, v): {'bandwidth_used': data.get('bandwidth_used', 0)}
                     for u, v, data in graph.edges(data=True)}
            initial_resource_snapshot[idx] = {'nodes': nodes, 'edges': edges}
        return initial_resource_snapshot

    # =================================================================================
    # 3. Funções Utilitárias e de Suporte
    # =================================================================================

    def set_current_sfc(self, sfc: SFC):
        """Define a SFC atual para alocação e inicializa seus parâmetros."""
        if not sfc:
            raise ValueError("SFC não pode ser None.")
        self.current_sfc = sfc
        self.reverse_vnf_list = self.define_reverse_vnf_list(sfc)
        self.current_vnf = self.reverse_vnf_list[0]
        self.latency_request = 10  # TODO: Considerar tornar dinâmico
        self.current_location = self.current_sfc.dst_node
        self.latency_used = 0
        self.servers_used = []

    def define_reverse_vnf_list(self, sfc: SFC) -> List[VNF]:
        """Retorna a lista de VNFs da SFC em ordem reversa (do destino para a origem)."""
        vnf_list = []
        dst_vnf = sfc.get_dst_vnf()
        current_vnf = sfc.get_previous_vnf(dst_vnf)
        while True:
            vnf_list.append(current_vnf)
            if current_vnf.previous_vnf is None or current_vnf.previous_vnf.id == 'src':
                break
            current_vnf = sfc.get_previous_vnf(current_vnf)
        return vnf_list


    def get_band_req(self, vnf: VNF = None) -> float:
        if vnf is None:
            raise ValueError("VNF não pode ser None.")    
        return vnf.get_outcome_interface_bandwidth()

    def is_reusable_at_node(self,sfc: SFC, graph: Graph, node_id: Union[int, str], vnf: VNF) -> bool:
        """Verifica se uma VNF compartilhável já está alocada em um nó."""
        service_name = vnf.id

        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()

        node = graph.nodes[node_id]

        cpu_used, cpu_cap = node["cpu_used"], node["cpu_capacity"]
        cache_used, cache_cap = node["cache_used"], node["cache_capacity"]

        if cpu_used+cpu_req >= cpu_cap or cache_used+cache_req >= cache_cap:
            return False

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False
        
        session_id = sfc.id.split("_")[-1]
        service_key = (service_name, session_id)
        
        return service_key in graph.nodes[node_id].get('services', {})

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

        reusable = self.is_reusable_at_node(self.current_sfc, self.graph, server_id, self.current_vnf)
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
            
def calcular_latencia_total(path:list, graph: Graph, vnf: VNF = None) -> float:
    """
    Calcula a latência total de um caminho considerando os links e a latência computacional.
    """
    edge_latency = 0
    if len(path) > 1:
        for u, v in zip(path[:-1], path[1:]):
            edge_latency += calculate_latency_betwen_nodes(graph, u, v, vnf)
    computational_latency = calculate_computational_latency(graph, path[0], vnf)
    total_latency = edge_latency + computational_latency
    return total_latency if total_latency >= 0 else 0.0 