import copy
import gymnasium
from gymnasium import spaces
import numpy as np
from networkx import Graph
from typing import Union, List, Dict
from core.sfc import SFC, VNF
from algorithms.networkUtils import   calculate_latency_betwen_nodes, get_available_shortest_path_fast
from utils.network_utils import get_sfc_latency_from_route
from algorithms.environments.env_utils.utils import calculate_comunication_latency, create_route_info_from_allocation_results
from algorithms.environments.env_utils.utils import calculate_total_latency
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')



class SFC_AllocationEnv_DARSPPO(gymnasium.Env):
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
                 is_training = True):
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
                             {"eta1":1, "eta2":1,"bw":1}
        
        self.is_training = is_training
        if self.is_training:
            self.initial_resource_snapshot=self._initialize_snapshots(self.list_graph)
 
        # --- Estado do Episódio ---
        self.graph: Graph = None
        self.current_sfc: SFC = None
        self.current_vnf: VNF = None
        self.current_location: Union[int, str] = None
        self.features = None

        # --- Espaços de Ação e Observação ---
        num_nodes = len(valid_nodes)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Dict({ 
        #0 se cache e 1 se unique
        "Se": spaces.Box(low=0, high=1, shape=(num_nodes,1), dtype=np.float32),
        "Bw": spaces.Box(low=0, high=1, shape=(num_nodes,1), dtype=np.float32),
        "Sm": spaces.Box(low=0, high=1, shape=(num_nodes, 1), dtype=np.float32),
        "Sr": spaces.Box(low=0, high=1, shape=(num_nodes, 2), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        """
        Reseta o ambiente para o início de um novo episódio.
        Seleciona aleatoriamente um grafo e uma sfc de um caso real,
        restaura o estado inicial dos recursos e prepara a primeira SFC para alocação.
        """
        super().reset(seed=seed)
        self.latency_used = 0
        idx = 0

        if self.is_training:
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

        else:
            self.graph = self.list_graph[idx]
        sfc_sorteada = self.list_sfc[idx]
        
        self.set_current_sfc(sfc_sorteada)
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        vnf = self.current_vnf
        bw_req = self.service_requirements[vnf.id]["out_bw"]
        current_node = self.current_location
        self.allocation_results['dst'] = {'allocated_server': sfc_sorteada.dst_node, 'path': [], 'cost': 0}
        self._deploy_initial_solution()
        
        self.features = self._get_nodes_features(vnf, bw_req, current_node)

        obs = self._get_obs()
        
        return obs, {}

    def step(self, action: int):
        """
        Executa um passo no ambiente a partir de uma ação do agente.
        A ação corresponde à escolha de um nó para alocar a VNF atual.
        """
        # 1. Traduzir a ação para um nó do grafo
        if action == len(self.valid_nodes) - 1:
            chosen_server = self.current_sfc.dst_node
        else: chosen_server = self.valid_nodes[action]

        vnf = self.current_vnf
        band_req = self.service_requirements[vnf.id]['out_bw']
        current_location = self.current_location
        path = get_available_shortest_path_fast(self.graph, current_location, chosen_server, band_req)
        
        
        # 2. Tentar alocar recursos (CPU/cache) no nó escolhido
        if not self.allocate_resources_on_node(chosen_server, self.current_vnf):
            return self._fail_step('resource')
        
        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            return self._fail_step('bandwidth')

        self.servers_used.append(chosen_server)

        self.latency_used += calculate_total_latency(self.graph, path, vnf)

        dst = self.current_sfc.get_substrate_node(self.current_sfc.get_dst_vnf())
        past_route_info = create_route_info_from_allocation_results(dst, self.graph, self.allocation_results)
        past_delay = get_sfc_latency_from_route(self.graph, self.current_sfc,past_route_info)
        self.allocation_results[vnf.id]["allocated_server"] = chosen_server
        self.allocation_results[vnf.id]["path"] = path
        
        route_info = create_route_info_from_allocation_results(dst, self.graph, self.allocation_results)
        current_delay = get_sfc_latency_from_route(self.graph, self.current_sfc,route_info)
            
        # 6. Atualizar estado para o próximo passo
        self.current_location = chosen_server
        

        # 7. Verificar conclusão e avançar para a próxima VNF/SFC
        done = False
        if self.current_vnf == self.reverse_vnf_list[-1]:
            bw_cost = self.features[action,1]
            reward = self._calculate_reward(current_delay,past_delay, self.initial_delay, True, bw_cost)
            done = True
            self.success = True
            self.current_vnf = None

        else:
            bw_cost = self.features[action,1]
            reward = self._calculate_reward(current_delay,past_delay, self.initial_delay, False, bw_cost)
            idx=self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]

        bw_required = self.service_requirements[self.current_vnf.id]['out_bw'] if self.current_vnf else 0
        self.features=self._get_nodes_features(self.current_vnf, bw_required, self.current_location)
        obs = self._get_obs()

        
        # 🚀 CORREÇÃO: Retorne todos os valores, incluindo o dicionário 'info'.
        return obs, reward, done, False, {}


    # =================================================================================
    # 2. Lógica Central da Simulação e Estado
    # =================================================================================





    def _get_nodes_features(self, vnf: VNF, bw_required, current_location: any) -> np.ndarray:
        """
        Calcula o vetor de features para cada nó candidato.
        Inclui lógicas especiais com a seguinte prioridade:
        1. Prioriza "nós dourados" (reuso com baixo custo), se existirem.
        2. Se não houver "nó dourado", aplica regras para VNF de cache ou "unique".
        """

        # Features: []
        num_valid_nodes = len(self.valid_nodes)
        features = np.zeros((num_valid_nodes, 6))

        if not vnf:
            # Se não houver VNF para alocar, retorna features zeradas, marcando todos como inválidos
            features[:, 5] = 1 
            return features

        # --- 1. Loop principal para calcular as features de cada nó (sem alterações) ---
        for i, node_id in enumerate(self.valid_nodes):
            # ... (código do loop, sem alterações) ...
            if i == num_valid_nodes - 1:
                node_id = self.current_sfc.dst_node
                
            node_data = self.graph.nodes[node_id]
            is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
            features[i, 2] = float(is_reusable)
            cpu_req = vnf.get_cpu_request() if not is_reusable else 0
            cache_req = vnf.get_cache_request() if not is_reusable else 0

            features[i, 3] = (node_data["cpu_capacity"] - node_data["cpu_used"]) / node_data["cpu_capacity"]
            features[i, 4] = (node_data["cache_capacity"] - node_data["cache_used"]) / node_data["cache_capacity"]
            if (node_data["cpu_used"] + cpu_req) > node_data["cpu_capacity"] or \
               (node_data["cache_used"] + cache_req) > node_data["cache_capacity"]:
                features[i, 5] = 1
            path = get_available_shortest_path_fast(self.graph, current_location, node_id, bw_required)
            if not path:
                features[i, 0] = 1.0
                features[i, 1] = 1.0
                features[i, 5] = 1.0
            else:
                bw_cost, latency_cost = self.calculate_bw_lat_cost(vnf, node_id, path, bw_required)
                features[i, 0] = latency_cost
                features[i, 1] = bw_cost

        return features


    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Monta a observação do ambiente de forma estruturada e eficiente usando NumPy.
        """

        # --- 1. Determinação do Último Nó Escolhido ---
        # Se = np.zeros(num_valid_nodes, dtype=np.float32)
        # Sm = np.zeros(1, dtype=np.float32)
        # Sr = np.zeros(1, dtype=np.float32)

        

        # --- 2. Coleta de Features dos Nós Válidos (Versão Otimizada) ---
        # self.features é um array NumPy com as colunas:
        # [0:cpu, 1:cache, 2:reusable, 3:path(não usado aqui), 4:band, 5:latency, 6:is_invalid]

        # Define as colunas que queremos selecionar do array self.features
        # para formar nossa observação.
        idx_Se = [0]
        idx_bw = [1]
        idx_Sm = [2]
        idx_Sr = [3,4]

        # Usa o fatiamento avançado do NumPy para selecionar todas as linhas
        # e apenas as colunas desejadas de uma só vez.
        # Isso elimina a necessidade de um loop em Python, sendo muito mais rápido.
        Se = self.features[:, idx_Se].astype(np.float32)
        Bw = self.features[:, idx_bw].astype(np.float32)
        Bw = Bw/10
        Sm = self.features[:, idx_Sm].astype(np.float32)
        Sr = self.features[:, idx_Sr].astype(np.float32)

        # Normaliza a coluna de latência (que agora é a coluna de índice 4 no novo array)
        # A operação é feita em toda a coluna de uma vez.
        Se[:, 0] /= 100

        obs = {
            "Se": Se ,
            "Bw": Bw ,
            "Sm": Sm,
            "Sr": Sr
        }

        return obs



    def action_masks(self) -> np.ndarray:
        """
        Cria uma máscara de ações válidas para a decisão atual de forma declarativa.

        Returns:
            np.ndarray: Um array binário (máscara) de ações válidas [1, 0, 0, 1, ...].
        """
        # Se não houver VNF para alocar, nenhuma ação é possível.
        
        if self.current_vnf is None:
            return np.zeros(len(self.valid_nodes), dtype=np.int8)
        
        if self.features is None:
            vnf=self.current_vnf
            bw_req = self.service_requirements[vnf.id]["out_bw"]
            current_node = self.current_location
            self.features = self._get_nodes_features(vnf, bw_req, current_node)
        
        mask = [
            1 if self.features[i, 5] == 0 else 0
            for i, _ in enumerate(self.valid_nodes)
        ]
        if np.nan in mask:
            epa = 1
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
            # print("RESET EM environment no _set_list_graph_sfcs")


    def _fail_step(self, reason: str):
        """
        Finaliza um episódio com falha, aplicando uma penalidade alta.
        """
        self.fail_reason = reason
        # print(f"Causa da falha: {reason}")
        self.success = False
        
        # --- USA A PENALIDADE CONFIGURADA ---
        reward = -100
        
        done = True
        # 🚀 CORREÇÃO: Garanta que mesmo em falha, a última observação e info sejam retornados.
        bw_required = self.service_requirements[self.current_vnf.id]['out_bw']
        self.features = self._get_nodes_features(self.current_vnf, bw_required, self.current_location)
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
    
    def set_current_sfc(self, sfc: SFC):
        """Define a SFC atual para alocação e inicializa seus parâmetros."""
        if not sfc:
            raise ValueError("SFC não pode ser None.")
        self.current_sfc = sfc
        self.reverse_vnf_list = self.define_reverse_vnf_list(sfc)
        self.current_vnf = self.reverse_vnf_list[0]
        self.current_location = self.current_sfc.dst_node
        self.servers_used = []

        service_requirements = {} 
        services = []  # Lista para guardar os nomes
        sfs_dict = sfc.vnfs_dict
        
        for item in sfs_dict:
            nome = item['name']
            services.append(nome)  # Adiciona o nome à lista de nomes
            service_requirements[nome] = {
                'cpu': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw']}

        if True:
            services.append('dst')
            service_requirements['dst'] = {'cpu': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0}  
        
        self.service_requirements = service_requirements

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

    def is_reusable_at_node(self,sfc: SFC, graph: Graph, node_id: Union[int, str], vnf: VNF) -> bool:
        """Verifica se uma VNF compartilhável já está alocada em um nó."""
        if not vnf:
            return False
        service_name = vnf.id

        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()

        node = graph.nodes[node_id]

        cpu_used, cpu_cap = node["cpu_used"], node["cpu_capacity"]
        cache_used, cache_cap = node["cache_used"], node["cache_capacity"]

        if not service_name.startswith(SHAREABLE_PREFIXES):
            return False
        
        session_id = sfc.id.split("_")[-1]
        service_key = (service_name, session_id)
        result = service_key in graph.nodes[node_id].get('services', {})

        if result and cpu_used+cpu_req >= cpu_cap and cache_used+cache_req >= cache_cap:
            return False

        return result

    def calculate_bw_lat_cost(self, vnf: VNF, server_id, path: List, bw_required: float):
        # Latência computacional

        if not path or len(path) < 2:
            return 0, 0
        latency_cost = 0
        bw_cost = 0
        for u, v in zip(path[:-1], path[1:]):
            # Acesso à aresta da rede
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get('bandwidth_capacity', None)
            bd_used = edge.get('bandwidth_used', 0)

            # Calcula latência do enlace
            latency_cost += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

            # Se a capacidade de banda for insuficiente, retorna custo infinito
            if bd_capacity is None or bd_capacity == 0 or bw_required + bd_used > bd_capacity:
                return float(999), latency_cost

            # Cálculo do custo de banda
            link_cost = (bw_required + bd_used) / bd_capacity
            bw_cost += link_cost

        return bw_cost, latency_cost

    
    def _calculate_reward(self, current_delay: float,past_delay: float,initial_delay: float, is_last_step: bool, bw_cost = 0) -> float:
        """
        Calcula a recompensa para a ação atual usando a função de duas etapas
        descrita na Equação (20) do artigo.

        Args:
            current_delay (float): O atraso médio total calculado após a ação atual.
            is_last_step (bool): Um indicador se esta foi a última ação do episódio.

        Returns:
            float: O valor da recompensa a ser retornado para o agente.
        """
        # Obtém os pesos eta1 e eta2 da configuração do ambiente.
        # Estes são os fatores de ponderação da Equação (20).
        eta1 = self.pesos_fatores['eta1']
        eta2 = self.pesos_fatores['eta2']
        bw_factor = self.pesos_fatores['bw']

        # A recompensa é baseada na *redução* do atraso. Um atraso menor é melhor.
        # Usamos (atraso_anterior - atraso_atual) para que uma redução no atraso
        # resulte em uma recompensa positiva, incentivando o agente.
        # Esta é a primeira parte da recompensa da Equação (20).
        current_action_reward = past_delay - current_delay

        # A recompensa base é sempre calculada.
        reward = eta1 * current_action_reward

        # Se for a última ação do episódio (t = N), adicionamos a segunda parte da recompensa.
        if is_last_step:
            # Esta parte compara o desempenho final deste episódio com o episódio anterior.
            # Incentiva o agente não apenas a melhorar a cada passo, mas também a
            # alcançar um resultado final melhor do que o da última vez.
            last_step_reward = initial_delay - current_delay
            reward += eta2 * last_step_reward

        reward -= bw_cost*bw_factor
        
        return reward
    


    # Coloque esta função DENTRO da sua classe SFC_AllocationEnv

    def _deploy_initial_solution(self):
        """
        Realiza uma alocação inicial completa para a self.current_sfc usando uma
        heurística baseada na menor latência de comunicação, conforme sugerido pelo artigo.
        O processo é feito na ordem inversa (do destino para a origem), seguindo
        a lógica existente no ambiente.

        Retorna:
            bool: True se a alocação inicial foi bem-sucedida, False caso contrário.
        """
        # A lista de VNFs já está em ordem reversa (do destino para a origem)
        vnf_list = self.reverse_vnf_list
        
        # O ponto de partida para a primeira VNF é o destino final da SFC
        dst=self.current_sfc.dst_node
        previous_node_location = dst
        
        # Vamos rastrear o caminho completo da solução inicial
        # Começamos com o destino e vamos adicionando os nós no início da lista

        # Itera sobre cada VNF que precisa ser alocada
        for vnf in vnf_list:
            best_node_for_vnf = None
            min_latency_found = float('inf')
            best_path_segment = None
            
            bw_required = self.service_requirements[vnf.id]['out_bw']

            # Testa todos os nós válidos como possíveis locais para a VNF atual
            for candidate_node in self.valid_nodes:
                if candidate_node == "M":
                    candidate_node = self.current_sfc.dst_node
                node_data = self.graph.nodes[candidate_node]
                reusable = self.is_reusable_at_node(self.current_sfc, self.graph, candidate_node,vnf)
                cpu_req = vnf.get_cpu_request() if not reusable else 0
                cache_req = vnf.get_cache_request() if not reusable else 0
                
                if (node_data['cpu_used'] + cpu_req > node_data['cpu_capacity']) or \
                (node_data['cache_used'] + cache_req > node_data['cache_capacity']):
                    continue # Pula para o próximo nó candidato se não houver recursos

                # 2. Encontrar o caminho e a latência de comunicação
                # O caminho é do nó candidato ATÉ o local da VNF anterior (que na verdade é a próxima na cadeia)
                path_segment = get_available_shortest_path_fast(self.graph, candidate_node, previous_node_location, bw_required)
                
                if path_segment:
                    # Use a função que criamos para calcular apenas a latência de comunicação
                    communication_latency = calculate_comunication_latency(self.graph, path_segment, vnf)
                    
                    # 3. Verificar se este candidato é a melhor opção até agora
                    if communication_latency < min_latency_found:
                        min_latency_found = communication_latency
                        best_node_for_vnf = candidate_node
                        best_path_segment = path_segment

            # 4. Após testar todos os candidatos, alocar no melhor nó encontrado
            if best_node_for_vnf is not None:
            
                # Atualiza a localização para a próxima iteração do loop
                previous_node_location = best_node_for_vnf
                
                # Adiciona o nó escolhido no início da lista do caminho completo
                self.allocation_results[vnf.id] = {'allocated_server': best_node_for_vnf, 'path': best_path_segment, 'cost': 0}
  
            else:
                # Se nenhum nó válido foi encontrado para esta VNF, a alocação inicial falhou
                return False
        
        G=self.graph
        route_info=create_route_info_from_allocation_results(dst, G, self.allocation_results)
        self.initial_delay = get_sfc_latency_from_route(G, self.current_sfc, route_info)
        return True
    


         
            

