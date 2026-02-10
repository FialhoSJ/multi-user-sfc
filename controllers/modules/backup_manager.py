import copy
import time
import random
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import defaultdict
from controllers.sfc_generator import SFCGenerator
from core.net_v2 import Net2
from core.sfc import SFC
# Assumindo que o ambiente RL esteja disponível neste caminho
from algorithms.environments.env_sbrc import SFC_AllocationEnv

class BackupManager:
    """
    Gerencia o ciclo de vida lógico e as estratégias de criação de backups.
    
    Responsabilidades:
    1. Definir estratégias de alocação (Greedy, RL, Seletiva).
    2. Identificar backups obsoletos para coleta de lixo.
    3. Manter o registro lógico de quais backups pertencem a quais SFCs.
    
    Nota: Esta classe NÃO deve manipular a rede física diretamente (undeploy).
    """

    def __init__(self, args):
        # Mapeia: ID da SFC Original -> Lista de metadados dos backups
        # Ex: {'sfc_1': [{'sfc_backup_id': 'sfc_1_bk', 'vnf_id': 'vnf1', ...}]}
        self.sfcs_backups_instatiated: Dict[str, List[Dict[str, Any]]] = {}
        
        # Mapeia: ID do Backup -> ID da SFC Original
        # Ex: {'sfc_1_bk': 'sfc_1'}
        self.backups_sfc_instantiated: Dict[str, str] = {}
        
        # Lista de backups que estão ativos (assumiram o lugar do primário)
        self.backups_activated: List[str] = []
        
        target_algorithm = 'SBRCMASKABLEPPO'
        is_target_alg = (args.alg == target_algorithm)
        user_wants_backup = (args.backup == 'y') and (args.ava != '1.0')
        
        self.backup_activated = user_wants_backup and is_target_alg
        self.alg = args.alg
        
        if user_wants_backup and not is_target_alg:
            print(f"[BackupManager] INFO: Backup proativo desativado. '{args.alg}' não suporta essa estratégia.")

        self.standard_reduction_factor = 1.0

    # ==========================================
    # Lifecycle & State Management (Refatorado)
    # ==========================================

    def identify_obsolete_backups(self) -> List[str]:
        """
        Identifica backups que podem ser removidos com base na estratégia do algoritmo
        (ex: estratégia probabilística do Vegeta/GA).
        
        Returns:
            List[str]: Lista de IDs de backups que devem ser removidos pelo Controller.
        """
        backups_to_remove = []
        
        # Se o algoritmo não for um destes, não fazemos limpeza proativa probabilística
        if self.alg not in ['vegeta', 'ga']:
            return []

        # Itera sobre uma cópia das chaves para segurança
        for backup_id in list(self.backups_sfc_instantiated.keys()):
            # Se o backup já foi ativado (está segurando o tráfego), não remova!
            if backup_id in self.backups_activated:
                continue
            
            # Estratégia probabilística: 70% de chance de limpar backups ociosos
            if random.random() < 0.7:
                backups_to_remove.append(backup_id)
                
        return backups_to_remove

    def cleanup_internal_state(self, backup_id: str) -> None:
        """
        Remove os registros lógicos de um backup. Deve ser chamado pelo Controller
        SOMENTE APÓS a remoção física ter sido bem sucedida.
        
        Args:
            backup_id (str): O ID da SFC de backup que foi removida.
        """
        # 1. Identifica a SFC original dona deste backup
        original_sfc = self.backups_sfc_instantiated.get(backup_id)

        # 2. Remove da lista da SFC original
        if original_sfc and original_sfc in self.sfcs_backups_instatiated:
            backups_list = self.sfcs_backups_instatiated[original_sfc]
            
            # Filtra a lista mantendo apenas os OUTROS backups
            self.sfcs_backups_instatiated[original_sfc] = [
                b for b in backups_list if b.get("sfc_backup_id") != backup_id
            ]

            # Se a lista ficar vazia, remove a entrada da SFC original para economizar memória
            if not self.sfcs_backups_instatiated[original_sfc]:
                del self.sfcs_backups_instatiated[original_sfc]
        
        # 3. Remove o mapeamento reverso
        if backup_id in self.backups_sfc_instantiated:
            del self.backups_sfc_instantiated[backup_id]
            
        # 4. Remove da lista de ativados, se estiver lá
        if backup_id in self.backups_activated:
            self.backups_activated.remove(backup_id)

    def register_backup_deployment(self, original_sfc_id: str, backup_sfc_id: str, 
                                 vnf_id: str, route_info: Dict) -> None:
        """
        Registra um novo backup implantado com sucesso.
        """
        if original_sfc_id not in self.sfcs_backups_instatiated:
            self.sfcs_backups_instatiated[original_sfc_id] = []

        self.sfcs_backups_instatiated[original_sfc_id].append({
            "sfc_backup_id": backup_sfc_id,
            "vnf_id": vnf_id,
            "route_info": route_info
        })
        self.backups_sfc_instantiated[backup_sfc_id] = original_sfc_id

    # ==========================================
    # Creation Strategies
    # ==========================================

    def create_backups(self, network: Net2, agent_ref: Any = None) -> Tuple[List[Any], str]:
        """Orquestra a criação e o registro de backups baseado na estratégia definida.
        
        Atua como uma fachada para as estratégias de alocação, garantindo que
        qualquer backup gerado seja imediatamente registrado no estado interno.

        Args:
            network: A instância da rede de substrato atual.
            agent_ref: Referência opcional para o agente de RL (se aplicável).

        Returns:
            Uma tupla contendo a lista de SFCs de backup criadas e o nome da estratégia.
        """
        if not self.backup_activated:
            return [], "none"

        # ... (código de preparação do sfc_id_duration mantém-se igual) ...
        # Apenas para contexto, mantive a lógica de dicionário auxiliar aqui
        sfc_id_duration = {} 
        for sfc_id, sfc in network.sfc_dict.items():
            if "backup" in sfc_id: continue
            start_t = getattr(sfc, 'arrival_time', time.time())
            sfc_id_duration[sfc_id] = {"timer": start_t, "duration": getattr(sfc, 'duration', 100)}

        # 1. Seleção e Execução da Estratégia
        backups_mount: List[List[Any]] = []
        strategy_name: str = "greedy"

        if self.alg == 'SBRCMASKABLEPPO' and agent_ref is not None:
            backups_mount = self.rl_based_strategy(network, sfc_id_duration, agent_ref)
            strategy_name = 'rl_based'
        elif self.alg in ['vegeta', 'ga']:
            backups_mount = self.seletive_strategy(network, sfc_id_duration)
            strategy_name = 'seletive'
        else:
            backups_mount = self.greedy_strategy(network, sfc_id_duration)
            strategy_name = 'greedy'

        # 2. Persistência de Estado (Correção do Bug)
        # O Manager assume a responsabilidade de registrar o que acabou de criar.
        self._commit_backup_state(backups_mount)

        return backups_mount, strategy_name
    

    def _commit_backup_state(self, backups_groups: List[List[Any]]) -> None:
        """Registra internamente os backups recém-criados para evitar duplicidade.
        
        Este método privado encapsula a lógica de atualização de estado,
        impedindo que detalhes de implementação vazem para o Controller.

        Args:
            backups_groups: Lista de listas contendo objetos SFC de backup.
        """
        if not backups_groups:
            return

        for group in backups_groups:
            for backup_sfc in group:
                # Extração segura de atributos usando getattr para robustez [cite: 198]
                original_id = getattr(backup_sfc, 'original_sfc_id', None)
                target_vnf = getattr(backup_sfc, 'target_vnf_id', None)
                route_info = getattr(backup_sfc, 'pre_calculated_route', None)

                # Validação estrita antes do registro
                if original_id and target_vnf:
                    # Se não houver rota pré-calculada (ex: greedy), usamos um dict vazio
                    # ou a lógica específica da sua implementação greedy
                    final_route = route_info if route_info else {}
                    
                    self.register_backup_deployment(
                        original_sfc_id=original_id,
                        backup_sfc_id=backup_sfc.id,
                        vnf_id=target_vnf,
                        route_info=final_route
                    )

    

    def _calc_virtual_reliability(self, network: Net2, sfc_id: str, 
                                pending_backups_sfcs: List[Any]) -> Tuple[float, List[Dict]]:
        """
        Calcula a confiabilidade efetiva da SFC, agrupando VNFs por nó físico (Domínio de Falha).
        
        Boas Práticas Aplicadas:
        - O(N) Lookup: Pré-processamento de backups para evitar loops aninhados.
        - Grouping: Uso de defaultdict para agrupar VNFs por nó.
        - Math: Lógica de RBD (Reliability Block Diagram) para sistemas Série-Paralelo.
        """
        # Fail-fast: Se a rota não existe, não há confiabilidade a calcular.
        if sfc_id not in network.sfc_route_info:
            return 0.0, []

        route_info = network.sfc_route_info[sfc_id]
        
        # ---------------------------------------------------------
        # 1. OTIMIZAÇÃO: Mapa de Backups (VNF ID -> Confiabilidade do Backup)
        # Transforma busca linear O(M) em busca constante O(1)
        # ---------------------------------------------------------
        backup_reliability_map = {}

        # A) Mapeia Backups JÁ Instanciados (Prioridade Alta)
        if sfc_id in self.sfcs_backups_instatiated:
            for b in self.sfcs_backups_instatiated[sfc_id]:
                vnf_id = b.get('vnf_id')
                # Extrai o nó físico da rota do backup
                bk_route = b.get('route_info', {})
                # Busca a chave que termina com '_b' (ex: 'vnf1_b')
                bk_node_list = next((v for k, v in bk_route.items() if k.endswith('_b') and v), None)
                
                if vnf_id and bk_node_list:
                    node_id = bk_node_list[0]
                    backup_reliability_map[vnf_id] = network.get_node_reliability(node_id)

        # B) Mapeia Backups Pendentes (Apenas se ainda não houver um instanciado)
        for mini_sfc in pending_backups_sfcs:
            target_vnf = getattr(mini_sfc, 'target_vnf_id', None)
            
            # Só processa se ainda não temos um backup firme para essa VNF
            if target_vnf and target_vnf not in backup_reliability_map:
                route = getattr(mini_sfc, 'pre_calculated_route', {})
                bk_key = f"{target_vnf}_b"
                
                if route and bk_key in route and route[bk_key]:
                    node_id = route[bk_key][0]
                    backup_reliability_map[target_vnf] = network.get_node_reliability(node_id)

        # ---------------------------------------------------------
        # 2. AGRUPAMENTO: Mapeia Nó Físico -> Lista de VNFs
        # (Domínio de Falha: Se o nó cai, todas as VNFs nele caem)
        # ---------------------------------------------------------
        node_groups = defaultdict(list)
        
        for vnf_id, path in route_info.items():
            if vnf_id in ['src', 'dst'] or not path:
                continue
            # path[0] é o nó físico onde a VNF está alocada
            node_groups[path[0]].append(vnf_id)

        # ---------------------------------------------------------
        # 3. CÁLCULO: Confiabilidade Série-Paralelo
        # ---------------------------------------------------------
        total_reliability = 1.0
        candidates = []

        for node_id, vnfs_list in node_groups.items():
            # Obtém confiabilidade do nó primário (com fallback seguro)
            try:
                reliability_primary = network.get_node_reliability(node_id)
            except (AttributeError, KeyError):
                reliability_primary = 1.0

            # Verifica proteção para o GRUPO
            all_vnfs_protected = True
            prod_failure_backups = 1.0 # Probabilidade de falha conjunta dos backups
            
            for vnf_id in vnfs_list:
                reliability_backup = backup_reliability_map.get(vnf_id, 0.0)
                
                if reliability_backup > 0.0:
                    # Se tem backup, acumulamos a probabilidade de falha dele
                    # P(Falha Backup) = 1 - R_backup
                    prod_failure_backups *= (1.0 - reliability_backup)
                else:
                    # Se UMA VNF do nó não tem backup, o grupo não está totalmente protegido
                    all_vnfs_protected = False
                    candidates.append({
                        'vnf_id': vnf_id,
                        'node_rel': reliability_primary,
                        'node_id': node_id
                    })

            # FÓRMULA DO GRUPO:
            if all_vnfs_protected:
                # Sistema Paralelo: O serviço sobrevive se (Primário Vivo) OU (Backups Vivos)
                # P(Sistema Falhar) = P(Primário Falhar) * P(Todos Backups Falharem)
                # R_total = 1 - P(Sistema Falhar)
                
                prob_primary_fail = 1.0 - reliability_primary
                # Nota: prod_failure_backups já é o produtório de (1 - r_backup)
                
                group_reliability = 1.0 - (prob_primary_fail * prod_failure_backups)
            else:
                # Sistema Série: Limitado pelo elo mais fraco (o nó primário)
                # Se o nó cair, a VNF sem backup cai, e a SFC morre.
                group_reliability = reliability_primary

            total_reliability *= group_reliability

        # Ordena candidatos: Prioridade para nós menos confiáveis
        sorted_candidates = sorted(candidates, key=lambda x: x['node_rel'])
        
        return total_reliability, sorted_candidates
            
    def rl_based_strategy(self, network: Net2, sfc_id_duration: Dict, agent: Any) -> List[List[Any]]:
        """
        Estratégia baseada em RL com 'Shadow State' para garantir consistência 
        de recursos durante o planejamento em lote.
        """
        import time
        import copy # Garantir import
        
        backups_mount = []
        target_reliability = 0.85
        MAX_BACKUPS_PER_SFC = 4
        
        # --- ARQUITETURA: Criação do Shadow State ---
        # Criamos uma cópia MÚTAVEL da topologia atual para simular o consumo
        # de recursos sequencialmente, sem afetar a rede real (Net2) ainda.
        # Isso evita que múltiplos backups sejam agendados para o mesmo slot de recurso.
        simulation_graph = copy.deepcopy(network.graph)
        
        # Ordenação para determinismo
        sorted_sfcs = sorted(list(sfc_id_duration.keys()))

        for sfc_id in sorted_sfcs:
            if "backup" in sfc_id: continue 
            if sfc_id not in network.sfc_dict: continue
            
            sfc = network.get_sfc_by_id(sfc_id)
            pending_sfcs_this_cycle = []
            
            last_reliability = -1.0 
            loop_safety_counter = 0
            MAX_LOOP_ATTEMPTS = 5
            
            while len(pending_sfcs_this_cycle) < MAX_BACKUPS_PER_SFC:
                loop_safety_counter += 1
                
                if loop_safety_counter > MAX_LOOP_ATTEMPTS:
                    break

                current_r, candidates = self._calc_virtual_reliability(
                    network, sfc_id, pending_sfcs_this_cycle
                )
                
                last_reliability = current_r

                if current_r >= target_reliability:
                    break
                
                if not candidates:
                    break

                target_info = candidates[0] 
                target_vnf = target_info['vnf_id']
                weak_node = target_info['node_id']

                mini_sfc = self.create_contextual_mini_sfc(network, sfc, target_vnf, weak_node)
                if not mini_sfc: 
                    break 

                # --- MUDANÇA: Usamos o simulation_graph (estado acumulado) ---
                # NÃO fazemos deepcopy de network.graph aqui dentro.
                # Fazemos deepcopy do simulation_graph para o ENV não estragar
                # o nosso estado sombra caso falhe, mas persistimos se der sucesso.
                graph_for_env = copy.deepcopy(simulation_graph)
                
                self._enrich_graph_with_mobility(graph_for_env, network, mini_sfc)

                valid_types = ['server', 'mobile_device']
                all_servers = [n for n, d in graph_for_env.nodes(data=True) if d.get('type') in valid_types]
                
                if not all_servers:
                    break 

                env = SFC_AllocationEnv(
                    valid_nodes=all_servers,
                    list_graph=[graph_for_env],
                    list_sfc=[mini_sfc],
                    is_training=False
                )
                
                # --- Lógica de Forbidden Nodes (Mantida igual) ---
                primary_nodes_used = set()
                original_route_info = network.sfc_route_info.get(sfc_id, {})
                for vnf_p, path_p in original_route_info.items():
                    if vnf_p not in ['src', 'dst'] and path_p:
                        primary_nodes_used.add(path_p[0])

                for pending_bk in pending_sfcs_this_cycle:
                    if hasattr(pending_bk, 'pre_calculated_route'):
                        for b_path in pending_bk.pre_calculated_route.values():
                            if b_path:
                                primary_nodes_used.add(b_path[0])

                forbidden = []
                for node in primary_nodes_used:
                    forbidden.append(node)
                    if isinstance(node, float):
                        forbidden.append(int(node))
                    elif isinstance(node, int):
                        forbidden.append(float(f"{node}.1"))
                    elif isinstance(node, str) and node.replace('.', '').isdigit():
                        try:
                             val = float(node)
                             forbidden.append(val)
                             forbidden.append(int(val))
                        except: pass

                env.set_forbidden_nodes(forbidden)

                agent.install_substrate_network(graph_for_env)
                agent.install_SFC(mini_sfc)
                
                try:
                    success = agent.start_algorithm(env)
                except Exception as e:
                    print(f"[BackupManager] Exception no Agent para {sfc_id}: {e}")
                    success = False
                
                if success:
                    route_info_backup = agent.get_route_info()
                    if not route_info_backup:
                         break

                    mini_sfc.pre_calculated_route = route_info_backup
                    pending_sfcs_this_cycle.append(mini_sfc)
                    backups_mount.append([mini_sfc])
                    
                    # --- CRÍTICO: Atualiza o Shadow State ---
                    # Deduzimos os recursos do simulation_graph para que a próxima
                    # iteração saiba que este nó está ocupado.
                    self._apply_virtual_reservation(simulation_graph, mini_sfc, route_info_backup)
                    
                else:
                    break
        
        return backups_mount

    def _apply_virtual_reservation(self, graph: Any, sfc: SFC, route_info: Dict) -> None:
        """
        Aplica a redução de recursos no grafo de simulação (Shadow State).
        Isso mimetiza a alocação física de forma simplificada.
        """
        for vnf_name, path in route_info.items():
            if vnf_name in ['src', 'dst'] or "virt" in vnf_name:
                continue
            
            if not path: continue
            
            # 1. Consumo de Nó
            node_id = path[0]
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                vnf_obj = sfc.get_vnf_by_id(vnf_name)
                
                # Assume-se sem reuso para ser conservador no backup (Worst Case)
                if vnf_obj:
                    cpu_req = vnf_obj.get_cpu_request()
                    cache_req = vnf_obj.get_cache_request()
                    
                    node['cpu_used'] = node.get('cpu_used', 0) + cpu_req
                    node['cache_used'] = node.get('cache_used', 0) + cache_req

            # 2. Consumo de Banda
            if len(path) > 1:
                # Recupera requisito de banda
                vnf_obj = sfc.get_vnf_by_id(vnf_name)
                bw_req = vnf_obj.get_outcome_interface_bandwidth() if vnf_obj else 0
                
                for u, v in zip(path[:-1], path[1:]):
                    if graph.has_edge(u, v):
                        edge = graph.edges[u, v]
                        edge['bandwidth_used'] = edge.get('bandwidth_used', 0) + bw_req

    def _enrich_graph_with_mobility(self, graph, network, mini_sfc):
        """Helper para adicionar nós móveis ao grafo copiado para o RL."""
        mobile_node_id = getattr(mini_sfc, 'mobile_node', None)
        closer_router_id = getattr(mini_sfc, 'closer_router', None)

        if mobile_node_id:
            if mobile_node_id in network.md_graph and mobile_node_id not in graph:
                md_data = network.md_graph.nodes[mobile_node_id]
                graph.add_node(mobile_node_id, **md_data)
            
            if closer_router_id and closer_router_id in graph:
                router_data = graph.nodes[closer_router_id]
                w_cap = router_data.get('w_channel_capacity', 0.0)
                w_used = router_data.get('w_channel_used', 0.0)
                wireless_free = max(0.0, w_cap - w_used)
                
                graph.add_edge(
                    mobile_node_id, 
                    closer_router_id, 
                    bandwidth_capacity=wireless_free, 
                    bandwidth_used=0.00, 
                    latency=1, 
                    services_in_transit={}
                )

    def greedy_strategy(self, network: Net2, sfc_id_duration: Dict) -> List[List[Any]]:
        """Estratégia gulosa aleatória para criação de backups."""
        backups_mount = []
        sfcs_id = list(sfc_id_duration.keys())
        random.shuffle(sfcs_id)
        
        if not sfcs_id: return []

        for sfc_id in sfcs_id:
            # Validações básicas
            parts = sfc_id.split("_")
            if len(parts) > 2 and parts[2] == 'backup': continue
            if sfc_id not in network.sfc_dict: continue
            if sfc_id in self.sfcs_backups_instatiated: continue
            
            # Chance aleatória
            if random.random() < 0.6: continue

            name = f"{parts[0]}_{parts[1]}_backup_{parts[2]}_{parts[3]}"
            if name in self.backups_sfc_instantiated: continue

            # Tempo restante
            current_time = time.time()
            start_time = sfc_id_duration[sfc_id]['timer']
            total_duration = sfc_id_duration[sfc_id]['duration']
            elapsed = current_time - start_time
            remaining_duration = max(10, total_duration - elapsed + 10)

            # Redução de recursos
            sfc = network.get_sfc_by_id(sfc_id)
            vnf_info = sfc.vnfs_dict
            original_latency_req = getattr(sfc, 'latency_request', 10)
            reduction_factor = self.standard_reduction_factor 

            new_vnfs_dict = []
            for info in vnf_info:
                new_info = info.copy()
                for aspect, value in info.items():
                    if aspect in ['CPU', 'cache', 'in_bw', 'out_bw']:
                        new_info[aspect] = value * reduction_factor
                    if aspect == 'name':
                        new_info[aspect] = value + "_b"
                new_vnfs_dict.append(new_info)

            player_dict = {
                'name': name,
                'vnf_list': new_vnfs_dict,
                'bandwidth': sfc.input_throughput,
                'src_node': sfc.src.substrate_node,
                'dst_node': sfc.dst.substrate_node,
                'duration': remaining_duration,
                'latency': original_latency_req
            }

            new_sfc = SFCGenerator(player_dict).generate()
            new_sfc.original_sfc_id = sfc_id
            new_sfc.is_backup = True
            backups_mount.append([new_sfc])

        return backups_mount
    
    def escolher_src_dst(self, dicionario: Dict, vnf_escolhida: str, 
                        latency_limit: int = 10) -> Tuple[Optional[str], Optional[str], int]:
        """Seleciona nós de origem e destino baseados na topologia da SFC."""
        chaves = list(dicionario.keys())
        if vnf_escolhida not in chaves:
            return None, None, None

        idx = chaves.index(vnf_escolhida)
        latency_dismiss = 0
        
        if idx == 0:  # Primeira VNF
            dst = chaves[idx]
            src = chaves[idx + 1]
            latency_dismiss = len(dicionario[src]) - 1
        elif idx == len(chaves) - 1:  # Última VNF
            dst = chaves[idx - 1]
            src = chaves[idx]
            latency_dismiss = len(dicionario[vnf_escolhida]) - 1
        else:  # Intermediária
            dst = chaves[idx - 1]
            src = chaves[idx + 1]
            latency_dismiss = (len(dicionario[src]) - 1) + (len(dicionario[vnf_escolhida]) - 1)

        src_node = dicionario[src][0]
        dst_node = dicionario[dst][0]
        
        latency_requirement = latency_limit - latency_dismiss
        return dst_node, src_node, latency_requirement

    def seletive_strategy(self, network: Net2, sfc_id_duration: Dict, threshold: float = 0) -> List[List[Any]]:
        """Estratégia seletiva baseada na confiabilidade dos nós."""
        backups_mount = []
        nodes_fail_p = network.nodes_reliability.copy()
        nodes_highest_p = {node: rel for node, rel in nodes_fail_p.items() if rel > threshold}
        
        if not nodes_highest_p:
            return []

        chosen_server = max(nodes_highest_p, key=nodes_highest_p.get)
        servers = [chosen_server]

        for server in servers:
            server_info = network.get_node_sfc_vnf_list(server)
            current_time = time.time()
            
            if not server_info: continue

            for info in server_info:
                sfc_id = info[0]
                vnf = info[1]
                vnf_id = vnf.id
                parts = sfc_id.split("_")

                if len(parts) > 2 and parts[2] == 'backup': continue
                if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration: continue

                name = f"{parts[0]}_{parts[1]}_backup_{vnf_id}_{parts[2]}_{parts[3]}"
                if name in self.backups_sfc_instantiated: continue

                sfc = network.get_sfc_by_id(sfc_id)
                vnf_info = sfc.vnfs_dict
                resources_info = next((i for i in vnf_info if i['name'] == vnf_id), None)
                
                sfc_rf = copy.deepcopy(network.sfc_route_info[sfc_id])
                if 'src' in sfc_rf: del sfc_rf['src']
                if 'dst' in sfc_rf: del sfc_rf['dst']

                if vnf_id not in sfc_rf: continue
                location = sfc_rf[vnf_id][0]
                
                # Recursos
                dst_in = resources_info['out_bw']
                cpu = resources_info['CPU']
                cache = resources_info['cache']

                original_latency = getattr(sfc, 'latency_request', 10)
                dst_node, src_node, latency_req = self.escolher_src_dst(sfc_rf, vnf_id, original_latency)
                
                if latency_req < 0 or dst_node is None: continue

                # Definição do Mini-SFC
                src_name = "src_virt"
                backup_vnf_name = vnf_id + "_b" 
                dst_name = "dst_virt"
                reduction_factor = self.standard_reduction_factor

                backup_sf_list = [
                    {
                        "type": 2, "name": src_name, "CPU": 0, "cache": 0, 
                        "in_bw": 0, "out_bw": 0, "latency": 0, "location": src_node
                    },
                    {
                        "type": 2, "name": backup_vnf_name, 
                        "CPU": cpu * reduction_factor, 
                        "cache": cache * reduction_factor, 
                        "in_bw": 0, "out_bw": dst_in * reduction_factor, 
                        "latency": 0, "original_loc": location, "original_sfc": sfc_id
                    },
                    {
                        "type": 2, "name": dst_name, "CPU": 0, "cache": 0, 
                        "in_bw": 0, "out_bw": 0, "latency": 0, "location": dst_node
                    }
                ]

                time_elapsed = current_time - sfc_id_duration[sfc_id]["timer"]
                duration = max(10, sfc_id_duration[sfc_id]["duration"] - time_elapsed + 10)

                new_sfc_dict = {
                    "name": name,
                    "vnf_list": backup_sf_list,
                    "bandwidth": sfc.input_throughput,
                    "src_node": src_node,
                    "dst_node": dst_node,
                    "duration": duration,
                    "latency": latency_req
                }

                new_sfc = SFCGenerator(new_sfc_dict).generate()
                new_sfc.original_sfc_id = sfc_id
                new_sfc.is_backup = True
                backups_mount.append([new_sfc])

        return backups_mount

    def create_contextual_mini_sfc(self, network: Net2, original_sfc: SFC, 
                                 vnf_to_replicate_id: str, primary_node_id: str) -> Optional[SFC]:
        """
        Cria uma Mini-SFC (3 saltos) para alocação via DRL.
        """
        sfc_id = original_sfc.id

        if hasattr(original_sfc, 'session_id'):
            session_id = original_sfc.session_id
        else:
            parts = sfc_id.split('_')
            session_id = parts[-1]

        target_vnf_info = next((info for info in original_sfc.vnfs_dict if info['name'] == vnf_to_replicate_id), None)
        
        factor = self.standard_reduction_factor
        
        if not target_vnf_info: return None

        # Contexto Físico
        route_info = network.sfc_route_info.get(sfc_id)
        if not route_info: return None

        current_vnf_obj = original_sfc.get_vnf_by_id(vnf_to_replicate_id)
        
        # Localização Anterior
        prev_vnf = original_sfc.get_previous_vnf(current_vnf_obj)
        if prev_vnf.id == 'src':
            prev_node = original_sfc.src.substrate_node
        else:
            prev_node = route_info[prev_vnf.id][0] 

        # Localização Próxima
        next_vnf = original_sfc.get_next_vnf(current_vnf_obj)
        if next_vnf.id == 'dst':
            next_node = original_sfc.dst.substrate_node
        else:
            next_node = route_info[next_vnf.id][0]

        # Constrói VNF List
        mini_sfc_vnfs = [
            {
                "type": 2, "name": "src_virt", 
                "CPU": 0, "cache": 0, "in_bw": 0, "out_bw": 0, "latency": 0, "location": prev_node
            },
            {
                "type": 2, "name": vnf_to_replicate_id + "_b", 
                "CPU": target_vnf_info['CPU'] * factor, 
                "cache": target_vnf_info['cache'] * factor, 
                "in_bw": 0, "out_bw": 0, "latency": 0, "original_sfc": sfc_id
            },
            {
                "type": 2, "name": "dst_virt",
                "CPU": 0, "cache": 0, "in_bw": 0, "out_bw": 0, "latency": 0, "location": next_node
            }
        ]
        
        current_time = time.time()
        start_time = getattr(original_sfc, 'arrival_time', current_time) 
        elapsed_time = current_time - start_time
        remaining_duration = max(10, original_sfc.duration - elapsed_time + 10)
        latency_constraint = getattr(original_sfc, 'latency_request', 10)
        
        backup_name = f"{sfc_id}_backup_{vnf_to_replicate_id}"
        
        mini_sfc_dict = {
            "name": backup_name,
            "vnf_list": mini_sfc_vnfs,
            "bandwidth": original_sfc.input_throughput,
            "src_node": prev_node,
            "dst_node": next_node,
            "duration": remaining_duration, 
            "latency": latency_constraint,
            "closer_router": getattr(original_sfc, 'closer_router', None),
            "mobile_node": getattr(original_sfc, 'dst_node', None)
        }

        mini_sfc = SFCGenerator(mini_sfc_dict).generate()

        mini_sfc.original_sfc_id = original_sfc.id
        mini_sfc.is_backup = True
        mini_sfc.target_vnf_id = vnf_to_replicate_id
        mini_sfc.session_id = session_id
        
        mini_sfc.src_virt = prev_node
        mini_sfc.dst_virt = next_node

        return mini_sfc

    def get_backups_instantiated_q(self) -> int:
        """Retorna a quantidade de VNFs de backup instanciadas (para logs)."""
        vnfs_backup_instantiate = 0
        backups = self.backups_sfc_instantiated
        
        if self.alg == 'ga':
            for backup_id, original_sfc in backups.items():
                sfc_backups = self.sfcs_backups_instatiated.get(original_sfc, [])
                vnfs_backup_instantiate += len(sfc_backups)
        else:
            # Estimativa simples se não for GA
            vnfs_backup_instantiate = len(list(backups.keys())) * 4

        return vnfs_backup_instantiate