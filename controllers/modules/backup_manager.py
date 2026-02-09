import copy
import time
import random
from typing import List, Dict, Optional, Tuple, Any, Set

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

    def create_backups(self, network: Net2, agent_ref=None) -> Tuple[List[Any], str]:
        """
        Ponto de entrada para criação de backups.
        Retorna a lista de SFCs a serem implantadas e o nome da estratégia usada.
        """
        if not self.backup_activated:
            return [], "none"

        # Cria dicionário auxiliar necessário para as estratégias
        sfc_id_duration = {}
        for sfc_id, sfc in network.sfc_dict.items():
            # Evita criar backup de um backup
            if "backup" in sfc_id:
                continue
                
            start_t = getattr(sfc, 'arrival_time', time.time())
            sfc_id_duration[sfc_id] = {
                "timer": start_t,
                "duration": getattr(sfc, 'duration', 100)
            }

        # Seleção de Estratégia
        if self.alg == 'SBRCMASKABLEPPO' and agent_ref is not None:
            # Estratégia RL (Deep Reinforcement Learning)
            backups_mount = self.rl_based_strategy(network, sfc_id_duration, agent_ref)
            return backups_mount, 'rl_based'
            
        elif self.alg in ['vegeta', 'ga']:
            # Estratégia Seletiva
            backups_mount = self.seletive_strategy(network, sfc_id_duration)
            return backups_mount, 'seletive'
            
        else:
            # Estratégia Greedy (Padrão)
            backups_mount = self.greedy_strategy(network, sfc_id_duration)
            return backups_mount, 'greedy'

    def _calc_virtual_reliability(self, network: Net2, sfc_id: str, 
                                pending_backups_sfcs: List[Any]) -> Tuple[float, List[Dict]]:
        """
        Calcula a confiabilidade total REAL da SFC, consultando a confiabilidade
        do nó físico onde o backup está (ou será) alocado.
        """
        if sfc_id not in network.sfc_route_info:
            return 0.0, []

        route_info = network.sfc_route_info[sfc_id]
        total_reliability = 1.0
        candidates = []

        # Itera sobre as VNFs da rota principal
        for vnf_id, path in route_info.items():
            if vnf_id in ['src', 'dst'] or not path:
                continue
            
            # 1. Confiabilidade do Nó Principal
            node_id = path[0]
            try:
                r_prim = network.get_node_reliability(node_id)
            except AttributeError:
                r_prim = 1.0

            # 2. Busca Confiabilidade do Nó de Backup (se existir)
            r_backup = 0.0
            has_backup = False
            
            # A) Verifica Backups JÁ Instanciados (Ciclos passados)
            if sfc_id in self.sfcs_backups_instatiated:
                for b in self.sfcs_backups_instatiated[sfc_id]:
                    if b['vnf_id'] == vnf_id:
                        bk_route = b.get('route_info', {})
                        for k, v in bk_route.items():
                            if k.endswith('_b') and v:
                                bk_node = v[0]
                                r_backup = network.get_node_reliability(bk_node)
                                has_backup = True
                                break
                    if has_backup: break

            # B) Verifica Backups Pendentes (Criados neste loop)
            if not has_backup:
                for mini_sfc in pending_backups_sfcs:
                    target_backup_name = f"{vnf_id}_b"
                    
                    if hasattr(mini_sfc, 'pre_calculated_route') and mini_sfc.pre_calculated_route:
                        bk_route = mini_sfc.pre_calculated_route
                        if target_backup_name in bk_route and bk_route[target_backup_name]:
                            bk_node = bk_route[target_backup_name][0]
                            r_backup = network.get_node_reliability(bk_node)
                            has_backup = True
                            break

            # 3. Cálculo do Estágio (Fórmula Paralela)
            if has_backup:
                # 1 - (Prob. Falha Prim * Prob. Falha Backup)
                stage_r = 1.0 - ((1.0 - r_prim) * (1.0 - r_backup))
            else:
                stage_r = r_prim
                candidates.append({'vnf_id': vnf_id, 'node_rel': r_prim, 'node_id': node_id})

            total_reliability *= stage_r

        sorted_candidates = sorted(candidates, key=lambda x: x['node_rel'])
        return total_reliability, sorted_candidates
        
    def rl_based_strategy(self, network: Net2, sfc_id_duration: Dict, agent: Any) -> List[List[Any]]:
        """
        Estratégia baseada na Confiabilidade Total da SFC.
        Cria backups iterativamente até que a confiabilidade atinja a meta.
        """
        backups_mount = []
        target_reliability = 0.9
        
        sorted_sfcs = sorted(list(sfc_id_duration.keys()))

        for sfc_id in sorted_sfcs:
            if "backup" in sfc_id: continue 
            if sfc_id not in network.sfc_dict: continue
            
            sfc = network.get_sfc_by_id(sfc_id)
            pending_sfcs_this_cycle = []
            
            while True:
                current_r, candidates = self._calc_virtual_reliability(
                    network, sfc_id, pending_sfcs_this_cycle
                )
                
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

                # Setup ambiente RL
                graph_for_rl = copy.deepcopy(network.graph)
                self._enrich_graph_with_mobility(graph_for_rl, network, mini_sfc)

                # Identifica nós válidos
                valid_types = ['server', 'mobile_device']
                all_servers = [n for n, d in graph_for_rl.nodes(data=True) if d.get('type') in valid_types]
                
                if not all_servers:
                    break 

                env = SFC_AllocationEnv(
                    valid_nodes=all_servers,
                    list_graph=[graph_for_rl],
                    list_sfc=[mini_sfc],
                    is_training=False
                )
                
                # Proíbe o nó fraco original
                forbidden = [weak_node]
                if isinstance(weak_node, (int, float)):
                    # Tratamento para nós com IDs numéricos flutuantes (se houver)
                    forbidden.append(weak_node - 0.1 if weak_node % 1 == 0.1 else weak_node + 0.1)
                env.set_forbidden_nodes(forbidden)

                agent.install_SFC(mini_sfc)
                try:
                    agent.install_substrate_network(graph_for_rl)
                    success = agent.start_algorithm(env)
                except KeyError as e:
                    print(f"[BackupManager] Erro no RL para SFC {sfc_id}: {e}")
                    success = False
                
                if success:
                    route_info_backup = agent.get_route_info()
                    mini_sfc.pre_calculated_route = route_info_backup
                    pending_sfcs_this_cycle.append(mini_sfc)
                    backups_mount.append([mini_sfc])
                else:
                    break
        
        return backups_mount

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