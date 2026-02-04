import time
import copy
import random
from typing import Optional

from controllers.modules.backup_manager import BackupManager
from controllers.sfc_generator import SFCGenerator
from utils.k_shortest_paths import k_shortest_paths


class SFCManager:
    def __init__(self, args, backup_manager, alg):
        # Configuration & Managers
        self.backup_manager: Optional[BackupManager] = backup_manager
        self.alg = alg
        self.alg_name = alg.name
        self.args = args
        self.verbose = True

        # State Trackers
        self.sfcs_tracker = {}        # SFCs running in the simulation (grouped by dst)
        self.sfcs_routing_info = {}   # Route info per SFC ID
        self.sfc_id_duration = {}     # Duration tracking
        self.player_sfc_id_list = {}
        self.sfcs_deployed_history = []
        self.crashed_servers = []
        self.sfs_backup = {}
        self.risk_servers = []
        self.sfc_reuse = {}

        # Counters & Flags
        self.counter = 0              # How many SFCs are running
        self.last_sfc_release = False

    # ==========================================
    # Core Lifecycle Methods (Deploy/Undeploy)
    # ==========================================

    def submit_solution(self, sfc_list, solution, substrate_network, is_backup=False) -> dict:
        """
        Implanta a solução na rede e registra no tracker.
        Retorna um dicionário com status de sucesso e informações da rota.
        """
        deployment_success = True
        route_info_export = None

        for sfc in sfc_list:
            if sfc.id not in solution:
                deployment_success = False
                continue

            rf = solution[sfc.id]['route_info']
            if not rf:
                deployment_success = False
                continue

            try:
                # Tenta realizar o deploy físico na rede
                substrate_network.deploy_sfc(sfc, rf)
                # Mantém cópia profunda da rota
                self.sfcs_routing_info[sfc.id] = copy.deepcopy(rf)
                route_info_export = rf
            except Exception as e:
                if self.verbose:
                    print(f"Deploy failed for {sfc.id}: {e}")
                deployment_success = False
                continue

        # Se for backup, não registramos no tracker de sessões (sfcs_tracker),
        # pois backups são gerenciados separadamente.
        if not is_backup and deployment_success:
            group_id = sfc_list[0].dst_node
            duration = sfc_list[0].duration

            if group_id in self.sfcs_tracker:
                raise ValueError("SFC já instanciada")
            else:
                self.sfcs_tracker[group_id] = {
                    "sfc_list": sfc_list,
                    'solution': solution,
                    'duration': duration,
                    'timer': time.time()
                }

        if not is_backup and deployment_success:
            self.counter += 1

        return {
            'is_success': deployment_success,
            'route_info': route_info_export
        }

    def undeploy_sfc(self, sfc_list_id: str, substrate_network, take_out_backup=True) -> None:
        """Remove a SFC group based on the destination node ID (sfc_list_id)."""
        if sfc_list_id in self.sfcs_tracker:
            for sfc in self.sfcs_tracker[sfc_list_id]['sfc_list']:
                
                # [CORREÇÃO DE ROBUSTEZ]
                # Primeiro tentamos remover os backups. Se der erro na principal,
                # pelo menos não deixamos lixo de backup na rede.
                if take_out_backup and sfc.id in self.backup_manager.sfcs_backups_instatiated:
                    try:
                        self.undeploy_sfc_backups(sfc.id, substrate_network)
                    except Exception as e:
                        print(f"❌ [ERRO CRÍTICO] Falha ao limpar backups de {sfc.id}: {e}")

                # Agora removemos a SFC principal
                try:
                    substrate_network.undeploy_sfc(sfc.id)
                except ValueError as ve:
                    # Se o erro for "SFC não encontrada", é o bug do zumbi. 
                    # Ignoramos para permitir que o loop continue para as próximas SFCs.
                    if "não encontrada" in str(ve) and self.verbose:
                        print(f"⚠️ [AVISO] Tentativa de remover SFC Fantasma {sfc.id} ignorada.")
                    else:
                        print(f"❌ Erro ao remover SFC {sfc.id}: {ve}")
                except Exception as e:
                    print(f"❌ Erro desconhecido ao remover SFC {sfc.id}: {e}")

            del self.sfcs_tracker[sfc_list_id]

    def deploy_success(self, sfc: object) -> None:
        if self.verbose:
            print("deploy succeed, sfc: ", sfc.id)

    def deploy_failed(self, sfc: object) -> None:
        if self.verbose:
            print(" deploy FAILED, sfc: ", sfc.id)

    # ==========================================
    # Backup Management Methods
    # ==========================================

    def create_backups(self, network):
        backups_mount = []
        sfc_id_duration = copy.deepcopy(self.sfc_id_duration)

        # 1. Gera as SFCs de backup (mas elas vêm sem rota definida ainda)
        if self.alg_name in ['vegeta', 'ga']:
            self.clean_backups(network)
            backups_mount = self.backup_manager.seletive_strategy(network, sfc_id_duration)
        else:
            backups_mount = self.backup_manager.greedy_strategy(network, sfc_id_duration)

        if backups_mount:
            for backup_list in backups_mount:
                sfc = backup_list[0] # SFC de backup gerada
                
                # Identifica a VNF de backup (aquela que não é src nem dst)
                backup_vnf = None
                for vnf in sfc.vnfs_dict:
                    if vnf['name'] not in ['src', 'dst'] and not vnf['name'].startswith('src_') and not vnf['name'].startswith('dst_'):
                        backup_vnf = vnf
                        break
                
                if not backup_vnf:
                    continue

                # Tenta encontrar um servidor válido (simples greedy/shortest path)
                valid_placement_found = False
                route_info = {}
                
                candidates = [n for n, d in network.graph.nodes(data=True) if d.get('type') == 'server']
                random.shuffle(candidates) 

                for server in candidates:
                    node_data = network.graph.nodes[server]
                    if (node_data['cpu_used'] + backup_vnf['CPU'] <= node_data['cpu_capacity']) and \
                       (node_data['cache_used'] + backup_vnf['cache'] <= node_data['cache_capacity']):
                        
                        try:
                            # src -> server
                            path1 = k_shortest_paths(network, sfc.src_substrate_node, server, k=1, weight='latency')[0]
                            # server -> dst
                            path2 = k_shortest_paths(network, server, sfc.dst_substrate_node, k=1, weight='latency')[0]
                            
                            if path1 and path2:
                                route_info[backup_vnf['name']] = [server] + path2 
                                valid_placement_found = True
                                break
                        except Exception:
                            continue
                
                if not valid_placement_found:
                    continue

                # Agora chamamos submit_solution com a rota calculada
                solution = {sfc.id: {'route_info': route_info}}
                results_dict = self.submit_solution([sfc], solution, network, is_backup=True)

                if results_dict['is_success']:
                    original_sfc = None
                    vnf_id = None

                    if self.alg_name == 'ga':
                        original_sfc = sfc.vnfs_dict[1]['original_sfc']
                        vnfs_dict = sfc.vnfs_dict[1]
                        vnf_id = vnfs_dict['name']
                    else:
                        split = sfc.id.split("_")
                        # Reconstrói ID original: sfc_unique_p6_0
                        original_sfc = f"{split[0]}_{split[1]}_{split[4]}_{split[5]}"
                        
                        raw_vnf_name = backup_vnf['name']
                        vnf_id = raw_vnf_name.removesuffix("_b") if hasattr(raw_vnf_name, 'removesuffix') else raw_vnf_name.replace("_b", "")

                    if original_sfc not in self.backup_manager.sfcs_backups_instatiated:
                        self.backup_manager.sfcs_backups_instatiated[original_sfc] = []

                    self.backup_manager.sfcs_backups_instatiated[original_sfc].append({
                        "sfc_backup_id": sfc.id,
                        "vnf_id": vnf_id,
                        "route_info": results_dict["route_info"]
                    })
                    self.backup_manager.backups_sfc_instantiated[sfc.id] = original_sfc

    def clean_backups(self, network):
        for backup in list(self.backup_manager.backups_sfc_instantiated.keys()):
            if backup in self.backup_manager.backups_activated:
                continue
            if random.random() < 0.7:  # 70% chance to delete
                self.remove_backup_by_id(backup, network)
        network.update()

    def undeploy_sfc_backups(self, sfc_id, substrate_network):
        """Remove todos os backups associados a uma SFC da rede física e lógica."""
        if sfc_id in self.backup_manager.sfcs_backups_instatiated:
            # Pega a lista de backups (Mini-SFCs) associados
            backups_list = self.backup_manager.sfcs_backups_instatiated.pop(sfc_id)
            
            for backup_entry in backups_list:
                b_id = backup_entry["sfc_backup_id"]
                
                # 1. Remoção Física (O mais importante para liberar recursos)
                try:
                    substrate_network.undeploy_sfc(b_id)
                    if self.verbose:
                        print(f"[CLEANUP] Backup {b_id} removido da rede física.")
                except Exception as e:
                    pass # Já removido ou inexistente

                # 2. Remoção Lógica no BackupManager
                if b_id in self.backup_manager.backups_sfc_instantiated:
                    del self.backup_manager.backups_sfc_instantiated[b_id]
                
                if b_id in self.backup_manager.backups_activated:
                    self.backup_manager.backups_activated.remove(b_id)

    def remove_backup_by_id(self, backup_id, substrate_network):
        """Remove todos os backups de forma segura e agressiva."""
        
        # Tenta remover da rede física INCONDICIONALMENTE primeiro
        # Isso garante que não sobrem recursos zumbis
        try:
            substrate_network.undeploy_sfc(backup_id)
        except Exception:
            # Ignora erro se já não existia na rede, mas garante a tentativa
            pass

        # Agora limpa os registros lógicos (dicionários)
        if backup_id in self.backup_manager.backups_sfc_instantiated:
            original_sfc = self.backup_manager.backups_sfc_instantiated[backup_id]

            if original_sfc in self.backup_manager.sfcs_backups_instatiated:
                backups = self.backup_manager.sfcs_backups_instatiated[original_sfc]
                # Filtra a lista mantendo apenas os outros backups
                self.backup_manager.sfcs_backups_instatiated[original_sfc] = [
                    b for b in backups if b["sfc_backup_id"] != backup_id
                ]

                # Se a lista ficou vazia, remove a entrada da SFC original
                if len(self.backup_manager.sfcs_backups_instatiated[original_sfc]) == 0:
                    del self.backup_manager.sfcs_backups_instatiated[original_sfc]
            
            # Remove o mapeamento reverso
            del self.backup_manager.backups_sfc_instantiated[backup_id]

    # ==========================================
    # Risk Assessment & Recovery Methods
    # ==========================================

    def calculate_sfc_reliability(self, sfc_id, substrate_network):
        """
        Calcula a confiabilidade da SFC agrupando VNFs por nó físico.
        [MODIFICADO] Agora imprime a confiabilidade total com 3 casas decimais.
        """
        if sfc_id not in self.sfcs_routing_info:
            return 0.0, None, None

        sfc = substrate_network.get_sfc_by_id(sfc_id)
        route_info = self.sfcs_routing_info[sfc_id]
        
        weakest_vnf_id = None
        min_stage_reliability = 1.1 
        weakest_node_primary = None
        node_groups = {}
        
        # 1. Agrupamento (Mapeia VNF -> Nó Físico)
        current_vnf = sfc.get_src_vnf()
        while current_vnf:
            if current_vnf.id in ['src', 'dst']:
                current_vnf = sfc.get_next_vnf(current_vnf)
                continue

            path = route_info.get(current_vnf.id)
            if not path:
                return 0.0, None, None
            
            primary_node = path[0]
            if primary_node not in node_groups:
                node_groups[primary_node] = []
            node_groups[primary_node].append(current_vnf.id)
            current_vnf = sfc.get_next_vnf(current_vnf)

        # 2. Cálculo da Confiabilidade Total
        total_reliability = 1.0

        for node_id, vnfs_list in node_groups.items():
            r_prim = substrate_network.get_node_reliability(node_id)
            all_vnfs_have_backup = True
            prod_reliability_backups = 1.0

            for vnf_id in vnfs_list:
                has_replica = False
                r_rep = 0.0
                
                # Verifica se existe backup para esta VNF específica
                if sfc_id in self.backup_manager.sfcs_backups_instatiated:
                    backups = self.backup_manager.sfcs_backups_instatiated[sfc_id]
                    for backup in backups:
                        if backup['vnf_id'] == vnf_id:
                            rep_route = backup['route_info']
                            # Encontra onde o backup está alocado fisicamente
                            for k, v in rep_route.items():
                                if k.endswith('_b') and v:
                                    replica_node = v[0]
                                    r_rep = substrate_network.get_node_reliability(replica_node)
                                    has_replica = True
                                    break
                        if has_replica: break
                
                if not has_replica:
                    all_vnfs_have_backup = False
                else:
                    prod_reliability_backups *= r_rep

                # Cálculo do estágio individual para achar o elo mais fraco
                if has_replica:
                    # Confiabilidade paralela: 1 - (Prob Falha Prim * Prob Falha Backup)
                    stage_r = 1.0 - ((1.0 - r_prim) * (1.0 - r_rep))
                else:
                    stage_r = r_prim
                
                if stage_r < min_stage_reliability:
                    min_stage_reliability = stage_r
                    weakest_vnf_id = vnf_id
                    weakest_node_primary = node_id

            # Aplicação da Fórmula no Grupo Físico
            if all_vnfs_have_backup:
                group_reliability = r_prim + ((1.0 - r_prim) * prod_reliability_backups)
            else:
                group_reliability = r_prim

            total_reliability *= group_reliability

        # --- [NOVO] Print de Monitoramento ---
        if self.verbose:
            print(f"[RELIABILITY] SFC {sfc_id}: {total_reliability:.3f} (Elo mais fraco: {min_stage_reliability:.3f} em {weakest_node_primary})")
        # -------------------------------------

        return total_reliability, weakest_vnf_id, weakest_node_primary
    
    def reconstruct_and_redeploy(self, sfc_obj, crashed_node_id, old_route_info, substrate_network) -> bool:
        """
        Reconstrói a rota usando o backup de forma ATÔMICA e SEGURA.
        Inclui logs detalhados de falha.
        """
        # 1. Verifica Backups Disponíveis
        if sfc_obj.id not in self.backup_manager.sfcs_backups_instatiated:
            if self.verbose:
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Nenhum backup foi instanciado previamente.")
            return False

        # 2. Identifica VNF afetada
        affected_vnf_id = None
        for vnf_id, path in old_route_info.items():
            if vnf_id in ['src', 'dst'] or not path: continue
            if path[0] == crashed_node_id:
                affected_vnf_id = vnf_id
                break
        
        if not affected_vnf_id:
            if self.verbose:
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Não consegui mapear qual VNF estava no nó falho {crashed_node_id}.")
            return False

        # 3. Busca a Réplica Específica
        backups_list = self.backup_manager.sfcs_backups_instatiated[sfc_obj.id]
        target_backup = None
        for backup_entry in backups_list:
            b_vnf_clean = backup_entry['vnf_id'].replace("_b", "")
            if b_vnf_clean == affected_vnf_id:
                target_backup = backup_entry
                break
        
        if not target_backup:
            if self.verbose:
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Existe backup da SFC, mas NÃO para a VNF específica {affected_vnf_id}.")
            return False

        # 4. Prepara a nova rota (Stitching)
        new_route_info = copy.deepcopy(old_route_info)
        backup_route = target_backup['route_info']
        backup_sfc_id = target_backup['sfc_backup_id']
        
        # Encontra chaves de entrada e saída no backup
        key_ingress = 'src_virt'
        key_egress = None
        for k in backup_route.keys():
            if k.endswith('_b') or k == f"{affected_vnf_id}_b":
                key_egress = k
                break
        
        path_ingress = backup_route.get(key_ingress)
        path_egress = backup_route.get(key_egress)

        if not path_ingress or not path_egress:
            if self.verbose:
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Rota do backup incompleta/corrompida (Ingress/Egress missing).")
            return False

        # Verifica saúde do nó de backup
        backup_node = path_egress[0]
        backup_node_obj = substrate_network.graph.nodes[backup_node]
        if not backup_node_obj.get('is_active', True):
            if self.verbose:
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Azar! O nó de backup {backup_node} TAMBÉM está caído.")
            return False

        # =========================================================================
        # 5. EXECUÇÃO CRÍTICA: ATOMIC SWAP & COLD STANDBY ACTIVATION
        # =========================================================================
        
        affected_vnf_obj = sfc_obj.get_vnf_by_id(affected_vnf_id)
        bw_in = affected_vnf_obj.get_income_interface_bandwidth()
        bw_out = affected_vnf_obj.get_outcome_interface_bandwidth()

        # Tenta reservar banda no path_ingress
        if not substrate_network.activate_backup_path_bandwidth(path_ingress, bw_in, sfc_obj.id):
            if self.verbose: 
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Congestionamento! Falha ao ativar banda de ENTRADA no caminho {path_ingress}.")
            return False

        # Tenta reservar banda no path_egress
        if not substrate_network.activate_backup_path_bandwidth(path_egress, bw_out, sfc_obj.id):
            if self.verbose: 
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Congestionamento! Falha ao ativar banda de SAÍDA no caminho {path_egress}.")
            return False

        # B) TRANSFERÊNCIA DE TITULARIDADE (Swap de CPU/RAM)
        backup_vnf_name = key_egress # ex: "vnf1_b"
        session_backup = backup_sfc_id.split("_")[-1]
        
        backup_service_key = (backup_vnf_name, session_backup)
        original_vnf_name = affected_vnf_id
        session_original = sfc_obj.id.split("_")[-1]
        original_service_key = (original_vnf_name, session_original)
        
        if backup_service_key in backup_node_obj['services']:
            res_data = backup_node_obj['services'].pop(backup_service_key)
            backup_node_obj['services'][original_service_key] = res_data
            
            if backup_sfc_id in backup_node_obj['sfcs_list']:
                backup_node_obj['sfcs_list'].remove(backup_sfc_id)
            if sfc_obj.id not in backup_node_obj['sfcs_list']:
                backup_node_obj['sfcs_list'].append(sfc_obj.id)
        else:
            if self.verbose: 
                actual_services = substrate_network.graph.nodes[backup_node].get('services', {})
                print(f"DEBUG: Serviços no nó {backup_node}: {list(actual_services.keys())}")
                print(f"DEBUG: Chave buscada: {backup_service_key}")
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Erro de Consistência! Serviço de backup {backup_service_key} não encontrado no nó {backup_node}.")
            return False

        # C) Atualiza Rotas
        prev_vnf = sfc_obj.get_previous_vnf(affected_vnf_obj)
        if prev_vnf.id == 'src':
            new_route_info['src'] = path_ingress
        else:
            new_route_info[prev_vnf.id] = path_ingress
            
        new_route_info[affected_vnf_id] = path_egress

        # D) Consolidação
        self.sfcs_routing_info[sfc_obj.id] = new_route_info

        if hasattr(substrate_network, 'sfc_route_info'):
            substrate_network.sfc_route_info[sfc_obj.id] = copy.deepcopy(new_route_info)

        # ==============================================================================
        # [CORREÇÃO CRÍTICA] Ressuscitar a SFC no dicionário da Rede
        # ==============================================================================
        if sfc_obj.id not in substrate_network.sfc_dict:
            substrate_network.sfc_dict[sfc_obj.id] = sfc_obj
            if self.verbose:
                print(f"⚠️ [FIX] SFC {sfc_obj.id} reinserida no sfc_dict para evitar zumbis.")
        # ==============================================================================
        
        # IMPORTANTE: Remover o backup da rede mas EVITAR erro se o serviço já foi movido
        try:
            self.remove_backup_by_id(backup_sfc_id, substrate_network)
        except Exception as e:
            print(f"Aviso: Limpeza do backup {backup_sfc_id} gerou erro não fatal: {e}")
        
        if backup_sfc_id not in self.backup_manager.backups_activated:
            self.backup_manager.backups_activated.append(backup_sfc_id)

        if self.verbose:
            print(f"✅ [STITCH-SUCCESS] {sfc_obj.id}: Recuperada! VNF {affected_vnf_id} movida de {crashed_node_id} -> {backup_node}")
            
        return True



    # ==========================================
    # Helpers & Statistics
    # ==========================================

    def get_sfc_List(self, sfc_id, sb_net):
        sfc = sb_net.get_sfc_by_id(sfc_id)
        group_id = sfc.dst_node
        return self.sfcs_tracker[group_id]['sfc_list']

    def is_Backup(self, sfc_id):
        return True if sfc_id.split("_")[2] == 'backup' else False

    def calculate_latency(self, route_info):
        total_latency = sum(
            len(path) - 1 for key, path in route_info.items() if path and key not in ["src", "dst"]
        )
        return total_latency

    def get_running_players_sessions(self):
        running_sfcs = self.sfc_id_duration
        players = {}
        sessions = set()

        for key in list(running_sfcs.keys()):
            player = key.split('_')[-2][-1]
            session = key.split('_')[-1]
            sessions.add(session)
            if player not in players:
                players[player] = set()
            players[player].add(session)

        num_players = sum(len(sessions) for sessions in players.values())
        num_sessions = len(sessions)
        return len(running_sfcs.keys()), num_players, num_sessions

    def get_shareable_sfs(self, sfc):
        shareable_sfs_ar = []
        src_sf = sfc.get_src_vnf()
        dst_sf = sfc.get_dst_vnf()
        sf = src_sf
        while sf != dst_sf:
            for (node, candidate_sf_id, candidate_sf) in self.shareable_list:
                if candidate_sf_id == sf.id:
                    if len(self.substrate_network.sf_linkeds[candidate_sf]) != self.max_connections:
                        shareable_sfs_ar.append((node, candidate_sf_id, candidate_sf))
            sf = sf.get_next_vnf()
        return shareable_sfs_ar