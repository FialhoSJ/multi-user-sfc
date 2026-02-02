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

        # Retorno corrigido para evitar crash no create_backups
        return {
            'is_success': deployment_success,
            'route_info': route_info_export
        }

    def undeploy_sfc(self, sfc_list_id: str, substrate_network, take_out_backup=True) -> None:
        """Remove a SFC group based on the destination node ID (sfc_list_id)."""
        if sfc_list_id in self.sfcs_tracker:
            for sfc in self.sfcs_tracker[sfc_list_id]['sfc_list']:
                
                # --- [MODIFICAÇÃO] Limpeza de Backups ---
                # Verifica se existem backups registrados para esta SFC específica
                if take_out_backup and sfc.id in self.backup_manager.sfcs_backups_instatiated:
                    self.undeploy_sfc_backups(sfc.id, substrate_network)
                # ----------------------------------------

                substrate_network.undeploy_sfc(sfc.id)
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
                
                # --- CORREÇÃO: Busca de Rota (Placement) ---
                # Como o backup_manager apenas cria o objeto SFC, precisamos encontrar 
                # um servidor válido para hospedar a VNF de backup.
                
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
                
                # Lista de candidatos: todos os servidores (exceto router/mobile)
                candidates = [n for n, d in network.graph.nodes(data=True) if d.get('type') == 'server']
                random.shuffle(candidates) # Aleatoriza para balancear

                for server in candidates:
                    # Checa recursos (CPU/Cache)
                    node_data = network.graph.nodes[server]
                    if (node_data['cpu_used'] + backup_vnf['CPU'] <= node_data['cpu_capacity']) and \
                       (node_data['cache_used'] + backup_vnf['cache'] <= node_data['cache_capacity']):
                        
                        # Checa conectividade (Path: Src -> Server -> Dst)
                        try:
                            # src -> server
                            path1 = k_shortest_paths(network, sfc.src_substrate_node, server, k=1, weight='latency')[0]
                            # server -> dst
                            path2 = k_shortest_paths(network, server, sfc.dst_substrate_node, k=1, weight='latency')[0]
                            
                            # Se encontrou caminhos
                            if path1 and path2:
                                route_info[backup_vnf['name']] = path1 # Caminho até a VNF
                                # Precisamos estruturar como o deploy_sfc espera.
                                # Normalmente: 'vnf_name': [node, path_to_next...]
                                # Mas k_shortest retorna nós.
                                # Vamos simplificar assumindo que deploy_sfc re-calcula ou aceita o nó.
                                # O deploy_sfc espera {vnf_id: [node_allocated, ...path_to_next_vnf...]}
                                
                                # Ajuste para formato do deploy_sfc:
                                # VNF Backup: alocada em 'server'. Caminho até PRÓXIMA (dst) é path2.
                                route_info[backup_vnf['name']] = [server] + path2 
                                
                                # Src virtual: alocada em src_node. Caminho até Backup é path1.
                                # route_info['src'] = [sfc.src_substrate_node] + path1 # Opcional dependendo da imp.
                                
                                valid_placement_found = True
                                break
                        except Exception:
                            continue
                
                if not valid_placement_found:
                    if self.verbose:
                        print(f"Skipping backup {sfc.id}: No placement found.")
                    continue

                # --- FIM CORREÇÃO ROTA ---

                # Agora chamamos submit_solution com a rota calculada
                solution = {sfc.id: {'route_info': route_info}}
                results_dict = self.submit_solution([sfc], solution, network, is_backup=True)

                # Verifica se o deploy funcionou
                if results_dict['is_success']:
                    original_sfc = None
                    vnf_id = None

                    # Lógica para extrair ID original corrigida
                    if self.alg_name == 'ga':
                        original_sfc = sfc.vnfs_dict[1]['original_sfc']
                        vnfs_dict = sfc.vnfs_dict[1]
                        vnf_id = vnfs_dict['name']
                    else:
                        # CORREÇÃO PARA GREEDY/SBRC
                        # O nome vem como "sfc_unique_pX_Y_backup_VNFID_Z_W" ou similar
                        # backup_vnf['name'] contém o nome da VNF de backup (ex: "vnf1_b")
                        split = sfc.id.split("_")
                        # Reconstrói ID original (assumindo padrão sfc_tipo_pX_Y)
                        # Ex: sfc_unique_p6_0_backup_vnf1... -> sfc_unique_p6_0
                        original_sfc = f"{split[0]}_{split[1]}_{split[3]}_{split[4]}"
                        
                        # Extrai a VNF ID removendo sufixo '_b' se existir
                        raw_vnf_name = backup_vnf['name']
                        vnf_id = raw_vnf_name.removesuffix("_b") if hasattr(raw_vnf_name, 'removesuffix') else raw_vnf_name.replace("_b", "")

                    if original_sfc not in self.backup_manager.sfcs_backups_instatiated:
                        self.backup_manager.sfcs_backups_instatiated[original_sfc] = []

                    self.backup_manager.sfcs_backups_instatiated[original_sfc].append({
                        "sfc_backup_id": sfc.id,
                        "vnf_id": vnf_id, # Agora é o ID real, não None
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

    def check_backups(self, sfc_id, vnfs, substrate_network):
        if sfc_id in list(self.backup_manager.sfcs_backups_instatiated.keys()):
            if self.alg_name == 'ga':
                for vnf in vnfs:
                    vnf_id = vnf.id
                    backups = self.backup_manager.sfcs_backups_instatiated[sfc_id]
                    for backup in backups:
                        vnf_backup = backup['vnf_id'].removesuffix("_b")
                        if vnf_id == vnf_backup:
                            print("Tem backup daquela vnf caída, então ativa")
                            self.trigger_sfc_backup(sfc_id, vnf_id, substrate_network, self.crashed_servers[0])
                            self.backup_manager.backups_activated.append(backup['sfc_backup_id'])
                            return backup
            else:
                backup = self.backup_manager.sfcs_backups_instatiated[sfc_id][0]
                backup_id = backup['sfc_backup_id']
                duration = self.sfc_id_duration[sfc_id]['duration']
                timer = self.sfc_id_duration[sfc_id]['timer']
                
                self.sfc_id_duration[backup_id] = {"duration": duration, "timer": timer}
                # self.sfc_list.append(backup_id) # Atenção: self.sfc_list não está definido no init, mantendo original
                # self.undeploy_sfc(sfc_id,substrate_network)
                return backup
        return False

    def trigger_sfc_backup(self, sfc_id, vnf_id, substrate_network, node_id):
        if sfc_id in list(self.sfs_backup.keys()):
            substrate_network.reset_vnf_cpu_request(node_id, sfc_id, vnf_id)

    def undeploy_sfc_backups(self, sfc_id, substrate_network):
        """Retira os backups da SFC, inclusive os ativos."""
        # Proteção: só tenta remover se a chave existir
        if sfc_id in self.backup_manager.sfcs_backups_instatiated:
            backups_removed = self.backup_manager.sfcs_backups_instatiated.pop(sfc_id)
            
            for backup in backups_removed:
                backup_id = backup["sfc_backup_id"]
                
                # Limpa do registro reverso (backup -> original)
                if backup_id in self.backup_manager.backups_sfc_instantiated:
                    del self.backup_manager.backups_sfc_instantiated[backup_id]
                
                # Remove da lista de ativados se estiver lá
                if backup_id in self.backup_manager.backups_activated:
                    self.backup_manager.backups_activated.remove(backup_id)
                
                # Remove fisicamente da rede
                try:
                    substrate_network.undeploy_sfc(backup_id)
                except ValueError:
                    # Caso já tenha sido removido por outro processo
                    pass

    def remove_backup_by_id(self, backup_id, substrate_network):
        """Remove todos os backups (menos os ativos)."""
        if backup_id in self.backup_manager.backups_sfc_instantiated:
            original_sfc = self.backup_manager.backups_sfc_instantiated[backup_id]

            if original_sfc in self.backup_manager.sfcs_backups_instatiated:
                backups = self.backup_manager.sfcs_backups_instatiated[original_sfc]
                self.backup_manager.sfcs_backups_instatiated[original_sfc] = [
                    b for b in backups if b["sfc_backup_id"] != backup_id
                ]

                if len(self.backup_manager.sfcs_backups_instatiated[original_sfc]) == 0:
                    del self.backup_manager.sfcs_backups_instatiated[original_sfc]
                del self.backup_manager.backups_sfc_instantiated[backup_id]

                # Nota: self.sfc_list não inicializado no init, mantido do original
                if hasattr(self, 'sfc_list') and backup_id in self.sfc_list:
                    self.sfc_list.remove(backup_id)
                    del self.sfc_id_duration[backup_id]
                
                substrate_network.undeploy_sfc(backup_id)
            else:
                try:
                    del self.backup_manager.backups_sfc_instantiated[backup_id]
                    substrate_network.undeploy_sfc(backup_id)
                except:
                    pass

    # ==========================================
    # Risk Assessment & Recovery Methods
    # ==========================================

    def set_risk_sfcs(self, servers, network):
        if len(servers) == 0:
            return []

        backups_mount = []
        sfc_id_duration = copy.deepcopy(self.sfc_id_duration)

        for server in servers:
            server_info = network.get_node_sfc_vnf_list(server)
            current_time = time.time()
            
            if not server_info:
                continue

            for info in server_info:
                # info = [sfc_id, vnf_object]
                sfc_id = info[0]
                vnf = info[1]
                vnf_id = vnf.id

                if sfc_id.split("_")[2] == 'backup':
                    continue

                if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration:
                    continue

                # Check if backup already exists logic (simplified in original)
                split = sfc_id.split("_")
                name_check = f"{split[0]}_{split[1]}_backup_{split[2]}_{split[3]}"
                try:
                    network.get_sfc_by_id(name_check)
                    continue
                except:
                    pass

                # Gather SFC Info
                sfc = network.get_sfc_by_id(sfc_id)
                vnf_info = sfc.vnfs_dict
                resources_info = [x for x in vnf_info if x['name'] == vnf_id][0]
                sfc_rf = network.sfc_route_info[sfc_id]

                src = None
                location = None
                dst = None
                
                src_out = resources_info['in_bw']
                dst_in = resources_info['out_bw']
                cpu = resources_info['CPU']
                cache = resources_info['cache']

                i = 0
                latency_dismiss = 0
                
                # Determine topology details
                for key, value in sfc_rf.items():
                    if i == 1:
                        src = value[0]
                        if src == 0:  # SF IA
                            src = value[-1]
                        else:
                            src = value[0]
                            latency_dismiss += len(value) - 1
                            print() # Mantido do original
                        break
                    if key == vnf_id:
                        location = value[0]
                        dst = value[-1]
                        latency_dismiss += len(value) - 1
                        i += 1

                # Construct Backup SFC
                new_sfc_list = []
                backup_sf_list = []
                
                name = f"{split[0]}_{split[1]}_backup_{vnf_id}_{split[2]}_{split[3]}"
                src_name = "src_" + name
                dst_name = "dst_" + name

                backup_sf_list.append({
                    "type": 2, "name": src_name, "CPU": 0, "cache": 0, 
                    "in_bw": 0, "out_bw": src_out, "latency": 0, "location": src
                })
                backup_sf_list.append({
                    "type": 2, "name": vnf_id, "CPU": cpu, "cache": cache, 
                    "in_bw": src_out, "out_bw": dst_in, "latency": 0, 
                    "restriction": location, "original_sfc": sfc_id
                })
                backup_sf_list.append({
                    "type": 2, "name": dst_name, "CPU": 0, "cache": 0, 
                    "in_bw": dst_in, "out_bw": 0, "latency": 0, "location": dst
                })

                duration = sfc_id_duration[sfc_id]["duration"] - (current_time - sfc_id_duration[sfc_id]["timer"])
                
                new_sfc_dict = {
                    "name": name,
                    "vnf_list": backup_sf_list,
                    "bandwidth": sfc.input_throughput,
                    "src_node": src,
                    "dst_node": dst,
                    "duration": duration + 20,
                    "latency": sfc.latency_request - self.calculate_latency(sfc_rf) + latency_dismiss
                }

                new_sfc = SFCGenerator(new_sfc_dict).generate()
                new_sfc_list.append(new_sfc)
                backups_mount.append(new_sfc_list)

        return backups_mount

    # ==========================================
    # Risk Assessment & Recovery Methods
    # ==========================================

    def calculate_sfc_reliability(self, sfc_id, substrate_network):
        """
        Calcula a confiabilidade total da SFC considerando réplicas (Sistema Paralelo).
        R_stage = 1 - ( (1-R_prim) * (1-R_rep) )
        R_total = Produto(R_stage) para todos os estágios.
        """
        if sfc_id not in self.sfcs_routing_info:
            return 0.0, None, None

        sfc = substrate_network.get_sfc_by_id(sfc_id)
        route_info = self.sfcs_routing_info[sfc_id]
        
        total_reliability = 1.0
        weakest_vnf_id = None
        min_stage_reliability = 1.1 # Sentinela
        weakest_node_primary = None

        # Itera sobre VNFs (Estágio por Estágio)
        current_vnf = sfc.get_src_vnf() # Começa da Origem
        
        # Percorre a cadeia
        while current_vnf:
            if current_vnf.id == 'src' or current_vnf.id == 'dst':
                # Pula src/dst virtuais para cálculo de confiabilidade (assumidos perfeitos ou externos)
                current_vnf = sfc.get_next_vnf(current_vnf)
                continue

            # 1. Identifica Nó Primário
            path = route_info.get(current_vnf.id)
            if not path:
                return 0.0, None, None # Cadeia quebrada
            
            primary_node = path[0]
            r_prim = substrate_network.get_node_reliability(primary_node)
            
            # 2. Verifica se há Réplica
            r_rep = 0.0
            has_replica = False
            
            if sfc_id in self.backup_manager.sfcs_backups_instatiated:
                backups = self.backup_manager.sfcs_backups_instatiated[sfc_id]
                for backup in backups:
                    # Verifica se este backup corresponde à VNF atual
                    if backup['vnf_id'] == current_vnf.id:
                        # Obtém Nó da Réplica da informação de rota do backup
                        rep_route = backup['route_info']
                        # A SFC de backup tem estrutura: src -> VNF_b -> dst
                        # Precisamos encontrar o nó da VNF_b
                        for k, v in rep_route.items():
                            if k.endswith('_b') and v:
                                replica_node = v[0]
                                r_rep = substrate_network.get_node_reliability(replica_node)
                                has_replica = True
                                break
                        break

            # 3. Calcula Confiabilidade do Estágio
            if has_replica:
                # Matemática de Sistema Paralelo: P(Pelo menos um funcionando)
                p_fail_prim = 1.0 - r_prim
                p_fail_rep = 1.0 - r_rep
                stage_reliability = 1.0 - (p_fail_prim * p_fail_rep)
            else:
                stage_reliability = r_prim

            # 4. Agrega
            total_reliability *= stage_reliability
            
            # 5. Rastreia Elo Mais Fraco (Alvo de Otimização)
            # Priorizamos replicar estágios que ainda não têm réplicas ou ainda são fracos
            if stage_reliability < min_stage_reliability:
                min_stage_reliability = stage_reliability
                weakest_vnf_id = current_vnf.id
                weakest_node_primary = primary_node
            
            current_vnf = sfc.get_next_vnf(current_vnf)
            
        return total_reliability, weakest_vnf_id, weakest_node_primary

    def attempt_recovery_by_replica(self, sfc_id, crashed_node_id, substrate_network) -> bool:
        """
        Tenta recuperar uma SFC afetada por falha trocando a VNF falha por sua réplica.
        Ativa a banda (que estava em standby/zero) no momento da recuperação.
        Retorna True se recuperado com sucesso, False se precisar de redeploy total.
        """
        # 1. Verifica se a SFC possui backups registrados no sistema
        if sfc_id not in self.backup_manager.sfcs_backups_instatiated:
            return False

        # 2. Identifica qual VNF específica estava no nó que caiu
        route_info = self.sfcs_routing_info.get(sfc_id)
        if not route_info:
            return False

        affected_vnf_id = None
        for vnf_id, path in route_info.items():
            if vnf_id in ['src', 'dst']: continue
            if not path: continue

            # O primeiro elemento da lista é o nó onde a VNF está hospedada
            if path[0] == crashed_node_id:
                affected_vnf_id = vnf_id
                break

        if not affected_vnf_id:
            return False  # Não encontrou VNF da SFC neste nó

        # 3. Busca se existe uma réplica ESPECÍFICA para essa VNF
        backups_list = self.backup_manager.sfcs_backups_instatiated[sfc_id]
        target_backup = None

        for backup_entry in backups_list:
            if backup_entry['vnf_id'] == affected_vnf_id:
                target_backup = backup_entry
                break

        if not target_backup:
            return False  # Existe backup para a SFC, mas não para a VNF que caiu

        # 4. Verifica se o nó da réplica está VIVO
        backup_route = target_backup['route_info']
        backup_node_id = None

        # Encontra o nó da VNF de backup (ignorando src/dst da mini-cadeia)
        # Procura por chaves que não sejam src/dst e que terminem em '_b' (padrão de nomeação de backup)
        # ou usa a chave vnf_id do target_backup se for consistente
        backup_vnf_key_target = affected_vnf_id + "_b"

        for b_vnf_key, b_path in backup_route.items():
            if b_vnf_key == backup_vnf_key_target and b_path:
                backup_node_id = b_path[0]
                break
        
        # Fallback se não achar pelo nome exato, pega a primeira VNF intermediária
        if backup_node_id is None:
             for b_vnf_key, b_path in backup_route.items():
                if b_vnf_key not in ['src', 'dst'] and "src" not in b_vnf_key and "dst" not in b_vnf_key and b_path:
                    backup_node_id = b_path[0]
                    backup_vnf_key_target = b_vnf_key # Atualiza a chave encontrada
                    break

        if backup_node_id is None:
            return False

        # Verifica no grafo se o nó da réplica está ativo
        backup_node_obj = substrate_network.graph.nodes[backup_node_id]
        if not backup_node_obj.get('is_active', True):
            if self.verbose:
                print(f"Recuperação falhou: Réplica para SFC {sfc_id} também está inativa no nó {backup_node_id}.")
            return False

        # --- [NOVO] 4.1. Ativação de Banda (Cold Standby) ---
        # Recupera a banda necessária da SFC Original
        sfc_orig = substrate_network.get_sfc_by_id(sfc_id)
        original_vnf_bw = 0
        
        # Procura nos metadados da SFC original quanto essa VNF consumia
        for vnf_dict in sfc_orig.vnfs_dict:
            if vnf_dict['name'] == affected_vnf_id:
                original_vnf_bw = vnf_dict['out_bw'] 
                break
        
        # Pega a rota física do backup (caminho de links)
        path_to_activate = backup_route.get(backup_vnf_key_target, [])
        
        # Tenta ativar a banda nos links físicos
        # Se os links estiverem congestionados, isso retornará False
        activation_success = substrate_network.activate_backup_path_bandwidth(
            path_to_activate, 
            original_vnf_bw, 
            backup_vnf_key_target
        )
        
        if not activation_success:
            if self.verbose:
                print(f"Recuperação falhou: Banda insuficiente para ativar réplica {backup_vnf_key_target} no caminho {path_to_activate}.")
            return False 

        # 5. COSTURA DA ROTA (Stitching)
        if self.verbose:
            print(f">>> [RECOVERY] Costurando rota da SFC {sfc_id}: VNF {affected_vnf_id} movida de {crashed_node_id} para {backup_node_id}")

        # Atualiza o routing_info LOCAL do Manager
        self.sfcs_routing_info[sfc_id][affected_vnf_id][0] = backup_node_id
        
        # --- [CORREÇÃO IMPORTANTE] ---
        # Atualiza também o routing_info da REDE (Topology)
        # Sem isso, o cálculo de latência continuará achando que passa pelo nó morto
        if sfc_id in substrate_network.sfc_route_info:
            substrate_network.sfc_route_info[sfc_id][affected_vnf_id][0] = backup_node_id
        # -----------------------------

        # Registra o backup como ativado
        if target_backup['sfc_backup_id'] not in self.backup_manager.backups_activated:
            self.backup_manager.backups_activated.append(target_backup['sfc_backup_id'])

        return True

    # ==========================================
    # Algorithm Specifics
    # ==========================================

    def musfico_method(self, sfc, substrate_network):
        """
        Calculates the latency and obtains the route information for the musfico algorithm.
        """
        route_info = copy.deepcopy(self.sfcs_routing_info[sfc.id])
        
        # Gets dst vnf and previous vnf
        dst_vnf = sfc.get_dst_vnf()
        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)
        prev_vnf_node = route_info[previous_vnf.id][0]
        
        # Applies k shortest to link the last vnf with the previous one
        shortest_path = k_shortest_paths(substrate_network, prev_vnf_node, dst_substrate_node, k=1, weight='latency')
        
        # Get the shortest among the k shortest paths
        route_info[previous_vnf.id] = shortest_path[0]
        latency = 0
        
        for vnf_id in route_info.keys():
            if vnf_id == 'src':
                continue
            path = route_info[vnf_id]
            for i in range(len(path) - 1):
                edge_latency = substrate_network.get_link_latency(path[i], path[i + 1])
                latency += edge_latency

        if latency > sfc.get_latency_request() or latency < 0:
            route_info = False
            latency = None

        return latency, route_info

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

    def set_sfc_reuse(self, sfc, route_info, shareable_sfs):
        return 1
        # Code unreachable based on original logic, but kept for preservation
        conta = 0
        if route_info:
            if route_info != {}:
                if route_info != False:
                    cpu_saved = 0
                    cache_saved = 0
                    total_cpu_req = 0
                    total_cache_req = 0

                    for vnf, servers in route_info.items():
                        if vnf not in ['src', 'dst']:
                            server_used = servers[0]
                            shared_sfs_in_node = list(map(lambda sf: sf.id, shareable_sfs[server_used]))
                            cpu_req = sfc.vnfs[vnf].get_cpu_request()
                            cache_req = sfc.vnfs[vnf].get_cache_request()
                            if vnf in shared_sfs_in_node:
                                cpu_saved += cpu_req
                                cache_saved += cache_req

                            total_cpu_req += cpu_req
                            total_cache_req += cache_req
                    conta = (cpu_saved + cache_saved) / (total_cpu_req + total_cache_req)
                    self.sfc_reuse[sfc.id] = conta
        return conta

    def get_shareable_sfs(self, sfc):
        """
        Returns: List[Tuple]: A list of SFs candidates.
        """
        shareable_sfs_ar = []
        src_sf = sfc.get_src_vnf()
        dst_sf = sfc.get_dst_vnf()
        sf = src_sf
        while sf != dst_sf:
            for (node, candidate_sf_id, candidate_sf) in self.shareable_list:
                if candidate_sf_id == sf.id:
                    print("Chegando onde não devia")
                    if len(self.substrate_network.sf_linkeds[candidate_sf]) != self.max_connections:
                        print("shareable sf found.........")
                        shareable_sfs_ar.append((node, candidate_sf_id, candidate_sf))
            sf = sf.get_next_vnf()
        return shareable_sfs_ar
