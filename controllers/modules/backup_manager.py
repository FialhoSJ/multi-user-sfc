import copy
import time
import random
from controllers.sfc_generator import SFCGenerator
from core.net_v2 import Net2
from core.sfc import SFC
from algorithms.environments.env_sbrc import SFC_AllocationEnv

class BackupManager:
    def __init__(self, args):
        self.sfcs_backups_instatiated = {}
        self.backups_sfc_instantiated = {}
        self.backups_activated = []
        
        target_algorithm = 'SBRCMASKABLEPPO'
        
        is_target_alg = (args.alg == target_algorithm)
        user_wants_backup = (args.backup == 'y') and (args.ava != '1.0')
        
        self.backup_activated = user_wants_backup and is_target_alg
        self.alg = args.alg
        
        if user_wants_backup and not is_target_alg:
            print(f"[BackupManager] INFO: Backup proativo desativado. '{args.alg}' não suporta essa estratégia.")

        self.standard_reduction_factor = 1.0

    def _calc_virtual_reliability(self, network: Net2, sfc_id, pending_backups_sfcs):
        """
        Calcula a confiabilidade total REAL da SFC, consultando a confiabilidade
        do nó físico onde o backup está (ou será) alocado.
        
        pending_backups_sfcs: Lista de objetos SFC (Mini-SFCs) criados neste ciclo,
                              que já possuem 'pre_calculated_route' definida pelo agente.
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
                    # Verifica se o backup protege ESTA vnf
                    if b['vnf_id'] == vnf_id:
                        # Extrai o nó da rota do backup
                        bk_route = b.get('route_info', {})
                        for k, v in bk_route.items():
                            # Procura a VNF de backup (sufixo _b)
                            if k.endswith('_b') and v:
                                bk_node = v[0]
                                r_backup = network.get_node_reliability(bk_node)
                                has_backup = True
                                break
                    if has_backup: break

            # B) Verifica Backups Pendentes (Criados neste loop while)
            if not has_backup:
                for mini_sfc in pending_backups_sfcs:
                    # O nome da VNF de backup no mini_sfc deve ser "vnf_id + _b"
                    target_backup_name = f"{vnf_id}_b"
                    
                    # Verifica se a rota foi calculada pelo agente
                    if hasattr(mini_sfc, 'pre_calculated_route') and mini_sfc.pre_calculated_route:
                        bk_route = mini_sfc.pre_calculated_route
                        
                        # Verifica se essa Mini-SFC contém a VNF que estamos procurando
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
                # Se não tem backup, é candidato a receber um
                candidates.append({'vnf_id': vnf_id, 'node_rel': r_prim, 'node_id': node_id})

            total_reliability *= stage_r

        # Retorna candidatos ordenados pelo nó MENOS confiável (Prioridade)
        sorted_candidates = sorted(candidates, key=lambda x: x['node_rel'])
        return total_reliability, sorted_candidates
        
        
    def rl_based_strategy(self, network: Net2, sfc_id_duration, agent):
        """
        Estratégia baseada na Confiabilidade Total da SFC.
        Cria backups iterativamente até que a confiabilidade COMPOSTA (Real) atinja a meta.
        """
        backups_mount = []
        target_reliability = 0.9
        
        sorted_sfcs = sorted(list(sfc_id_duration.keys()))

        for sfc_id in sorted_sfcs:
            if "backup" in sfc_id: continue 
            if sfc_id not in network.sfc_dict: continue
            
            sfc = network.get_sfc_by_id(sfc_id)
            
            # Lista de objetos SFC (Mini-SFCs) criados nesta sessão para esta SFC
            pending_sfcs_this_cycle = []
            
            # Loop de Refinamento: Continua protegendo VNFs até bater a meta
            while True:
                # Calcula confiabilidade considerando o que já existe + o que acabamos de criar
                current_r, candidates = self._calc_virtual_reliability(
                    network, sfc_id, pending_sfcs_this_cycle
                )
                
                # Se já atingiu a meta (0.99), paramos de gastar recursos
                if current_r >= target_reliability:
                    break
                
                
                # Se não tem mais VNFs desprotegidas para melhorar, paramos
                if not candidates:
                    break

                # Pega o pior caso
                target_info = candidates[0] 
                target_vnf = target_info['vnf_id']
                weak_node = target_info['node_id']

                # Cria o Contexto do Backup (Mini-SFC)
                mini_sfc = self.create_contextual_mini_sfc(network, sfc, target_vnf, weak_node)
                if not mini_sfc: 
                    break 

                # --- Preparação do Ambiente RL ---
                graph_for_rl = copy.deepcopy(network.graph)
                
                mobile_node_id = getattr(mini_sfc, 'mobile_node', None)
                closer_router_id = getattr(mini_sfc, 'closer_router', None)

                if mobile_node_id:
                    if mobile_node_id in network.md_graph and mobile_node_id not in graph_for_rl:
                        md_data = network.md_graph.nodes[mobile_node_id]
                        graph_for_rl.add_node(mobile_node_id, **md_data)
                    
                    if closer_router_id and closer_router_id in graph_for_rl:
                        router_data = graph_for_rl.nodes[closer_router_id]
                        w_cap = router_data.get('w_channel_capacity', 0.0)
                        w_used = router_data.get('w_channel_used', 0.0)
                        wireless_free = max(0.0, w_cap - w_used)
                        
                        graph_for_rl.add_edge(
                            mobile_node_id, 
                            closer_router_id, 
                            bandwidth_capacity=wireless_free, 
                            bandwidth_used=0.00, 
                            latency=1, 
                            services_in_transit={}
                        )

                valid_types = ['server', 'mobile_device']
                
                all_servers = [
                    n for n, d in graph_for_rl.nodes(data=True) 
                    if d.get('type') in valid_types
                ]
                
                if not all_servers:
                    if self.alg == 'SBRCMASKABLEPPO': # Apenas loga se for debug relevante
                         pass 
                    break 

                env = SFC_AllocationEnv(
                    valid_nodes=all_servers,
                    list_graph=[graph_for_rl],
                    list_sfc=[mini_sfc],
                    is_training=False
                )
                
                # Proíbe o nó fraco original para forçar redundância real
                forbidden = [weak_node]
                if isinstance(weak_node, (int, float)):
                    forbidden.append(weak_node - 0.1 if weak_node % 1 == 0.1 else weak_node + 0.1)
                env.set_forbidden_nodes(forbidden)

                # Executa o Agente
                agent.install_SFC(mini_sfc)
                
                try:
                    debub = network.sfc_route_info[sfc_id]
                    agent.install_substrate_network(graph_for_rl)
                    success = agent.start_algorithm(env)
                except KeyError as e:
                    print(f"[BackupManager] Erro crítico no RL para SFC {sfc_id}: {e}. Pulando.")
                    success = False
                
                if success:
                    route_info_backup = agent.get_route_info()
                    debug = network.sfc_route_info.get(sfc.id)
                    # Anexa a rota calculada ao objeto Mini-SFC
                    mini_sfc.pre_calculated_route = route_info_backup
                    
                    
                    # Adiciona à lista local para o próximo cálculo de _calc_virtual_reliability
                    pending_sfcs_this_cycle.append(mini_sfc)
                    
                    # Adiciona à lista final de retorno
                    backups_mount.append([mini_sfc])
                else:
                    # Se falhou em alocar backup para este candidato, removemos ele da lista
                    # de candidatos no próximo loop implicitamente ou forçamos o break
                    # para evitar loop infinito tentando alocar o inalocável.
                    break
        
        if backups_mount:
            debug = 1
        return backups_mount

    def greedy_strategy(self, network, sfc_id_duration, threshold=0):
        backups_mount = []
        sfcs_id = list(sfc_id_duration.keys())
        
        random.shuffle(sfcs_id)
        
        if not sfcs_id:
            return []

        for sfc_id in sfcs_id:
            # Validações iniciais
            parts = sfc_id.split("_")
            if len(parts) > 2 and parts[2] == 'backup':
                continue

            if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration:
                continue

            if sfc_id in self.sfcs_backups_instatiated:
                continue

            if random.random() < 0.6:
                continue

            name = f"{parts[0]}_{parts[1]}_backup_{parts[2]}_{parts[3]}"
            if name in self.backups_sfc_instantiated:
                continue

            # --- CORREÇÃO DE DURAÇÃO ---
            # Calcula quanto tempo falta para a SFC acabar
            current_time = time.time()
            start_time = sfc_id_duration[sfc_id]['timer']
            total_duration = sfc_id_duration[sfc_id]['duration']
            
            elapsed = current_time - start_time
            remaining_duration = max(10, total_duration - elapsed + 10)

            # Preparação da nova SFC
            reduction_factor = self.standard_reduction_factor 
            
            sfc = network.get_sfc_by_id(sfc_id)
            vnf_info = sfc.vnfs_dict

            # --- CORREÇÃO DE LATÊNCIA ---
            # Herda o requisito original ou usa 10ms como fallback seguro
            original_latency_req = getattr(sfc, 'latency_request', 10)

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
                'duration': remaining_duration, # Valor corrigido
                'latency': original_latency_req # Valor corrigido (Antes era 7)
            }

            new_sfc = SFCGenerator(player_dict).generate()
            new_sfc.original_sfc_id = sfc_id
            new_sfc.is_backup = True
            backups_mount.append([new_sfc])

        return backups_mount
    
    def escolher_src_dst(self, dicionario, vnf_escolhida, latency_limit=10):
        chaves = list(dicionario.keys())
        
        if vnf_escolhida not in chaves:
            return None, None, None

        idx = chaves.index(vnf_escolhida)
        latency_dismiss = 0
        src = None
        dst = None

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
        
        # --- CORREÇÃO: Usa o limite passado como argumento ---
        latency_requirement = latency_limit - latency_dismiss
        
        return dst_node, src_node, latency_requirement

    def seletive_strategy(self, network, sfc_id_duration, threshold=0):
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
            
            if not server_info:
                continue

            for info in server_info:
                sfc_id = info[0]
                vnf = info[1]
                vnf_id = vnf.id
                parts = sfc_id.split("_")

                if len(parts) > 2 and parts[2] == 'backup':
                    continue

                if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration:
                    continue

                name = f"{parts[0]}_{parts[1]}_backup_{vnf_id}_{parts[2]}_{parts[3]}"
                if name in self.backups_sfc_instantiated:
                    continue

                sfc = network.get_sfc_by_id(sfc_id)
                vnf_info = sfc.vnfs_dict
                resources_info = next((i for i in vnf_info if i['name'] == vnf_id), None)
                
                sfc_rf = copy.deepcopy(network.sfc_route_info[sfc_id])
                if 'src' in sfc_rf: del sfc_rf['src']
                if 'dst' in sfc_rf: del sfc_rf['dst']

                # Location pode falhar se vnf_id não estiver na rota (ex: src/dst virtual)
                if vnf_id not in sfc_rf: continue
                location = sfc_rf[vnf_id][0]
                
                src_out = resources_info['in_bw']
                dst_in = resources_info['out_bw']
                cpu = resources_info['CPU']
                cache = resources_info['cache']

                original_latency = getattr(sfc, 'latency_request', 10)
                dst_node, src_node, latency_req = self.escolher_src_dst(sfc_rf, vnf_id, original_latency)
                
                if latency_req < 0 or dst_node is None:
                    continue

                # --- CORREÇÃO AQUI: Adicionado sufixo _b para consistência ---
                src_name = "src_virt"
                backup_vnf_name = vnf_id + "_b" 
                dst_name = "dst_virt"
                reduction_factor = self.standard_reduction_factor

                backup_sf_list = [
                    {
                        "type": 2, "name": src_name, "CPU": 0, "cache": 0, 
                        "in_bw": 0, "out_bw": 0, 
                        "latency": 0, "location": src_node
                    },
                    {
                        "type": 2, "name": backup_vnf_name, 
                        "CPU": cpu * reduction_factor, 
                        "cache": cache * reduction_factor, 
                        "in_bw": 0, 
                        "out_bw": dst_in * reduction_factor, 
                        "latency": 0, "original_loc": location, "original_sfc": sfc_id
                    },
                    {
                        "type": 2, "name": dst_name, "CPU": 0, "cache": 0, 
                        "in_bw": 0, "out_bw": 0, 
                        "latency": 0, "location": dst_node
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

    def create_backups(self, network):
        """Método corrigido para gerar backups reativos/base."""
        if not self.backup_activated:
            return [], None

        # Cria dicionário auxiliar necessário para as estratégias
        sfc_id_duration = {}
        for sfc_id, sfc in network.sfc_dict.items():
            start_t = getattr(sfc, 'arrival_time', time.time())
            sfc_id_duration[sfc_id] = {
                "timer": start_t,
                "duration": sfc.duration
            }

        # Se for SBRC ou Vegeta, usa estratégia seletiva como base
        if self.alg in ['vegeta', 'ga', 'SBRCMASKABLEPPO']:
            backups_mount = self.seletive_strategy(network, sfc_id_duration)
            return backups_mount, 'seletive'
        else:
            backups_mount = self.greedy_strategy(network, sfc_id_duration)
            return backups_mount, 'greedy'
        
    def create_contextual_mini_sfc(self, network, original_sfc: SFC, vnf_to_replicate_id, primary_node_id):
        """
        Cria uma Mini-SFC (3 saltos) para alocação via DRL.
        Contexto: A Origem é fixada no nó da VNF Anterior. 
                  O Destino é fixado no nó da VNF Seguinte.
        """
        sfc_id = original_sfc.id

        # --- [CORREÇÃO 1] Extração Robusta do Session ID ---
        # Tenta pegar atributo, senão faz o parse uma última vez
        if hasattr(original_sfc, 'session_id'):
            session_id = original_sfc.session_id
        else:
            # Fallback para o padrão sfc_pX_YYY
            parts = sfc_id.split('_')
            session_id = parts[-1]

        vnf_info = original_sfc.vnfs_dict
        
        # Obtém Especificações Técnicas da VNF
        target_vnf_info = next((info for info in original_sfc.vnfs_dict if info['name'] == vnf_to_replicate_id), None)
        
        factor = self.standard_reduction_factor
        
        if not target_vnf_info:
            return None

        # Determina Vizinhos (Contexto Físico)
        route_info = network.sfc_route_info.get(sfc_id)
        if not route_info: return None

        # Lógica para encontrar Nó Anterior e Próximo
        current_vnf_obj = original_sfc.get_vnf_by_id(vnf_to_replicate_id)
        
        # ENCONTRA LOCALIZAÇÃO FÍSICA ANTERIOR
        prev_vnf = original_sfc.get_previous_vnf(current_vnf_obj)
        if prev_vnf.id == 'src':
            prev_node = original_sfc.src.substrate_node
        else:
            prev_node = route_info[prev_vnf.id][0] 

        # ENCONTRA LOCALIZAÇÃO FÍSICA PRÓXIMA
        next_vnf = original_sfc.get_next_vnf(current_vnf_obj)
        if next_vnf.id == 'dst':
            next_node = original_sfc.dst.substrate_node
        else:
            next_node = route_info[next_vnf.id][0]

        # Constrói Dicionário da Mini-SFC
        backup_vnf_name = vnf_to_replicate_id + "_b"
        
        mini_sfc_vnfs = [
            # Nó Virtual de Entrada (Fixo no prev_node)
            {
                "type": 2, 
                "name": "src_virt",  # <--- NOME FIXO
                "CPU": 0, "cache": 0, "in_bw": 0, "out_bw": 0, "latency": 0, 
                "location": prev_node
            },
            
            # A VNF de Backup (O que queremos proteger)
            {
                "type": 2, 
                "name": vnf_to_replicate_id + "_b", 
                "CPU": target_vnf_info['CPU'] * factor, 
                "cache": target_vnf_info['cache'] * factor, 
                "in_bw": 0,   
                "out_bw": 0, 
                "latency": 0, 
                "original_sfc": sfc_id
            },
            
            # Nó Virtual de Saída (Fixo no next_node)
            {
                "type": 2, 
                "name": "dst_virt", # <--- NOME FIXO
                "CPU": 0, "cache": 0, "in_bw": 0, "out_bw": 0, "latency": 0, 
                "location": next_node
            }
        ]
        
        # --- CORREÇÃO 1: DURAÇÃO DINÂMICA ---
        current_time = time.time()
        start_time = getattr(original_sfc, 'arrival_time', current_time) 
        elapsed_time = current_time - start_time
        remaining_duration = max(10, original_sfc.duration - elapsed_time + 10)

        # --- CORREÇÃO 3: LATÊNCIA DINÂMICA ---
        latency_constraint = getattr(original_sfc, 'latency_request', 10)
        
        # --- [CORREÇÃO 2] ID Único e Explícito ---
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

        # --- Injeção de Metadados ---
        mini_sfc.original_sfc_id = original_sfc.id
        mini_sfc.is_backup = True
        mini_sfc.target_vnf_id = vnf_to_replicate_id
        mini_sfc.session_id = session_id
        
        # [CORREÇÃO] A lógica estava invertida. 
        # O fluxo é: Prev Node -> [VNF Backup] -> Next Node
        mini_sfc.src_virt = prev_node  # O "src" virtual é de onde vem o dado (nó anterior)
        mini_sfc.dst_virt = next_node  # O "dst" virtual é para onde vai o dado (nó seguinte)

        return mini_sfc


    def get_backups_instantiated_q(self):
        vnfs_backup_instantiate = 0
        backups = self.backups_sfc_instantiated
        
        if self.alg == 'ga':
            for backup_id, original_sfc in backups.items():
                sfc_backups = self.sfcs_backups_instatiated.get(original_sfc, [])
                vnfs_backup_instantiate += len(sfc_backups)
        else:
            vnfs_backup_instantiate = len(list(backups.keys())) * 4

        return vnfs_backup_instantiate