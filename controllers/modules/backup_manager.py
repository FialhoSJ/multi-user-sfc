import copy
import time
import random
from controllers.sfc_generator import SFCGenerator
from algorithms.environments.env_sbrc import SFC_AllocationEnv

class BackupManager:
    def __init__(self, args):
        self.sfcs_backups_instatiated = {}
        self.backups_sfc_instantiated = {}
        self.backups_activated = []
        
        # --- [MODIFICAÇÃO 1] TRAVA DE ALGORITMO ---
        target_algorithm = 'SBRCMASKABLEPPO'  # Nome exato do algoritmo
        
        is_target_alg = (args.alg == target_algorithm)
        user_wants_backup = (args.backup == 'y') and (args.ava != '1.0')
        
        # Só ativa se o usuário quis E se for o algoritmo SBRC
        self.backup_activated = user_wants_backup and is_target_alg
        self.alg = args.alg
        
        if user_wants_backup and not is_target_alg:
            print(f"[BackupManager] INFO: Backup proativo desativado. '{args.alg}' não suporta essa estratégia.")

        self.standard_reduction_factor = 1.0
        
        
    def rl_based_strategy(self, network, sfc_id_duration, agent):
        """
        Usa o Agente RL (SBRC) para decidir onde alocar backups.
        Retorna uma lista de [Mini-SFCs] com a rota JÁ CALCULADA anexada.
        """
        backups_mount = []
        target_reliability = 0.99
        
        sorted_sfcs = sorted(list(sfc_id_duration.keys()))

        for sfc_id in sorted_sfcs:
            # ... (Lógica de filtro existente: filtros de elegibilidade e identificação de necessidade) ...
            if "backup" in sfc_id: continue 
            if sfc_id in self.sfcs_backups_instatiated: continue
            if sfc_id not in network.sfc_dict: continue
            
            sfc = network.get_sfc_by_id(sfc_id)
            route_info = network.sfc_route_info.get(sfc_id, {})
            
            # ... (Lógica de identificação do elo fraco) ...
            needs_backup = False
            target_vnf = None
            weak_node = None
            
            # (Seu loop de verificação de confiabilidade aqui...)
            for vnf_id, path in route_info.items():
                if vnf_id in ['src', 'dst'] or not path: continue
                node = path[0]
                try:
                    reliability = network.get_node_reliability(node)
                except AttributeError:
                    reliability = 1.0
                
                if reliability < target_reliability:
                    needs_backup = True
                    target_vnf = vnf_id
                    weak_node = node
                    break
            
            if not needs_backup: continue

            # 3. Cria o Contexto do Backup (Mini-SFC)
            mini_sfc = self.create_contextual_mini_sfc(network, sfc, target_vnf, weak_node)
            if not mini_sfc: continue

            # --- CORREÇÃO DO CRASH (KeyError) AQUI ---
            
            # 1. Trabalhamos com uma CÓPIA do grafo para não sujar a topologia oficial com nós móveis temporários
            graph_for_rl = copy.deepcopy(network.graph)
            
            # 2. Injeção do Nó Móvel E da Conexão (Link)
            mobile_node_id = getattr(mini_sfc, 'mobile_node', None)
            closer_router_id = getattr(mini_sfc, 'closer_router', None)

            if mobile_node_id:
                # A) Adiciona o Nó (Dispositivo Móvel) se ele existir na rede móvel
                if mobile_node_id in network.md_graph and mobile_node_id not in graph_for_rl:
                    md_data = network.md_graph.nodes[mobile_node_id]
                    # Injeta o nó com todos os seus atributos (cpu, cache, posição, etc)
                    graph_for_rl.add_node(mobile_node_id, **md_data)
                
                # B) Adiciona a Aresta (Conexão Wireless) - Igual ao sfcs_instatiator.py
                # O nó móvel precisa estar conectado ao seu roteador de borda para ser alcançável
                if closer_router_id and closer_router_id in graph_for_rl:
                    router_data = graph_for_rl.nodes[closer_router_id]
                    
                    # Calcula a capacidade disponível no canal wireless do roteador
                    w_cap = router_data.get('w_channel_capacity', 0.0)
                    w_used = router_data.get('w_channel_used', 0.0)
                    wireless_free = max(0.0, w_cap - w_used)
                    
                    # Adiciona a aresta conectando o Mobile Device ao Roteador
                    # Usamos signal_latency=1 como padrão, conforme visto no instatiator
                    graph_for_rl.add_edge(
                        mobile_node_id, 
                        closer_router_id, 
                        bandwidth_capacity=wireless_free, 
                        bandwidth_used=0.00, 
                        latency=1, 
                        services_in_transit={}
                    )

            # 3. Atualiza a lista de nós válidos para o RL (Servidores + Nó Móvel)
            all_servers = [n for n, d in graph_for_rl.nodes(data=True) if d.get('type') != 'router']

            # 4. Configura o Ambiente RL usando o grafo modificado
            env = SFC_AllocationEnv(
                valid_nodes=all_servers,
                list_graph=[graph_for_rl], # <--- Passamos o grafo contendo o nó móvel
                list_sfc=[mini_sfc],
                is_training=False
            )
            
            # ... (Restante da lógica: forbidden nodes, agent execution) ...
            forbidden = [weak_node]
            if isinstance(weak_node, (int, float)):
                forbidden.append(weak_node - 0.1 if weak_node % 1 == 0.1 else weak_node + 0.1)
            env.set_forbidden_nodes(forbidden)

            agent.install_SFC(mini_sfc)
            
            # Tratamento de erro robusto caso o agente falhe
            try:
                agent.install_substrate_network(graph_for_rl)
                success = agent.start_algorithm(env)
            except KeyError as e:
                print(f"[BackupManager] Erro crítico no RL para SFC {sfc_id}: {e}. Pulando.")
                success = False
            

            if success:
                route_info_backup = agent.get_route_info()
                mini_sfc.pre_calculated_route = route_info_backup
                backups_mount.append([mini_sfc])

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
        
    def create_contextual_mini_sfc(self, network, original_sfc, vnf_to_replicate_id, primary_node_id):
        """
        Cria uma Mini-SFC (3 saltos) para alocação via DRL.
        Contexto: A Origem é fixada no nó da VNF Anterior. 
                  O Destino é fixado no nó da VNF Seguinte.
        """
        sfc_id = original_sfc.id
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
            {"type": 2, "name": "src_virt", "CPU": 0, "cache": 0, "in_bw": 0, 
            "out_bw": 0, "latency": 0, "location": prev_node},
            
            # --- [MODIFICAÇÃO 2] BANDA ZERO (Cold Standby) ---
            {"type": 2, "name": vnf_to_replicate_id + "_b", 
            "CPU": target_vnf_info['CPU'] * factor, 
            "cache": target_vnf_info['cache'] * factor, 
            "in_bw": 0,   # Define 0 para não gastar link agora
            "out_bw": 0,  # Define 0 para não gastar link agora
            "latency": 0, "original_sfc": sfc_id},
            # -------------------------------------------------
            
            {"type": 2, "name": "dst_virt", "CPU": 0, "cache": 0, 
            "in_bw": 0, "out_bw": 0, "latency": 0, "location": next_node}
        ]
        
        # --- CORREÇÃO 1: DURAÇÃO DINÂMICA ---
        current_time = time.time()
        start_time = getattr(original_sfc, 'arrival_time', current_time) 
        elapsed_time = current_time - start_time
        remaining_duration = max(10, original_sfc.duration - elapsed_time + 10)

        # --- CORREÇÃO 3: LATÊNCIA DINÂMICA ---
        # Herda o requisito original. O Agente tentará minimizar a latência para caber neste teto.
        # Se disponível, pegamos request, senão um padrão seguro (ex: 10ms)
        latency_constraint = getattr(original_sfc, 'latency_request', 10)
        
        

        mini_sfc_dict = {
            "name": f"{sfc_id}_backup",
            "vnf_list": mini_sfc_vnfs,
            "bandwidth": original_sfc.input_throughput,
            "src_node": prev_node,
            "dst_node": next_node,
            "duration": remaining_duration, 
            "latency": latency_constraint, # <--- Valor Corrigido
            "closer_router": getattr(original_sfc, 'closer_router', None),
            "mobile_node": getattr(original_sfc, 'dst_node', None)
        }

        mini_sfc = SFCGenerator(mini_sfc_dict).generate()

        mini_sfc.original_sfc_id = original_sfc.id
        mini_sfc.is_backup = True

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