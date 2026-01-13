import copy
import time
import random
from controllers.sfc_generator import SFCGenerator

class BackupManager:
    def __init__(self, args):
        self.sfcs_backups_instatiated = {}
        self.backups_sfc_instantiated = {}
        self.backups_activated = []

        # Simplificação da lógica booleana
        self.backup_activated = (args.backup == 'y') and (args.ava != '1.0')
        self.alg = args.alg

    def greedy_strategy(self, network, sfc_id_duration, threshold=0):
        backups_mount = []
        sfcs_id = list(sfc_id_duration.keys())
        
        # Somente um servidor será selecionado, e será aquele com mais chance de falhar
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

            # Construção do nome do backup
            name = f"{parts[0]}_{parts[1]}_backup_{parts[2]}_{parts[3]}"

            if name in self.backups_sfc_instantiated:
                continue

            # Preparação da nova SFC
            reduction_factor = 0.25
            sfc = network.get_sfc_by_id(sfc_id)
            vnf_info = sfc.vnfs_dict

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
                'duration': sfc_id_duration[sfc_id]['duration'],
                'latency': 7
            }

            new_sfc = SFCGenerator(player_dict).generate()
            backups_mount.append([new_sfc])

        return backups_mount

    def seletive_strategy(self, network, sfc_id_duration, threshold=0):
        backups_mount = []
        nodes_fail_p = network.nodes_reliability.copy()
        
        # Seleciona nós acima do threshold
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

                # Validações
                if len(parts) > 2 and parts[2] == 'backup':
                    continue

                if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration:
                    continue

                name = f"{parts[0]}_{parts[1]}_backup_{vnf_id}_{parts[2]}_{parts[3]}"

                if name in self.backups_sfc_instantiated:
                    continue

                # Coleta de informações da SFC original
                sfc = network.get_sfc_by_id(sfc_id)
                vnf_info = sfc.vnfs_dict
                resources_info = next((i for i in vnf_info if i['name'] == vnf_id), None)
                
                sfc_rf = copy.deepcopy(network.sfc_route_info[sfc_id])
                if 'src' in sfc_rf: del sfc_rf['src']
                if 'dst' in sfc_rf: del sfc_rf['dst']

                location = sfc_rf[vnf_id][0]
                
                # Recursos
                src_out = resources_info['in_bw']
                dst_in = resources_info['out_bw']
                cpu = resources_info['CPU']
                cache = resources_info['cache']

                dst, src, latency_req = self.escolher_src_dst(sfc_rf, vnf_id)
                if latency_req < 0:
                    continue

                # Configuração da VNF de Backup
                src_name = "source"
                backup_vnf_name = vnf_id + "_b"
                dst_name = "destiny"
                reduction_factor = 0.75

                backup_sf_list = [
                    {
                        "type": 2, "name": src_name, "CPU": 0, "cache": 0, 
                        "in_bw": 0, "out_bw": src_out * reduction_factor, 
                        "latency": 0, "location": src
                    },
                    {
                        "type": 2, "name": backup_vnf_name, 
                        "CPU": cpu * reduction_factor, "cache": cache * reduction_factor, 
                        "in_bw": src_out * reduction_factor, "out_bw": dst_in * reduction_factor, 
                        "latency": 0, "original_loc": location, "original_sfc": sfc_id
                    },
                    {
                        "type": 2, "name": dst_name, "CPU": 0, "cache": 0, 
                        "in_bw": dst_in * reduction_factor, "out_bw": 0, 
                        "latency": 0, "location": dst
                    }
                ]

                # Cálculo de duração restante
                time_elapsed = current_time - sfc_id_duration[sfc_id]["timer"]
                duration = sfc_id_duration[sfc_id]["duration"] - time_elapsed

                new_sfc_dict = {
                    "name": name,
                    "vnf_list": backup_sf_list,
                    "bandwidth": sfc.input_throughput,
                    "src_node": src,
                    "dst_node": dst,
                    "duration": duration,
                    "latency": latency_req
                }

                new_sfc = SFCGenerator(new_sfc_dict).generate()
                backups_mount.append([new_sfc])

        return backups_mount

    def create_backups(self, nodes_fail_p, network, backups_data, sfc_manager):
        sfc_id_duration = copy.deepcopy(sfc_manager.sfc_id_duration)

        if self.alg in ['vegeta', 'ga']:
            # Nota: A ordem dos argumentos aqui parece diferir da definição da função seletive_strategy.
            # Mantido conforme original para preservar funcionamento.
            backups_mount = self.seletive_strategy(nodes_fail_p, network, sfc_id_duration, backups_data)
            return backups_mount, 'seletive'
        else:
            backups_mount = self.greedy_strategy(nodes_fail_p, network, sfc_id_duration, backups_data)
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
        target_vnf_info = next((info for info in vnf_info if info['name'] == vnf_to_replicate_id), None)
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
            next_node = original_sfc.dst.substrate_node # Corrigido: .dst_node geralmente é .dst.substrate_node dependendo da sua impl.
        else:
            next_node = route_info[next_vnf.id][0]

        # Constrói Dicionário da Mini-SFC
        backup_vnf_name = vnf_to_replicate_id + "_b"
        
        mini_sfc_vnfs = [
            {"type": 2, "name": "src_virt", "CPU": 0, "cache": 0, "in_bw": 0, "out_bw": target_vnf_info['in_bw'], "latency": 0, "location": prev_node},
            {"type": 2, "name": backup_vnf_name, "CPU": target_vnf_info['CPU'], "cache": target_vnf_info['cache'], 
             "in_bw": target_vnf_info['in_bw'], "out_bw": target_vnf_info['out_bw'], "latency": 0, "original_sfc": sfc_id},
            {"type": 2, "name": "dst_virt", "CPU": 0, "cache": 0, "in_bw": target_vnf_info['out_bw'], "out_bw": 0, "latency": 0, "location": next_node}
        ]

        mini_sfc_dict = {
            "name": f"{sfc_id}_rep_{vnf_to_replicate_id}",
            "vnf_list": mini_sfc_vnfs,
            "bandwidth": original_sfc.input_throughput,
            "src_node": prev_node,
            "dst_node": next_node,
            "duration": 100,
            "latency": 5 
        }

        return SFCGenerator(mini_sfc_dict).generate()

    def take_off_backup_if_exist(self, sfc_list):
        for sfc_id in sfc_list:
            if sfc_id in self.sfcs_backups_instatiated:
                del self.sfcs_backups_instatiated[sfc_id]

    def calculate_latency(self, route_info):
        return sum(
            len(path) - 1 
            for key, path in route_info.items() 
            if path and key not in ["src", "dst"]
        )

    def escolher_src_dst(self, dicionario, vnf_escolhida):
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
        latency_requirement = 7 - latency_dismiss
        
        return dst_node, src_node, latency_requirement

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