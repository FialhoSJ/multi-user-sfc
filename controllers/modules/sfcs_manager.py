
import sys
import time
import re
import copy
import random
from typing import Optional
from controllers.modules.backup_manager import BackupManager
from controllers.sfc_generator import SFCGenerator
from utils.k_shortest_paths import k_shortest_paths


class SFCManager:
    def __init__(self,args,backup_manager,alg):
        self.sfcs_tracker = {} # sfcs that are running in the simulation
        self.sfcs_routing_info = {}
        self.sfc_id_duration = {}
        self.backup_manager: Optional[BackupManager] = backup_manager
        self.alg_name = alg.name
        self.alg = alg
        self.player_sfc_id_list = {}
        self.sfcs_deployed_history = []
        #self.sfcs_that_crashed = []
        self.sfs_backup = {}
        self.last_sfc_release = False
        self.risk_servers = []
        self.counter = 0 # how many sfcs are running
        self.verbose = True
        self.crashed_servers = []
        self.sfc_reuse = {}

    def submit_solution(self, sfc_list,solution,substrate_network,is_backup=False) -> bool:
        # if sfc.id in self.sfc_list:
        #     return False
        for sfc in sfc_list:
            substrate_network.deploy_sfc(sfc, solution[sfc.id]['route_info'])
        
        group_id = sfc_list[0].group_id
        duration = sfc_list[0].duration
        if group_id in self.sfcs_tracker:
            raise ValueError("SFC já instanciada")
        else:
            self.sfcs_tracker[group_id] = {"sfc_list":sfc_list,'solution':solution,'duration':duration,'timer':time.time()}

        self.counter += 1 # at this time all verifications are done. So we add 1 to counter of sfc

        #return {"current_time":current_time,"latency":latency,"run_duration":run_duration,"resource_info":r_info,"is_success":is_success,"route_info":route_info,"fail_reason":fail_reason,"backup_sfc":is_backup}
    

    def undeploy_sfc(self, sfc_id: str,substrate_network,take_out_backup=True) -> None:
        if not self.is_Backup(sfc_id): # Se não for uma sfc de backup
            substrate_network.undeploy_sfc(sfc_id)    
            

            #del self.sfc_id_duration[sfc_id]
            # Se a SFC possui backup, retire ela da lista de sfcs com backup e faça undeploy do backup 
            if take_out_backup:
                if sfc_id in list(self.sfs_backup.keys()): 
                    self.undeploy_sfc_backups(sfc_id,substrate_network)
        else:
            self.remove_backup_by_id(sfc_id,substrate_network)

        # TODO fazer a lógica de remover sfs
        # quando o tempo acaba
        substrate_network.update()
        # except ValueError:
        #     print(f"Error: SFC ID {sfc_id} not found in the list when trying to remove.")
        #     return -1

    def create_backups(self,network):
        backups_mount = []
        sfc_id_duration = copy.deepcopy(self.sfc_id_duration)
        #node_fail_p = network.nodes_reliability.copy()

        if self.alg_name in ['vegeta','ga']:
            self.clean_backups(network)
            backups_mount = self.backup_manager.seletive_strategy(network,sfc_id_duration) 
        else:        
            backups_mount = self.backup_manager.greedy_strategy(network,sfc_id_duration)

        if backups_mount:
            for backup in backups_mount:
                sfc = backup[0]
                results_dict = self.deploy_sfc(sfc,network,is_backup=True)
                
                if sfc.id in self.backup_manager.backups_sfc_instantiated:
                    continue

                if results_dict['is_success']:
                    if self.alg_name == 'ga':
                        original_sfc = sfc.vnfs_dict[1]['original_sfc'] 
                        vnfs_dict = sfc.vnfs_dict[1]
                        vnf_id = vnfs_dict['name']
                    else:
                        split = sfc.id.split("_")
                        original_sfc = f"{split[0]}_{split[1]}_{split[3]}_{split[4]}" 
                        vnf_id = None

                    if original_sfc not in self.backup_manager.sfcs_backups_instatiated: # Se ela nunca teve backup, então inicializa
                        self.backup_manager.sfcs_backups_instatiated[original_sfc] = []  # Inicializa uma lista para backups

                    # Adiciona um backup à lista de backups do original
                    self.backup_manager.sfcs_backups_instatiated[original_sfc].append({"sfc_backup_id": sfc.id,"vnf_id": vnf_id,"route_info": results_dict["route_info"]})
                    self.backup_manager.backups_sfc_instantiated[sfc.id] = original_sfc

    def clean_backups(self, network):
        for backup in list(self.backup_manager.backups_sfc_instantiated.keys()):
            if backup in self.backup_manager.backups_activated:
                continue
            if random.random() < 0.7:  # 70% de chance de apagar
                self.remove_backup_by_id(backup, network)
        network.update()

        # Limpa os dicionários
        # self.backup_manager.sfcs_backups_instatiated.clear()
        # self.backup_manager.backups_sfc_instantiated.clear()

    def undeploy_sfc_backups(self,sfc_id,substrate_network): # Retira os backups da SFC, inclusive os ativos
        backups_removed = self.backup_manager.sfcs_backups_instatiated.pop(sfc_id)
        for backup in backups_removed:
            backup_id = backup["sfc_backup_id"]
            del self.backup_manager.backups_sfc_instantiated[backup_id]
            if backup_id in self.backup_manager.backups_activated:
                self.backup_manager.backups_activated.remove(backup_id)
            substrate_network.undeploy_sfc(backup_id)

    def remove_backup_by_id(self, backup_id, substrate_network): # Remove todos os backups (menos os ativos)
        if backup_id in self.backup_manager.backups_sfc_instantiated:
            
            original_sfc = self.backup_manager.backups_sfc_instantiated[backup_id]
            if original_sfc in self.backup_manager.sfcs_backups_instatiated:
                backups = self.backup_manager.sfcs_backups_instatiated[original_sfc]
                self.backup_manager.sfcs_backups_instatiated[original_sfc] = [backup for backup in backups if backup["sfc_backup_id"] != backup_id]
                
                if len(self.backup_manager.sfcs_backups_instatiated[original_sfc]) == 0:
                    del self.backup_manager.sfcs_backups_instatiated[original_sfc]
                del self.backup_manager.backups_sfc_instantiated[backup_id]
                
                if backup_id in self.sfc_list:
                    self.sfc_list.remove(backup_id)
                    del self.sfc_id_duration[backup_id]
                substrate_network.undeploy_sfc(backup_id)
            else:
                try: 
                    del self.backup_manager.backups_sfc_instantiated[backup_id]
                    substrate_network.undeploy_sfc(backup_id)
                except:
                    pass

    def check_backups(self,sfc_id,vnfs,substrate_network):
        if sfc_id in list(self.backup_manager.sfcs_backups_instatiated.keys()):
            if self.alg_name == 'ga':
                for vnf in vnfs:
                    vnf_id = vnf.id
                    backups = self.backup_manager.sfcs_backups_instatiated[sfc_id]
                    for backup in backups:
                        vnf_backup = backup['vnf_id'].removesuffix("_b") 
                        if vnf_id == vnf_backup:
                            print("Tem backup daquela vnf caída, então ativa")
                            self.trigger_sfc_backup(sfc_id,vnf_id,substrate_network,self.crashed_servers[0])
                            self.backup_manager.backups_activated.append(backup['sfc_backup_id'])
                            return backup
            else:
                backup = self.backup_manager.sfcs_backups_instatiated[sfc_id][0]
                backup_id = backup['sfc_backup_id']
                duration = self.sfc_id_duration[sfc_id]['duration']
                timer = self.sfc_id_duration[sfc_id]['timer']
                self.sfc_id_duration[backup_id] = {"duration": duration,"timer":timer}
                self.sfc_list.append(backup_id)
                #self.sfcs_routing_info[backup_id] = backup['route_info']
                # self.undeploy_sfc(sfc_id,substrate_network)
                return backup 
        return False

    def is_Backup(self,sfc_id):
        return True if sfc_id.split("_")[2] == 'backup' else False

    def trigger_sfc_backup(self,sfc_id,vnf_id, substrate_network, node_id):
        if sfc_id in list(self.sfs_backup.keys()):
            substrate_network.reset_vnf_cpu_request(node_id,sfc_id, vnf_id)

    def check_sfc_duration(self):
        remove_list = []
        current_time = time.time()  # Obtém o tempo atual uma única vez para evitar múltiplas chamadas
        for sfc_list_id, info in list(self.sfcs_tracker.items()):  # `info` é o dicionário com 'duration' e 'timer'
            elapsed_time = current_time - info["timer"]  # Calcula o tempo decorrido
            if elapsed_time >= info["duration"]:  # Verifica se a duração foi ultrapassada
                remove_list.append(sfc_list_id)
        return remove_list

    def get_list_sfc_duration(self, threshold=20):
        sfc_list_duration = []
        for sfc_id, info in self.sfc_id_duration.items():  # `info` contém as informações de cada SFC
            if info["duration"] <= threshold:
                sfc_list_duration.append(sfc_id)  # Adiciona o ID à lista
        return sfc_list_duration

    def check_resources_exceed(self,is_success,sfc_id,substrate_network,fail_reason):
        if is_success:
            for node in substrate_network.graph.nodes():
                if node in self.crashed_servers:
                    continue
                limit = 100
                cpu_used = substrate_network.get_node_cpu_used(node)
                cache_used = substrate_network.get_node_cache_used(node)
                
                if cpu_used > limit or cache_used > limit:
                    self.undeploy_sfc(sfc_id,substrate_network)
                    is_success = False
                    fail_reason = "exceed" 
                    return is_success,fail_reason
                
            for edge in substrate_network.graph.edges():
                bandwidth_used = substrate_network.get_link_bandwidth_used(edge[0], edge[1])
                bandwidth_capacity = substrate_network.get_link_bandwidth_capacity(edge[0], edge[1])
                if bandwidth_used > bandwidth_capacity:
                    is_success = False
                    fail_reason = "band_exceed"
                    self.undeploy_sfc(sfc_id,substrate_network)
                    return is_success,fail_reason
            return is_success,fail_reason
        else:
            return False,fail_reason
    
    def set_risk_sfcs(self,servers,network):
        if len(servers) == 0:
            return []
        #self.risk_servers = servers
        #server_infos = []
        #sfcs_id_backup_made = []        
        backups_mount = []
        sfc_id_duration = copy.deepcopy(self.sfc_id_duration)
        for server in servers:
            server_info = network.get_node_sfc_vnf_list(server)
            sfcs_l = []
            server_infos = []
            current_time = time.time()
            if server_info != []:
                for info in server_info:
                    server_infos.append(info)
                    # TODO acho que dava pra ter mais regras se cri ou não o backup (tempo que ainda resta para aquela SF)
                    sfc_id = info[0]
                    vnf = info[1]
                    vnf_id = vnf.id

                    if sfc_id.split("_")[2]=='backup':
                        continue

                    if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration:
                        continue 

                    # sfcs_with_backup = list(self.sfs_backup.keys())
                    # if sfc_id in sfcs_with_backup or sfc_id in sfcs_id_backup_made: # TODO isso deve ser revisto posteriormente (somente um backup por SFC)
                    #     continue
                    
                    split = sfc_id.split("_")
                    name = split[0] + "_" + split[1] + '_backup_' + split[2] + "_" +split[3]
                    
                    try:
                        network.get_sfc_by_id(name)
                        continue
                    except:
                        pass

                    sfc = network.get_sfc_by_id(sfc_id)
                    vnf_info = sfc.vnfs_dict
                    resources_info = [info for info in vnf_info if info['name'] == vnf_id][0]
                    sfc_rf = network.sfc_route_info[sfc_id]
                    
                    src = None
                    location = None
                    dst = None
                    
                    #src_in = 0 
                    src_out = resources_info['in_bw']
                    
                    dst_in = resources_info['out_bw']
                    #dst_out = 0

                    cpu = resources_info['CPU']
                    cache = resources_info['cache']

                    i = 0
                    latency_dismiss = 0
                    for key,value in sfc_rf.items():        
                        if i == 1:
                            src = value[0]
                            if src == 0: # SF IA 
                                src = value[-1]
                            else:
                                src = value[0]
                                latency_dismiss = latency_dismiss + len(value)-1
                                print()
                            break
                        if key == vnf_id:
                            location = value[0]
                            dst = value[-1]
                            latency_dismiss = latency_dismiss + len(value)-1
                            i = i + 1
                    
                    new_sfc_list = []

                    backup_sf_list = []
                    split = sfc_id.split("_")
                    name = split[0] + "_" + split[1] + '_backup_'+vnf_id+ '_' + split[2] + "_" +split[3]
                    src_name = "src_" + name
                    dst_name = "dst_" + name
                    backup_sf_list.append({"type": 2, "name":src_name,"CPU": 0, "cache": 0, "in_bw": 0, "out_bw":src_out ,"latency":0,"location":src})
                    backup_sf_list.append({"type": 2, "name":vnf_id,"CPU": cpu, "cache": cache, "in_bw": src_out, "out_bw": dst_in,"latency":0,"restriction":location,"original_sfc":sfc_id})
                    backup_sf_list.append({"type": 2, "name":dst_name,"CPU": 0, "cache": 0, "in_bw": dst_in, "out_bw":0 ,"latency":0,"location":dst})

                    new_sfc_dict = {}
                    new_sfc_dict["name"] = name
                    new_sfc_dict["vnf_list"] = backup_sf_list
                    new_sfc_dict["bandwidth"] = sfc.input_throughput
                    new_sfc_dict["src_node"] = src
                    new_sfc_dict["dst_node"] = dst

                    duration = sfc_id_duration[sfc_id]["duration"]-(current_time-sfc_id_duration[sfc_id]["timer"])
                    new_sfc_dict["duration"] = duration  + 20 
                    new_sfc_dict["latency"] =  sfc.latency_request - self.calculate_latency(sfc_rf) + latency_dismiss  # TODO deve ter aqui  um cálculo para não passar da latencia da original se implementada
                    # new_sfc_dict["original_sfc"] = sfc_rf
                    # new_sfc_dict["restrictions"] = [location]
                    new_sfc = SFCGenerator(new_sfc_dict).generate()
                    
                    #self.sfs_backup[sfc_id] = {"sfc_backup_id":new_sfc.id,"vnf_id":vnf_id}  # Por enquanto teremos apenas uma SFC de Backup por SFC 
                
                    new_sfc_list.append(new_sfc)
                    backups_mount.append(new_sfc_list)
                    #sfcs_id_backup_made.append(sfc_id)
        return backups_mount






















    def set_sfc_reuse(self,sfc,route_info,shareable_sfs):
        return 1
        conta = 0
        if route_info != None :
            if route_info != {}:
                if  route_info != False:
                    #sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                    cpu_saved = 0
                    cache_saved = 0

                    total_cpu_req = 0
                    total_cache_req = 0

                    for vnf,servers in route_info.items():
                        if vnf not in ['src','dst']:
                            server_used = servers[0]
                            shared_sfs_in_node = list(map(lambda sf: sf.id, shareable_sfs[server_used]))
                            cpu_req   = sfc.vnfs[vnf].get_cpu_request()
                            cache_req =  sfc.vnfs[vnf].get_cache_request()
                            if vnf in shared_sfs_in_node:
                                # Com reúso
                                cpu_saved += cpu_req
                                cache_saved += cache_req

                            total_cpu_req += cpu_req
                            total_cache_req += cache_req
                    conta = (cpu_saved + cache_saved) / (total_cpu_req + total_cache_req) 
                    self.sfc_reuse[sfc.id] = conta
        return conta
        

    # def check_sfc_duration(self):
    #     remove_list = []
    #     for sfc_id, duration in list(self.sfc_id_duration.items()):
    #         if duration <= 1:                    
    #             remove_list.append(sfc_id)
    #             # if duration is over, sfc routing info no longer needed
    #             del self.sfcs_routing_info[sfc_id]
    #             continue
    #         self.sfc_id_duration[sfc_id] = duration - 1
    #     return remove_list


    def get_shareable_sfs(self, sfc):
        """_summary_

        Args:
            route_info (Dict): A provided route info.

        Returns:
            List[Tuple]: A list of SFs candidates.
        """
        shareable_sfs_ar = []
        src_sf = sfc.get_src_vnf()
        dst_sf = sfc.get_dst_vnf()
        sf = src_sf
        while sf != dst_sf:
            for (node, candidate_sf_id, candidate_sf) in self.shareable_list:
                    if candidate_sf_id == sf.id:
                        # if max number of connections reached, cant be used.
                        print("Chegando onde não devia")
                        if len(self.substrate_network.sf_linkeds[candidate_sf]) != self.max_connections:
                            print("shareable sf found.........")
                            shareable_sfs_ar.append((node, candidate_sf_id, candidate_sf))
            sf = sf.get_next_vnf()
        return shareable_sfs_ar

    def calculate_latency(self, route_info):
        total_latency = sum(
            len(path) - 1 for key, path in route_info.items() if path and key not in ["src", "dst"]
        )
        return total_latency


    def deploy_success(self, sfc: object) -> None:
        """Print success message."""
        if self.verbose == True:
            print("deploy succeed, sfc: ", sfc.id)

    def deploy_failed(self, sfc: object) -> None:
        """Print failure message."""
        if self.verbose == True:
            print(" deploy FAILED, sfc: ", sfc.id)


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



    def musfico_method(self, sfc,substrate_network):
        """
        Calculates the latency and obtains the route information for the musfico algorithm.

        Args:
            sfc (object): The service function chain (SFC) object containing the SF details.

        Returns:
            tuple: A tuple containing the latency and route_info.
        """
        route_info = {}
        # get stored route info
        route_info = copy.deepcopy(self.sfcs_routing_info[sfc.id])
        # gets dst vnf
        dst_vnf = sfc.get_dst_vnf()
        previous_vnf = sfc.get_previous_vnf(dst_vnf)
        dst_substrate_node = sfc.get_substrate_node(dst_vnf)
        prev_vnf_node = route_info[previous_vnf.id][0]
        # applies k shortest to link the last vnf with the previous one
        shortest_path = k_shortest_paths(substrate_network, prev_vnf_node, 
                                        dst_substrate_node, k=1, weight='latency')
        # get the shortest among the k shortest paths
        route_info[previous_vnf.id] = shortest_path[0]
        latency = 0
        for vnf_id in route_info.keys():
            if vnf_id == 'src':
                continue
            path = route_info[vnf_id]
            for i in range(len(path) - 1):
                edge_latency = substrate_network.get_link_latency(
                    path[i], path[i + 1])
                latency += edge_latency

        if latency > sfc.get_latency_request() or latency < 0:
            route_info = False
            latency = None

        return latency, route_info
