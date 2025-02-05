import copy
from controllers.sfc_generator import SFCGenerator
import time

class BackupManager:
    def __init__(self,args):
        self.backups_instatiated = []
        self.sfc_backup = {}
        self.backup_activated = (args.backup) == 'y'
        self.alg = args.alg


    def greedy_strategy(self,servers,network,sfc_id_duration):
        pass

    def seletive_strategy(self,nodes_fail_p,network,sfc_id_duration,threshold=0):
        backups_mount = []
        
        nodes_highest_p = {node: rel for node, rel in nodes_fail_p.items() if rel > threshold}
        chosen_server = max(nodes_highest_p, key=nodes_highest_p.get)
        servers = [chosen_server] # Somente um servidor será selecionado, e será aquele com mais chance de falhar
        
        if len(servers) == 0:
            return []
        
        for server in servers:
            server_info = network.get_node_sfc_vnf_list(server)
            current_time = time.time()
            if server_info != []:
                for info in server_info:
                    # TODO acho que dava pra ter mais regras se cri ou não o backup (tempo que ainda resta para aquela SF)
                    sfc_id = info[0]
                    vnf = info[1]
                    vnf_id = vnf.id

                    if sfc_id.split("_")[2]=='backup': # Se for backup, não cria backup
                        continue

                    if sfc_id not in network.sfc_dict or sfc_id not in sfc_id_duration: # Se não estiver instanciada, passa (situação de erro)
                        continue 
                    
                    # try:
                    #     network.get_sfc_by_id(name)
                    #     continue
                    # except:
                    #     pass

                    sfc = network.get_sfc_by_id(sfc_id)
                    vnf_info = sfc.vnfs_dict
                    resources_info = [info for info in vnf_info if info['name'] == vnf_id][0]
                    sfc_rf = copy.deepcopy(network.sfc_route_info[sfc_id])
                    del sfc_rf['src']
                    del sfc_rf['dst']

                    src = None
                    location = sfc_rf[vnf_id][0]
                    dst = None
                    
                    #src_in = 0 
                    src_out = resources_info['in_bw']
                    
                    dst_in = resources_info['out_bw']
                    #dst_out = 0

                    cpu = resources_info['CPU']
                    cache = resources_info['cache']

                    i = 0
                    latency_dismiss = 0
                    src,dst,latency_req = self.escolher_src_dst(sfc_rf,vnf_id)
                    
                    if latency_req < 0:
                        continue

                    new_sfc_list = []
                    backup_sf_list = []

                    # split = sfc_id.split("_")
                    # name = split[0] + "_" + split[1] + '_backup_' + split[2] + "_" +split[3]

                    split = sfc_id.split("_")
                    name = split[0] + "_" + split[1] + '_backup_'+ vnf_id + '_' + split[2] + "_" +split[3]
                    
                    src_name = "src" #+ vnf_id
                    dst_name = "dst"

                    backup_sf_list.append({"type": 2, "name":src_name,"CPU": 0, "cache": 0, "in_bw": 0, "out_bw":src_out ,"latency":0,"location":src})
                    backup_sf_list.append({"type": 2, "name":vnf_id,"CPU": cpu, "cache": cache, "in_bw": src_out, "out_bw": dst_in,"latency":0,"original_loc":location,"original_sfc":sfc_id})
                    backup_sf_list.append({"type": 2, "name":dst_name,"CPU": 0, "cache": 0, "in_bw": dst_in, "out_bw":0 ,"latency":0,"location":dst})

                    new_sfc_dict = {}
                    new_sfc_dict["name"] = name
                    new_sfc_dict["vnf_list"] = backup_sf_list
                    new_sfc_dict["bandwidth"] = sfc.input_throughput
                    new_sfc_dict["src_node"] = src
                    new_sfc_dict["dst_node"] = dst

                    duration = sfc_id_duration[sfc_id]["duration"]-(current_time-sfc_id_duration[sfc_id]["timer"])
                    new_sfc_dict["duration"] = duration  + 20 
                    new_sfc_dict["latency"] = latency_req  # sfc.latency_request - self.calculate_latency(sfc_rf) + latency_dismiss  # TODO deve ter aqui  um cálculo para não passar da latencia da original se implementada
                    # new_sfc_dict["original_sfc"] = sfc_rf
                    # new_sfc_dict["restrictions"] = [location]
                    
                    new_sfc = SFCGenerator(new_sfc_dict).generate()
                    
                    #self.sfs_backup[sfc_id] = {"sfc_backup_id":new_sfc.id,"vnf_id":vnf_id}  # Por enquanto teremos apenas uma SFC de Backup por SFC 

                    new_sfc_list.append(new_sfc)
                    backups_mount.append(new_sfc_list)
        return backups_mount
   
    def create_backups(self,nodes_fail_p,network,sfc_manager):
        backups_mount = []
        sfc_id_duration = copy.deepcopy(sfc_manager.sfc_id_duration)

        if self.alg in ['vegeta','ga']:
            backups_mount = self.seletive_strategy(nodes_fail_p,network,sfc_id_duration)
        else:        
            backups_mount = self.greedy_strategy(nodes_fail_p,network,sfc_id_duration)
                    #sfcs_id_backup_made.appenc_id)
        return backups_mount
    
    def calculate_latency(self, route_info):
        total_latency = sum(
            len(path) - 1 for key, path in route_info.items() if path and key not in ["src", "dst"]
        )
        return total_latency
    
    def escolher_src_dst(self,dicionario, vnf_escolhida):
        chaves = list(dicionario.keys())  # Lista das chaves na ordem original
        if vnf_escolhida not in chaves:
            return None, None  # Caso a VNF escolhida não exista no dicionário
        latency_dismiss = 0
        idx = chaves.index(vnf_escolhida)  # Posição da VNF na lista de chaves
        
        if idx == 0:  # Se for a primeira VNF, dst é ela mesma e src é a próxima
            dst = chaves[idx]
            src = chaves[idx + 1]
            latency_dismiss = len(dicionario[src])-1
        elif idx == len(chaves) - 1:  # Se for a última VNF, dst é a anterior e src é ela mesma
            dst = chaves[idx - 1]
            src = chaves[idx]
            latency_dismiss = len(dicionario[vnf_escolhida])-1
        else:  # Caso intermediário, dst é a anterior e src é a próxima
            dst = chaves[idx - 1]
            src = chaves[idx + 1]
            latency_dismiss = (len(dicionario[src])-1) + (len(dicionario[vnf_escolhida])-1)

        src = dicionario[src][0]
        dst = dicionario[dst][0]
        latency_requirement = 7 - latency_dismiss
        return dst, src,latency_requirement