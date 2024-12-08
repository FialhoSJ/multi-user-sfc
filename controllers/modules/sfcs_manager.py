import sys
import time
import re
import copy

class SFCManager:
    def __init__(self):
        self.players = 4
        self.flows = 50
        self.player_sfc_id_list = {}
        self.sfcs_that_deployed = []
        self.sfcs_that_crashed = []
        self.backup_sfs = {}
        self.last_sfc_release = False
        self.risk_servers = []
    # def create_sfc(self, sfc_id, service_functions, user_id):
    #     sfc = {
    #         "id": sfc_id,
    #         "functions": service_functions,
    #         "user": user_id,
    #         "status": "pending"
    #     }
    #     self.sfc_queue.append(sfc)

    def submit_flows(self, sfc_list):
        last_sf_mono = 'sfc_unique_p4_' + str(self.flows)
        last_sf_dec = 'sfc_mono_p4_' + str(self.flows)

        player_sfc_id_list = []
        result = self.sfc_instatiator.submit_flows(sfc_list)

        for sfc in sfc_list:
            t_1 = time.time()
            self.deploy_sfc(sfc)
            player_sfc_id_list.append(sfc.id)
            self.update()
            t_2 = time.time()
            print("          algorithm take time: ", round(t_2 - t_1,4))
            if sfc.id in (last_sf_mono, last_sf_dec):
                print('Last SFC released')
                print('Max queue size:', self.max_queue_size )
                self.last_sfc_release = True

        self.players_sfc_list.append(player_sfc_id_list)
        
    def deploy_sfc(self, sfc: object) -> bool:
        """
        Deploys an SFC in the substrate network.

        Currently, this function is coupled with the
        decision function. Hence, not only it does
        deploy an SFC but also computes the route
        for each SF.

        Args:
            sfc (object): A given SFC with SFs.

        Returns:
            bool: Whether the instantiated occured
            succesfully or not.
        """
        
        if sfc.id in self.sfc_list:
            return
        
        alg = copy.deepcopy(self.alg)
        alg.clear_all()
        alg.install_substrate_network(self.substrate_network)
        alg.install_SFC(sfc)
        
        s = time.time() # Start measuring how long it takes to the alg run

        shareable_sfs = self.substrate_network.get_shareable_sfs()

        match self.alg.name:
            case 'ga' | 'osfem' | 'goku': # algs with active reuse and cost method
                alg.set_costs()
                alg.start_algorithm(shareable_sfs=shareable_sfs)
            case _: # algs with passive reuse and no cost method
                alg.start_algorithm() 

        s2 = time.time()

        route_info = alg.get_route_info() # Routes choosen by the alg
        latency = alg.get_latency() # latency of the solution
        
        # bit_rate_adjust =  1.0 if self.alg.name != 'osfem' else alg.get_bit_rate_used()
        # bw_transcode =  sfc.vnfs_dict[-1]['out_bw'] if self.alg.name != 'osfem' else alg.get_transcode_bw()

        if (self.alg.name == 'musfico') and (sfc.id in self.sfcs_routing_info.keys()): # musfico exclusive methodology
            latency,route_info = self.musfico_method(sfc)

        #is_acceptable,wait_time  = self.check_latency_and_bitrate(sfc, route_info,latency)
        route_info, latency = self.validate_solution(route_info, latency, sfc, self.sfc)

        is_success = 0
        current_time = s2
        run_duration = s2 - s
        arrival_time = sfc.arrival_time
        sfc.depart_time = s2
    
        if route_info:
            self.substrate_network.deploy_sfc(sfc, route_info)
            self.sfc_list.append(sfc.id)
            self.sfc_id_duration[sfc.id] = sfc.duration
            is_success = True
            self.deploy_success(sfc)
            if sfc not in self.sfcs_routing_info.keys():
                self.sfcs_routing_info[sfc.id] = copy.deepcopy(route_info)
        else:
            self.deploy_failure = 1
            self.deploy_failed(sfc)
            
        self.update()
        is_success = self.check_resources_exceed(is_success,sfc) # Check if any fees exceed %
        self.counter += 1 # at this time all verifications are done. So we add 1 to counter of sfc

        # output of the simulation
        self.output_network_resources(deploy_time=current_time)
        self.output_flows(current_time,sfc,latency,run_duration,is_success,latency_diff=None,wait_time=None)

        self.success.append(is_success)
        if self.verbose == True:
            print("__________________________________________")
            self.output_writter.print_output_info(self.substrate_network,self.success) #print log information
            print("") 

        if is_success == 1:
            self.mobility_manager.add_vehicle(sfc)
            self.sfcs_that_deployed.append(sfc.id)
            return True
        else:
            self.mobility_manager.remove_sfc(sfc.id)
            return False

    def set_risk_sfcs(self,servers,network):
        if len(servers) == 0:
            return
        #self.risk_servers = servers
        #server_infos = []
        for server in servers:
            comecou = time.time()
            server_info = network.get_node_sfc_vnf_list(server)
            sfcs_l = []
            server_infos = []
            if server_info != []:
                for info in server_info:
                    server_infos.append(info)
                    # TODO acho que dava pra ter mais regras se cri ou não o backup (tempo que ainda resta para aquela SF)
                    sfc_id = info[0]
                    vnf = info[1]
                    vnf_id = vnf.id
                    if sfc_id not in network.sfc_dict:
                        continue    
                    
                    sfc = network.get_sfc_by_id(sfc_id)
                    vnf_info = sfc.vnfs_dict
                    resources_info = [info for info in vnf_info if info['name'] == vnf_id][0]
                    sfc_rf = network.sfc_route_info[sfc_id]
                    
                    src = None
                    location = None
                    dst = None
                    
                    src_in = 0 
                    src_out = resources_info['in_bw']
                    
                    dst_in = resources_info['out_bw']
                    dst_out = 0

                    cpu = resources_info['CPU']
                    cache = resources_info['cache']

                    i = 0
                    for key,value in sfc_rf.items():        
                        if i == 1:
                            src = value[0]
                            break
                        if key == vnf_id:
                            location = value[0]
                            dst = value[-1]
                            i = i + 1
                    

                    self.backup_sfs[info[0]] = {'sf':vnf,
                                                'src': {'location': src,'cpu':0  ,'cache':0    ,'in_bw':src_in ,'out_bw':src_out},
                                                'vnf': {'location': 0  ,'cpu':cpu,'cache':cache,'in_bw':src_out,'out_bw':dst_in},
                                                'dst': {'location':dst ,'cpu':0  ,'cache':0    ,'in_bw':dst_in ,'out_bw':dst_out},
                                                'restrictions':location
                                                }  
                    print()
    # def create_backup_sfc(self):
        
    #     return sfcs_list
    def get_sfc_by_id(self, sfc_id):
        for sfc in self.sfc_list + self.sfc_queue:
            if sfc["id"] == sfc_id:
                return sfc
        return None


    # def find_predecessor(self,dictionary, key):
    #     keys = list(dictionary.keys())  # Lista das chaves na ordem
    #     if key in keys:
    #         key_index = keys.index(key)
    #         if key_index > 0:  # Se não for a primeira chave
    #             return keys[key_index + 1]
    #     return None  # Caso não exista um antecessor
