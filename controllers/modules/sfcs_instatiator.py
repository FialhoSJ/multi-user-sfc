import copy
import time
from utils.k_shortest_paths import k_shortest_paths

class SFCInstatiator:
    def __init__(self,alg):
        self.alg = alg
        self.sfc_list = []
        self.sfc_queue = []
        self.sfcs_routing_info = {}
        self.sfcs_that_deployed = []
        self.sfcs_that_crashed = []
        self.verbose = True

    def search_solution(self,sfc_list,substrate_network,is_backup=False):
        default_solution_format = {sfc.id: {'route_info': None, 'latency': None, 'run_duration': None} for sfc in sfc_list}

        algorithm = copy.deepcopy(self.alg)
        algorithm.clear_all()
        # O algoritmo deve criar variáveis temporárias e não usar a rede 'oficial'.
        algorithm.install_substrate_network(substrate_network)
        
        sequential_sub = True
        if sequential_sub:
            solution,is_success = self.sequential_search(algorithm,sfc_list,substrate_network,default_solution_format)
        else:
            # TODO  Isso pode ser necessário mudar caso o algoritmo não precise instanciar sequencialmente. Ou seja, ele pode instanciar em lotes
            # EX: solution,is_success = self.batch_search(algorithm,sfc_list,substrate_network,default_solution_format)
            pass
        
        if is_success:
            self.deploy_success_message(sfc_list)
        else:
            self.deploy_failed_message(sfc_list)
        return solution,is_success
    
    def sequential_search(self,algorithm,sfc_list: object,substrate_network:object,solution_format) -> None:
        is_success = True
        for sfc in sfc_list:     # TODO  Isso pode ser necessário mudar caso o algoritmo não precise instanciar sequencialmente              
            algorithm.install_SFC(sfc)

            s = time.time()
            is_success = algorithm.start_algorithm()
            s2 = time.time()

            # Validate latency
            if not is_success:
                break

            solution_format[sfc.id] = {
                'route_info': algorithm.get_route_info(),
                'latency': algorithm.get_latency() ,
                'run_duration': s2 - s
            }
        return solution_format,is_success

    def deploy_success_message(self, sfc_list: object) -> None:
        """Print success message."""
        if self.verbose == True:
            for sfc in sfc_list:
                print("deploy succeed, sfc: ", sfc.id)

    def deploy_failed_message(self, sfc_list: object) -> None:
        """Print failure message."""
        if self.verbose == True:
            for sfc in sfc_list:
                print(" deploy FAILED, sfc: ", sfc.id)


###########################################################################################
    # TODO IMPORTANTE PARA RODAR MUSFICO !!!
    # if sfc.id in list(self.sfcs_routing_info.keys()):
    #     del self.sfcs_routing_info[sfc.id]
    # # Adiciona a SFC com o novo route_info -> Importante para o musfico
    # self.sfcs_routing_info[sfc.id] = copy.deepcopy(route_info)
###########################################################################################

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

            # is_success = False
            # current_time = s2
            # run_duration = s2 - s
            # if  alg.name == 'ga':
            #     run_duration =alg.elapsed_time

            # arrival_time = sfc.arrival_time
            # sfc.depart_time = s2

            # if route_info:
            #     substrate_network.deploy_sfc(sfc, route_info)
            #     if not is_backup: # Se for uma SFC de Backup apenas coloque na lista de sfcs com backup
            #         self.sfc_list.append(sfc.id)
            #         self.sfc_id_duration[sfc.id] = {"duration":sfc.duration,"timer":time.time()}
            #         if sfc.id in list(self.sfcs_routing_info.keys()):
            #             del self.sfcs_routing_info[sfc.id]
            #         # Adiciona a SFC com o novo route_info -> Importante para o musfico
            #         self.sfcs_routing_info[sfc.id] = copy.deepcopy(route_info)
            #     is_success = True

            # substrate_network.update()
            # self.counter += 1 # at this time all verifications are done. So we add 1 to counter of sfc
            # is_success,fail_reason = self.check_resources_exceed(is_success,sfc.id,substrate_network,fail_reason) # Check if any fees exceed %

            # if is_success == False:
            #     self.deploy_failed(sfc)
            #     if sfc.id in list(self.sfc_reuse.keys()):
            #         r_info = None
            #         del self.sfc_reuse[sfc.id]
            # else:
            #     self.deploy_success(sfc)
                
            # return {"current_time":current_time,"latency":latency,"run_duration":run_duration,"resource_info":r_info,"is_success":is_success,"route_info":route_info,"fail_reason":fail_reason,"backup_sfc":is_backup}
            
            

