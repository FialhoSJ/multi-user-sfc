import copy
import time
from utils.k_shortest_paths import k_shortest_paths
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

class SFCInstatiator:
    def __init__(self,alg):
        self.alg = alg
        self.sfc_list = []
        self.sfc_queue = []
        self.sfcs_routing_info = {}
        self.sfcs_that_deployed = []
        self.sfcs_that_crashed = []
        self.verbose = True

    def search_solution(self,sfc_list, substrate_network, is_backup=False):
        default_solution_format = {sfc.id: {'route_info': None, 'latency': None, 'run_duration': None} for sfc in sfc_list}

        algorithm = copy.deepcopy(self.alg)
        algorithm.clear_all()
        # O algoritmo deve criar variáveis temporárias e não usar a rede 'oficial'.
        graph =  copy.deepcopy(substrate_network.graph)
        algorithm.install_substrate_network(graph)
        self.add_mobile_user_to_graph(graph,substrate_network,sfc_list)
        
        sequential_sub = True
        if sequential_sub:
            solution,is_success = self.sequential_search(algorithm,sfc_list,graph,default_solution_format)
        else:
            # TODO  Isso pode ser necessário mudar caso o algoritmo não precise instanciar sequencialmente. Ou seja, ele pode instanciar em lotes
            # EX: solution,is_success = self.batch_search(algorithm,sfc_list,substrate_network,default_solution_format)
            pass
        
        if is_success:
            self.deploy_success_message(sfc_list)
        else:
            self.deploy_failed_message(sfc_list)
        return solution,is_success
    
    def sequential_search(self,algorithm,sfc_list: object,graph:object,solution_format) -> None:
        search_success = True
        for sfc in sfc_list:     # TODO  Isso pode ser necessário mudar caso o algoritmo não precise instanciar sequencialmente              
            # algorithm = copy.deepcopy(self.alg)
            # algorithm.clear_all()
            # algorithm.install_substrate_network(substrate_network)
            algorithm.install_SFC(sfc)
            s = time.time()
            alg_success = algorithm.start_algorithm()
            s2 = time.time()
            
            if alg_success:
                try:
                    self.submit_solution(graph,sfc,algorithm.get_route_info())
                except:
                    self.alg.handle_failure()

            solution_format[sfc.id] = {
                'route_info': algorithm.get_route_info(),
                'latency': algorithm.get_latency() ,
                'run_duration': s2 - s
                }
            
            # Validate latency
            if not alg_success:
                search_success = False

        return solution_format,search_success

    def add_mobile_user_to_graph(self,graph,substrate_network,sfc_list):
        mobile_device_id = sfc_list[0].dst_node
        closer_router    = sfc_list[0].closer_router

        # Os recursos do Mobile Device devem estar disponíveis somente para sua SFC
        md_info =  substrate_network.md_graph._node[mobile_device_id]
        graph.add_node(mobile_device_id,type='mobile_device',
                                cpu_capacity=md_info['cpu_capacity'],
                                cache_capacity=md_info['cache_capacity'],
                                cpu_used=md_info['cpu_used'],
                                cache_used=md_info['cache_used'],
                                position=md_info['position'],
                                services=md_info['services'])
        router = graph._node[closer_router]
        wireless_free = router['w_channel_capacity'] - router['w_channel_used']
        
        # TODO Permitir que o próprio algoritmo escolha o roteador
        # TODO calcular a latência do sinal
        signal_latency = 1
        graph.add_edge(mobile_device_id, closer_router, bandwidth_capacity=wireless_free, bandwidth_used=0.00 , latency=signal_latency, services_in_transit={})

    def submit_solution(self,graph,sfc,route_info):
        def allocate_microservice(node_id, service_id, cpu_required, cache_required):
            node = graph.nodes[node_id]

            # Verifica se há recursos disponíveis
            if node['cpu_used'] + cpu_required > node['cpu_capacity']:
                raise ValueError(f"CPU excedida no nó {node_id} para serviço {service_id}")
            if node['cache_used'] + cache_required > node['cpu_capacity']:
                raise ValueError(f"Cache excedido no nó {node_id} para serviço {service_id}")

            if service_id in node['services']:
                node['services'][service_id]['copys'] += 1  # Serviço já instanciado
                if not self.is_shareable(service_id):  # Se não for compartilhável
                    node['cpu_used'] += cpu_required
                    node['cache_used'] += cache_required
            else:
                node['services'][service_id] = {'cpu': cpu_required, 'cache': cache_required, 'copys': 1}
                node['cpu_used'] += cpu_required
                node['cache_used'] += cache_required

        def allocate_bandwidth(node1, node2, bw_required, ms_name):
            edge = graph.edges[node1, node2]

            # Verifica se há banda disponível
            if edge['bandwidth_used'] + bw_required > edge['bandwidth_capacity']:
                raise ValueError(f"Banda excedida entre os nós {node1} e {node2} para serviço {ms_name}")

            if ms_name in edge['services_in_transit']:
                edge['services_in_transit'][ms_name]['copys'] += 1
                edge['bandwidth_used'] += bw_required
            else:
                edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bw_required}
                edge['bandwidth_used'] += bw_required

        for ms_name, path in route_info.items():
            if ms_name in ['src', 'dst']:
                continue

            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            cpu_req = vnf.get_cpu_request()
            cache_req = vnf.get_cache_request()
            allocate_microservice(node_allocated, ms_name, cpu_req, cache_req)

            bw_req = vnf.get_outcome_interface_bandwidth()

            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    allocate_bandwidth(u, v, bw_req, ms_name)

    def is_shareable(self,service_name):
        # TODO Mudar para a informação de compartilháveis estar em uma variável separável.
        #if self.shareable_node:
        if True:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False

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
            
            

