import copy
import logging
import time
import math
import random
import traceback
from typing import List
from utils.k_shortest_paths import k_shortest_paths
from algorithms.networkUtils import calculate_computational_latency,calculate_latency_betwen_nodes
from utils.network_utils import calculate_average_sfc_latency
from utils.salvar_var import salvar_lista, dividir_em_n_grupos
from core.net_v2 import Net2
from algorithms.kuririn import Kuririn
from core.sfc import SFC
from algorithms.darsppo import DARSPPO
from algorithms.hephaestus import hephaestus
from algorithms.environments.environment import SFC_AllocationEnv
from algorithms.environments.env_da_rsppo import SFC_AllocationEnv_DARSPPO
from algorithms.environments.hephaestus_env import SFC_AllocationEnv_hephaestus


SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

class SFCInstatiator:
    def __init__(self,alg):
        self.alg = alg
        self.sfc_list = []
        self.sfc_queue = []
        self.sfcs_routing_info = {}
        self.sfcs_that_deployed = []
        self.session_id = None
        self.current_graph = None
        self.sfcs_that_crashed = []
        self.verbose = True
        self.env = None
        self.list_graph = []
        self.lists_sfcs = []

    def search_solution(self,sfc_list, substrate_network: Net2, is_backup=False):
        default_solution_format = {sfc.id: {'route_info': None, 'latency': None, 'run_duration': None} for sfc in sfc_list}

        # CORREÇÃO 1: Usamos a instância original do algoritmo, sem deepcopy.
        # algorithm = copy.deepcopy(self.alg)
        algorithm = self.alg
        algorithm.clear_all()
        
        graph =  copy.deepcopy(substrate_network.graph)
        self.add_mobile_user_to_graph(graph,substrate_network,sfc_list)


        if isinstance(algorithm, Kuririn):
            valid_nodes = [node for node in graph.nodes() if graph.nodes[node]['type'] != 'router']
            
            # CORREÇÃO 2: O ambiente é inicializado com uma lista de apenas uma SFC.
            self.env = SFC_AllocationEnv(
                valid_nodes=valid_nodes,
                list_graph=[graph],
                list_sfc=[sfc_list[0]],
                is_training=False
            )
        elif isinstance(algorithm, DARSPPO):
            valid_nodes = [node for node in graph.nodes() if graph.nodes[node]['type'] != 'router']
            self.env = SFC_AllocationEnv_DARSPPO(
                valid_nodes=valid_nodes,
                list_graph=[graph],
                list_sfc=[sfc_list[0]],
                is_training=False
            )

        elif isinstance(algorithm, hephaestus):
            valid_nodes = [node for node in graph.nodes() if graph.nodes[node]['type'] != 'router']
            self.env = SFC_AllocationEnv_hephaestus(
                valid_nodes=valid_nodes,
                list_graph=[graph],
                list_sfc=[sfc_list[0]],
                is_training=False
            )
            
        
        sequential_sub = True
        if sequential_sub:
            solution,is_success = self.sequential_search(algorithm,sfc_list,graph,default_solution_format)
        else:
            pass
        
        if is_success:
            self.deploy_success_message(sfc_list)
        else:
            self.deploy_failed_message(sfc_list)
        return solution,is_success
    

    def sequential_search(self,algorithm,sfc_list: List[SFC],graph:object,solution_format, graph_backup = None) -> None:
        search_success = True

        for sfc in sfc_list:
            
            # Prepara o algoritmo com uma CÓPIA da rede e a SFC atual
            algorithm.install_substrate_network(copy.deepcopy(graph))
            algorithm.install_SFC(sfc)
            
            s = time.time()


            # self.list_graph.append(copy.deepcopy(graph))
            # self.lists_sfcs.append(copy.deepcopy(sfc))

            # if "unique_p6_50" in sfc.id:
            #     lista_listas_grafos = dividir_em_n_grupos(self.list_graph, 5)
            #     lista_listas_sfcs = dividir_em_n_grupos(self.lists_sfcs, 5)
            #     for i in range(1,6):
            #         lista_grafo = lista_listas_grafos[i-1]
            #         lista_sfc = lista_listas_sfcs[i-1]
            #         salvar_lista(lista_grafo, f"list_graph{i}")
            #         salvar_lista(lista_sfc, f"list_sfc{i}")



            # if int(sfc.id.split('_')[-1]) <=10:
            #     salvar_variavel(sfc, "list_sfc1")
            #     salvar_variavel(graph, "list_graph1")
            # elif int(sfc.id.split('_')[-1]) <=20:
            #     salvar_variavel(sfc, "list_sfc2")
            #     salvar_variavel(graph, "list_graph2")    
            # elif int(sfc.id.split('_')[-1]) <=30:
            #     salvar_variavel(sfc, "list_sfc3")
            #     salvar_variavel(graph, "list_graph3")  

            # elif int(sfc.id.split('_')[-1]) <=40:
            #     salvar_variavel(sfc, "list_sfc4")
            #     salvar_variavel(graph, "list_graph4")  
            # else:
            #     salvar_variavel(sfc, "list_sfc5")
            #     salvar_variavel(graph, "list_graph5") 
            
            # A chamada ao algoritmo agora depende do seu tipo.
            alg_success = False
            if isinstance(algorithm, Kuririn):
                # Se for o Kuririn, passamos o ambiente que criamos.
                if self.env:
                    alg_success = algorithm.start_algorithm(self.env)
                else:
                    logging.error("Tentativa de usar Kuririn sem um ambiente inicializado.")
                    alg_success = False

            elif isinstance(algorithm, DARSPPO):
                 # Se for o DARSPPO, passamos o ambiente que criamos.
                if self.env:
                    alg_success = algorithm.start_algorithm(self.env)
                else:
                    logging.error("Tentativa de usar DARSPPO sem um ambiente inicializado.")
                    alg_success = False
            
            elif isinstance(algorithm, hephaestus):
                 # Se for o DARSPPO, passamos o ambiente que criamos.
                if self.env:
                    alg_success = algorithm.start_algorithm(self.env)
                else:
                    logging.error("Tentativa de usar DARSPPO sem um ambiente inicializado.")
                    alg_success = False
            else:
                # Para qualquer outro algoritmo, usamos a chamada original.
                alg_success = algorithm.start_algorithm()

            s2 = time.time()
            print(f"Algorithm {self.alg.name} Take time     :   {round((s2-s)*1000,3)} ms")
            
            total_latency = None
            comp_latency = None
            comm_latency = None

            if alg_success:
                try:
                    # Alocação padrão do algoritmo
                    route_info = algorithm.get_route_info()
                    total_latency, comp_latency, comm_latency = self.submit_solution(graph, sfc, route_info)
                    # Se submit_solution for bem-sucedido, ele modifica 'graph'
                except ValueError as ve:
                    logging.error(f"Falha na submissão da solução para SFC {sfc.id}: {ve}")
                    algorithm.handle_failure() 
                    alg_success = False # Marca como falha para o fallback
                except Exception as e:
                    logging.error(f"Erro inesperado ao submeter solução para SFC {sfc.id}: {e}")
                    logging.error(traceback.format_exc())
                    algorithm.handle_failure()
                    alg_success = False # Marca como falha para o fallback
            
            # --- INÍCIO DA LÓGICA DE FALLBACK (DRY RUN) ---
            if not alg_success:
                logging.warn(f"SFC {sfc.id} falhou na alocação. Calculando fallback (dry run) para node 0.")
                
                # Criamos um grafo temporário APENAS para o cálculo.
                # O 'graph' original não será modificado.
                fallback_graph = copy.deepcopy(graph)
                
                try:
                    fallback_node_id = 0
                    fallback_route_info = {}
                    
                    # --- LINHA CORRIGIDA ---
                    # Itera sobre sfc.vnfs_dict como uma LISTA de dicionários
                    vnf_names = [vnf['name'] for vnf in sfc.vnfs_dict]
                    # --- FIM DA CORREÇÃO ---

                    dst_node = sfc.dst_node

                    # 2. Define o 'path' para cada VNF
                    for i in range(len(vnf_names) - 1):
                        fallback_route_info[vnf_names[i]] = [fallback_node_id]
                    
                    last_vnf_name = vnf_names[-1]
                    path_list = k_shortest_paths(fallback_graph, fallback_node_id, dst_node, k=1, weight='latency')
                    
                    if not path_list:
                        logging.error(f"Cálculo de fallback para {sfc.id} falhou: Não há caminho do node 0 para {dst_node}")
                        algorithm.route_info = None 
                    else:
                        fallback_route_info[last_vnf_name] = path_list[0]
                        
                        # 3. Tenta submeter a solução no 'fallback_graph' (temporário)
                        # Este cálculo não afeta o 'graph' principal do loop
                        total_latency, comp_latency, comm_latency = self.submit_solution(fallback_graph, sfc, fallback_route_info)
                        
                        logging.info(f"Cálculo de fallback para {sfc.id} BEM SUCEDIDO (Latência: {total_latency}).")
                        
                        # 4. Salva a rota de fallback para o log
                        algorithm.route_info = fallback_route_info 

                except ValueError as ve:
                    logging.error(f"Cálculo de fallback para {sfc.id} FALHOU (Ex: node 0 sem recursos): {ve}")
                    algorithm.route_info = None
                    # Latências permanecem None
                except Exception as e:
                    logging.error(f"Erro inesperado no cálculo de fallback para {sfc.id}: {e}")
                    logging.error(traceback.format_exc()) # Adiciona o traceback para depuração
                    algorithm.route_info = None
                    # Latências permanecem None

                # CRUCIAL: 'alg_success' permanece False, 
                # então 'search_success' será definido como False abaixo.

            # --- FIM DA LÓGICA DE FALLBACK ---

            solution_format[sfc.id] = {
                'route_info': algorithm.get_route_info(), # Pega o route_info (original, falho, ou de fallback)
                'latency': total_latency,      # Pega a latência (calculada, de fallback, ou None)
                'comp_latency': comp_latency, 
                'comm_latency': comm_latency,
                'run_duration': s2 - s
                }
            
            if not alg_success:
                search_success = False # Garante que a falha seja propagada

        return solution_format,search_success

    def add_mobile_user_to_graph(self,graph,substrate_network,sfc_list):
        mobile_device_id = sfc_list[0].dst_node
        closer_router    = sfc_list[0].closer_router

        # Os recursos do Mobile Device devem estar disponíveis somente para sua SFC
        md_info =  copy.deepcopy(substrate_network.md_graph._node[mobile_device_id])
        # md_info["user_session"] = mobile_device_id
        graph.add_node(mobile_device_id,type='mobile_device',
                                cpu_capacity=md_info['cpu_capacity'],
                                cache_capacity=md_info['cache_capacity'],
                                cpu_used=md_info['cpu_used'],
                                cache_used=md_info['cache_used'],
                                position=md_info['position'],
                                services=md_info['services'],
                                ips=md_info['ips'],
                                reuse=md_info['reuse'])
        router = graph._node[closer_router]
        wireless_free = router['w_channel_capacity'] - router['w_channel_used']
        
        # TODO Permitir que o próprio algoritmo escolha o roteador
        # TODO calcular a latência do sinal
        signal_latency = 1
        graph.add_edge(mobile_device_id, closer_router, bandwidth_capacity=wireless_free, bandwidth_used=0.00 , latency=signal_latency, services_in_transit={})

    def submit_solution(self,graph,sfc,route_info):
        def allocate_microservice(vnf, node_id, session_id):
            service_id = vnf.id
            service_key = (service_id, session_id)
            cpu_required = vnf.get_cpu_request()
            cache_required = vnf.get_cache_request()
            node = graph.nodes[node_id]
            latency = calculate_computational_latency(graph, node_id, vnf)
            
            if  node_id == 0:
                return 0

            if node['type'] not in ['server', 'mobile_device']:
                raise ValueError(f"Serviços só podem ser alocados em servidores ou usuários, não em '{node['type']}'.")

            # --- LÓGICA CORRIGIDA ---

            # 1. Primeiro, verifica se o serviço já existe no nó
            if service_key in node['services']:
                node['services'][service_key]['copys'] += 1  # Apenas incrementa o contador de cópias

                # 2. Se não for compartilhável, aí sim verifica e consome novos recursos
                if not self.is_shareable(service_id):
                    # Verifica se há recursos para esta instância não compartilhável
                    if node['cpu_used'] + cpu_required > node['cpu_capacity']:
                        # Reverte o incremento de cópias antes de falhar
                        node['services'][service_key]['copys'] -= 1
                        raise ValueError(f"CPU excedida no nó {node_id} para instância não compartilhável do serviço {service_id}")
                    if node['cache_used'] + cache_required > node['cache_capacity']:
                        # Reverte o incremento de cópias antes de falhar
                        node['services'][service_key]['copys'] -= 1
                        raise ValueError(f"Cache excedido no nó {node_id} para instância não compartilhável do serviço {service_id}")

                    # Se passou, consome os recursos
                    node['cpu_used'] += cpu_required
                    node['cache_used'] += cache_required
                
                # Se for compartilhável (o 'else' do 'if not self.is_shareable'), não faz nada, pois o recurso está sendo reusado.

            # 3. Se o serviço for completamente novo no nó
            else:
                # Verifica se há recursos para a primeira instância do serviço
                if node['cpu_used'] + cpu_required > node['cpu_capacity']:
                    raise ValueError(f"CPU excedida no nó {node_id} para alocar novo serviço {service_id}")
                if node['cache_used'] + cache_required > node['cache_capacity']:
                    raise ValueError(f"Cache excedido no nó {node_id} para alocar novo serviço {service_id}")

                # Se passou, aloca o serviço e consome os recursos
                node['services'][service_key] = {'cpu': cpu_required, 'cache': cache_required, 'copys': 1}
                node['cpu_used'] += cpu_required
                node['cache_used'] += cache_required

                # Adiciona à lista de reuso se for compartilhável
                if self.is_shareable(service_id):
                    node['reuse'].append(vnf)
                    
            return latency
        
        def allocate_bandwidth(node1, node2, vnf, ms_name):
            bw_required = vnf.get_outcome_interface_bandwidth() 
            latency = calculate_latency_betwen_nodes(graph,node1,node2,vnf) 
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
            return latency
        
        session = sfc.id.split("_")[-1]
        total_latency = 0
        
        # Debugger
        tsaber = {
        'computacao': {},   # latência computacional por microsserviço (vnf)
        'comunicacao': {}   # latência de comunicação por enlace (u->v)
        }
        
        for ms_name, path in route_info.items():
            if ms_name in ['src', 'dst']:
                continue
        
            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]
            comp_latency = allocate_microservice(vnf, node_allocated,session)
            total_latency += comp_latency

            tsaber['computacao'][ms_name] = {
                'node': node_allocated,
                'latencia_comp': comp_latency
            }
            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    comm_latency = allocate_bandwidth(u, v, vnf, ms_name)
                    total_latency += comm_latency

                    if ms_name not in tsaber['comunicacao']:
                        tsaber['comunicacao'][ms_name] = []
                    tsaber['comunicacao'][ms_name].append({
                        'de': u,
                        'para': v,
                        'latencia_comm': comm_latency
                    })
        # if total_latency> 50:
        #     print(total_latency)
        
        total_comp_latency = sum(d['latencia_comp'] for d in tsaber['computacao'].values())
        total_comm_latency = sum(item['latencia_comm'] 
                                 for items in tsaber['comunicacao'].values() 
                                 for item in items)
        
        return round(total_latency, 2), round(total_comp_latency, 2), round(total_comm_latency, 2)
    
    def is_shareable(self,service_name):
        # TODO Mudar para a informação de compartilháveis estar em uma variável separável.
        #if self.shareable_node:
        if True:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False
        
    def calcular_latencia_5g(
        self,
        data,
        distancia_m=750,
        potencia_transmissao_dbm=20.0,
        largura_banda_hz=50e6,
        temperatura_kelvin=290,
        figura_ruido_db=10.0,
        eficiencia_codec=0.5,
        snr_minimo_db=0.0,
        freq_portadora_hz=3.5e9,
        sigma_shadowing_db=6.00, #8.00
    ):
        """
        Calcula latência (ms) para uma dada distância em 5G, considerando path loss com shadowing.

        Parâmetro:
        - distancia_m: distância em metros (float ou lista/tupla de floats)

        Retorna latência em ms (float ou lista de floats, conforme input)
        """
        BOLTZMANN = 1.380649e-23

        # Função de perda de caminho com shadowing
        def path_loss_5g(distancia_m):
            pl_db = 28.0 + 22 * math.log10(distancia_m) + 20 * math.log10(freq_portadora_hz / 1e9)
            pl_db += random.gauss(0, sigma_shadowing_db)
            return 10 ** (-pl_db / 10)  # ganho linear

        def calcular_latencia_um_ponto(dado):
            ganho = path_loss_5g(distancia_m)
            potencia_w = 10 ** (potencia_transmissao_dbm / 10) / 1000
            ruido_w_hz = BOLTZMANN * temperatura_kelvin * (10 ** (figura_ruido_db / 10))
            snr_linear = (ganho * potencia_w) / (ruido_w_hz * largura_banda_hz)
            snr_linear = max(snr_linear, 10 ** (snr_minimo_db / 10))
            taxa_bps = largura_banda_hz * math.log2(1 + snr_linear) * eficiencia_codec
            latencia_ms = (dado / taxa_bps) * 1000  
            return latencia_ms
        return calcular_latencia_um_ponto(data)

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
s
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
            
            

