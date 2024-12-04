"""_summary_

"""

import os
import threading
from typing import Tuple, List, Dict

import sys
import re
import time
import logging
import copy
import random
import _thread
from threading import Timer
from queue import Queue
import numpy as np
from controllers.sfc_generator import SFCGenerator

from controllers.modules.mobility_manager import MobilityManager
from controllers.modules.crasher import Crasher
from controllers.modules.resource_manager import ResourceManager

from controllers.sfc_queue import SFCQueue
from utils.k_shortest_paths import k_shortest_paths

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler('./logs/substrate_network_controller.log')
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')

class SubstrateNetworkController():
    def __init__(self, nw):
        # Rede Substrato
        self.substrate_network = nw
        self.edges = []
        self.nodes = None
        self.ec_servers = None
        self.routers = None
        self.node_info = {}
        self.number_of_nodes = None

        # Modules
        self.mobility_manager = None
        self.instantiator = None
        self.crasher_manager = None

        # Status da Rede
        self.remaining_time = None
        self.update_interval = 1
        self.is_stopped = True
        self.verbose = False

        # SFC (Service Function Chains)
        self.sfc_list = []
        self.sfc_queue = None
        self.sfc = None
        self.sfc_id_duration = {}
        self.sfcs_routing_info = {}
        self.sfcs_that_deployed = []

        # Compartilhamento
        self.shareable = False
        self.shareable_band = False  # not being used
        self.shareable_list = []

        # Implementações extras
        self.log_file = "mobilidade_log.txt"
        self.crasher_activate = False
        self.mobility_activated = None
        self.latency_interval = [6, 10]
        self.allow_high_latency = False
        self.altered_sfcs = {}

        # Estatísticas
        self.success = []
        self.deploy_failure = 0

        # Outros
        self.lock = threading.Lock()
        self.players = 0
        self.flows = 0
        self.players_sfc_list = []
        self.alg = None
        self.timer = None
        self.counter = 0
        self.output_writter = False
        self.max_queue_size = 0

    def simulation_timer(self) -> int:
        """Stops the simulation when the time is over."""
        while self.remaining_time > 0:
            time.sleep(1)
            self.remaining_time -= 1

        # stops the simulation when the loop ends.
        self.stop()
        sys.exit()

    def start(self) -> None:
        """Starts the simulation."""

        if not self.is_stopped:
            self.is_stopped = True
            time.sleep(2*self.update_interval)

        self.is_stopped = False
        self.update()
        self.check_sfc_duration()

        # If mobility is activated
        if self.mobility_activated:
            self.mobility_manager.start_simulation()
            self.start_tracer(interval = 5)

        if self.allow_high_latency:
            self.start_check_altered_sfc_thread(interval = 5)
        
        if self.crasher_activate:
            self.start_crasher(interval=25)

        #Run simulation
        _thread.start_new_thread(self.run, ())

    def output_network_resources(self,deploy_time):
        self.output_writter.output_cpu_utilization(self.substrate_network, deploy_time)
        self.output_writter.output_cache_utilization(self.substrate_network, deploy_time)
        self.output_writter.output_bandwidth_utilization(self.substrate_network, deploy_time)
        self.output_writter.output_nodes_sf_utilization(self.substrate_network, deploy_time)
        #self.output_utils.output_edges_sf_utilization(self.edges_vnf, self.existing_vnf, self.sfc_list, deploy_time, route_info, sfc)

    def output_flows(self,current_time, sfc, latency, run_duration, is_success,bw_transcode,latency_diff=None,wait_time=None):
        self.output_writter.output_flows(self.substrate_network,
                                         wait_time,
                                         self.get_running_players_sessions(),
                                         self.counter,
                                         self.remaining_time,
                                         current_time,
                                         sfc,
                                         latency,
                                         run_duration,
                                         is_success,
                                         bw_transcode,
                                         latency_diff)

    def start_check_altered_sfc_thread(self, interval):
        def task():
            while not self.is_stopped:
                self.check_altered_sfcs()
                time.sleep(interval)

        thread_check_alt = threading.Thread(target=task)
        thread_check_alt.daemon = True
        thread_check_alt.start()

    def start_tracer(self, interval=5):
        def task_mob():
            while not self.is_stopped:
                with self.lock:
                    start_time = time.time()  # Marca o tempo de início
                    if len(self.players_sfc_list) != 0:
                        sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes()

                        for sfc_list, new_location in zip(sfcs_moved, new_locations):
                            for sfc_id in sfc_list:
                                try:
                                    sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                                except:
                                    break
                                try:
                                    print(f"SFC {sfc.id} mudou de localização para {new_location}")
                                    self.send_back_to_qeue(sfc, changed_location=True, new_location=new_location)
                                except:
                                    print("Erro na reinstanciação da SFC")

                    end_time = time.time()  # Marca o tempo de fim
                    elapsed_time = end_time - start_time

                    # # Abre o arquivo de log em modo append para não sobrescrever os dados existentes
                    # with open(self.log_file, "a") as log:
                    #     log.write(f"Tempo total de execução: {elapsed_time:.2f} segundos\n")
                    
                    #     time.sleep(interval)  # Intervalo entre as verificações

        # # Verifica se o arquivo já existe, caso contrário, cria um novo
        # if not os.path.exists(self.log_file):
        #     with open(self.log_file, "w") as log:
        #         log.write("Log de Mobilidade\n")
        #         log.write("====================\n")

        # Inicia a thread
        thread_mob = threading.Thread(target=task_mob)
        thread_mob.daemon = True 
        thread_mob.start()

    def start_crasher(self,interval=2):
        def task_crash():
            time.sleep(interval)
            if not self.is_stopped:
                with self.lock:
                    nodes_to_crash = self.crasher_manager.activate_crasher(self.substrate_network)
                    self.crasher_manager.implement_crash(nodes_to_crash,self.substrate_network)
                    
                    self.mobility_manager.set_crashed_servers(nodes_to_crash)
                    sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes()

                    for sfc_list, new_location in zip(sfcs_moved, new_locations):
                        for sfc_id in sfc_list:
                            try:
                                sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                            except:
                                break  # Se não encontrar a SFC, interrompe o loop interno
                            try:
                                print(f"SFC {sfc.id} crashou e mudou de localização para {new_location}")
                                self.send_back_to_qeue(sfc, changed_location=True, new_location=new_location)
                            except:
                                print("Erro na reinstanciação da SFC")

        thread_mob = threading.Thread(target=task_crash)
        thread_mob.daemon = True 
        thread_mob.start()


    def get_nodes_information(self) -> None:
        """Get information for each node in the network."""
    
        for node in self.substrate_network.nodes():
            self.node_info[node] = self.get_node_information(node)

    def get_running_players_sessions(self):
        running_sfcs = self.sfc_id_duration
        players = {}
        sessions = set()

        for key in list(running_sfcs.keys()):
            player = key.split('_')[2][-1]
            session = key.split('_')[-1]
            sessions.add(session)
            if player not in players:
                players[player] = set()
            players[player].add(session)

        num_players = sum(len(sessions) for sessions in players.values())
        num_sessions = len(sessions)
        return len(running_sfcs.keys()), num_players, num_sessions

    def update(self) -> None:
        """Updates the network state and check for resource overhead."""
        self.substrate_network.update()

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
                #alg.set_costs()
                alg.start_algorithm(shareable_sfs=shareable_sfs)
            case _: # algs with passive reuse and no cost method
                alg.start_algorithm() 

        s2 = time.time()

        route_info = alg.get_route_info() # Routes choosen by the alg
        latency = alg.get_latency() # latency of the solution
        bit_rate_adjust =  1.0 if self.alg.name != 'osfem' else alg.get_bit_rate_used()
        bw_transcode =  sfc.vnfs_dict[-1]['out_bw'] if self.alg.name != 'osfem' else alg.get_transcode_bw()

        if (self.alg.name == 'musfico') and (sfc.id in self.sfcs_routing_info.keys()): # musfico exclusive methodology
            latency,route_info = self.musfico_method(sfc)

        is_acceptable,wait_time  = self.check_latency_and_bitrate(sfc, route_info,latency,bit_rate_adjust) # for scenarios where high latency is acceptable
        route_info, latency = self.validate_route_info_and_latency(route_info, latency, sfc, self.sfc,is_acceptable)

        is_success = 0
        current_time = s2
        run_duration = s2 - s
        arrival_time = sfc.arrival_time
        sfc.depart_time = s2
    
        if route_info:
            self.substrate_network.deploy_sfc(sfc, route_info)
            self.sfc_list.append(sfc.id)
            self.sfc_id_duration[sfc.id] = sfc.duration
            self.mobility_manager.add_vehicle(sfc)
            is_success = 1
            self.deploy_success(sfc)
            if sfc not in self.sfcs_routing_info.keys():
                self.sfcs_routing_info[sfc.id] = copy.deepcopy(route_info)
        else:
            self.deploy_failure = 1
            self.mobility_manager.remove_sfc(sfc.id)
            self.deploy_failed(sfc)
            
        self.update()
        is_success = self.check_resources_exceed(is_success,sfc) # Check if any fees exceed %
        self.counter += 1 # at this time all verifications are done. So we add 1 to counter of sfc

        # output of the simulation
        self.output_network_resources(deploy_time=current_time)
        self.output_flows(current_time,sfc,latency,run_duration,is_success,bw_transcode,latency_diff=None,wait_time=wait_time)

        self.success.append(is_success)
        if self.verbose == True:
            print("__________________________________________")
            self.output_writter.print_output_info(self.substrate_network,self.success) #print log information
            print("") 

        if is_success == 1:
            self.sfcs_that_deployed.append(sfc.id)
            return True
        
        return False

    def get_route_info(self) -> Dict:
        """
        Contains information about the SFC being 
        instantiated.

        Returns:
            Dict: A dictionary containing route info
            for the current SFC.
        """
        return self.substrate_network.sfc_route_info

    def undeploy_sfc(self, sfc_id: str) -> None:
        """
        Undeploys an SFC and all associated information
        from the substrate network.

        Args:
            sfc_id (str): The given SFC

        """
        if sfc_id not in self.sfc_list:
            print(sfc_id, "not on the substrate network")
            return -1
        
        try:
            self.substrate_network.undeploy_sfc(sfc_id)
            self.sfc_list.remove(sfc_id)

            # undeploys the sfc.

            #if self.shareable:
            #    self.check_sf_connections()
            del self.sfc_id_duration[sfc_id]

            # TODO fazer a lógica de remover sfs
            # quando o tempo acaba
            self.update()

        except ValueError:
            print(f"Error: SFC ID {sfc_id} not found in the list when trying to remove.")
            return -1


    def check_resources_exceed(self,is_success,sfc):
        if is_success == 1:
            cpu_utilization = round(self.substrate_network.get_cpu_utilization_rate(),4)
            cache_utilization = round(self.substrate_network.get_cache_utilization_rate(),4)
            bw_utilization = round(self.substrate_network.get_bandwidth_utilization_rate(),4)

            if cpu_utilization > 1 or cache_utilization > 1 or bw_utilization > 1: # Check if any fees exceed %
                self.undeploy_sfc(sfc.id)
                is_success = 0
            return is_success
        else:
            return is_success

    def run(self) -> None:
        """Starts the simulation."""
        while not self.is_stopped:
            self.submit_sfcs()

    def get_node_information(self, node_id: int) -> Tuple:
        """
        Return the topology's node attributes.

        Args:
            node_id (int): The choosen node.

        Returns:
            Tuple: A tuple with information about the nodes.
        """
        sfc_vnf_list = self.substrate_network.get_node_sfc_vnf_list(node_id)
        cpu_used = self.substrate_network.get_node_cpu_used(node_id)
        cpu_free = self.substrate_network.get_node_cpu_free(node_id)
        cpu_capacity = self.substrate_network.get_node_cpu_capacity(node_id)
        total_cpu_used = self.substrate_network.total_cpu_used
        total_cpu_capacity = self.substrate_network.total_cpu_capacity
        cache_used = self.substrate_network.get_node_cache_used(node_id)
        cache_free = self.substrate_network.get_node_cache_free(node_id)
        cache_capacity = self.substrate_network.get_node_cache_capacity(node_id)
        total_cache_used = self.substrate_network.total_cache_used
        total_cache_capacity = self.substrate_network.total_cache_capacity

        return (sfc_vnf_list, cpu_used, cpu_free, cpu_capacity, total_cpu_used, \
                total_cpu_capacity, cache_used, cache_free, cache_capacity, \
                    total_cache_used, total_cache_capacity)

    def musfico_method(self, sfc):
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
        shortest_path = k_shortest_paths(self.substrate_network, prev_vnf_node, 
                                        dst_substrate_node, k=1, weight='latency')
        # get the shortest among the k shortest paths
        route_info[previous_vnf.id] = shortest_path[0]
        latency = 0
        for vnf_id in route_info.keys():
            if vnf_id == 'src':
                continue
            path = route_info[vnf_id]
            for i in range(len(path) - 1):
                edge_latency = self.substrate_network.get_link_latency(
                    path[i], path[i + 1])
                latency += edge_latency

        if latency > sfc.get_latency_request() or latency < 0:
            route_info = False
            latency = None

        return latency, route_info


    def check_sfc_duration(self) -> None:
        """
        Implements a counter to remove SFCs whose durations
        are over, using a thread.
        """
        def task():
            while not self.is_stopped:
                time1 = time.time()
                
                remove_list = []
                for sfc_id, duration in list(self.sfc_id_duration.items()):
                    if duration <= 1:
                        remove_list.append(sfc_id)
                        # if duration is over, sfc routing info no longer needed
                        del self.sfcs_routing_info[sfc_id]
                        continue
                    self.sfc_id_duration[sfc_id] = duration - 1

                for sfc_id in remove_list:
                    # player = sfc_id.split("_")[2][1]
                    # p_session = sfc_id.split("_")[3]
                    # user_id = int(player + p_session)
                    self.mobility_manager.remove_sfc(sfc_id)
                    self.undeploy_sfc(sfc_id)

                time2 = time.time()
                elapsed_time = time2 - time1

                # Ajuste o intervalo baseado no tempo de execução da função
                if elapsed_time < self.update_interval:
                    sleep_time = self.update_interval - elapsed_time
                else:
                    sleep_time = 0  # Se o tempo de execução for maior que o intervalo, reexecuta imediatamente

                if not self.is_stopped:
                    time.sleep(sleep_time)  # Aguarda o tempo restante para completar o intervalo

        # Cria e inicia a thread
        thread = threading.Thread(target=task)
        thread.daemon = True  # Faz a thread rodar como daemon para que o programa termine sem bloqueios
        thread.start()

    
    def check_latency_and_bitrate(self, sfc,route_info, latency: int,bitrate :int) -> bool:
        """
        Checks if the latency is within acceptable limits and handles high latency scenarios.
        Checks if Trascoding bitrate was adjusted

        Args:
            sfc (object): The service function chain (SFC) object containing the SF details.
            latency (int): The measured latency for the SFC.
            bitrate (int): The bitrate factor used
        Returns:
            bool: True if the latency and bitrate are acceptable, False otherwise.
        """
        # if not isinstance(latency, int):
        #     if isinstance(latency,float):
        #         pass
        #     else:
        #         return False
            
        if route_info:
            altered_sfc = False
            wait_time = None

            if self.allow_high_latency:
                if latency > self.latency_interval[0] and latency <= self.latency_interval[1]:  #latency bettwen 6 and  ms
                    altered_sfc = True

            # TODO bitrate implementation is not being done in this simulation
            # if bitrate != 1.0:
            #     altered_sfc = True

            if altered_sfc == True:
                if sfc.id not in list(self.altered_sfcs.keys()):
                    self.altered_sfcs[sfc.id] = {'wait_time':0,'timestamp':time.time(),'flag':True} # Primeira tentativa de alocação sem delay alto
                    return True, None 
                
                else:
                    allow_reroute = self.altered_sfcs[sfc.id]['flag']
                    if allow_reroute:
                        return True,None#self.altered_sfcs[sfc.id]['wait_time']
                    else:
                        return False,30
            else:
                # Caso não alterado
                if sfc.id in self.altered_sfcs: # Nesse caso a sfc não está alterada agora, mas antes ela estava, o que quer dizer que ela saiu desse estado
                    wait_time = self.altered_sfcs[sfc.id]['wait_time']
                    self.altered_sfcs[sfc.id]
                    del self.altered_sfcs[sfc.id]
                    return True, wait_time
                else:   
                    return True,None
                
        else:
            if sfc.id in list(self.altered_sfcs.keys()): # Retira
                del self.altered_sfcs[sfc.id]
            return False, None
        
    def validate_route_info_and_latency(self,route_info, latency, sfc, sfc_mode,is_sfc_acceptable):
        # TODO otimizar essas verificações
        """
        Validates the route information and latency.

        Args:
            route_info (dict or bool): The route information to validate.
            latency (int, float, or None): The measured latency to validate.
            sfc (object): The service function chain (SFC) object containing the SF details.
            sfc_mode (str): The mode of the SFC ('on' or 'off').

        Returns:
            tuple: A tuple containing the validated route_info and latency.
        """
        sfc_id = sfc.id
        player = sfc_id.split("_")[2][1]
        p_session = sfc_id.split("_")[3]
        user_id = int(player + p_session)

        if route_info:
            if (len(route_info.keys()) != 6 and sfc_mode == 'on') or (len(route_info.keys()) != 4 and sfc_mode == 'off'):
                latency = None
                route_info = False

        if latency is not None:
            if isinstance(latency, (int, float)):
                if latency > sfc.get_latency_request() or latency < 0:
                    route_info = False
                    latency = None
                
        if is_sfc_acceptable == False: #sfc's denied because is the 2nd time it could not fit
            latency = None
            route_info = False   
        return route_info, latency
    
    def check_altered_sfcs(self) -> None:
        altered_sfcs = list(self.altered_sfcs.items())  # Create a list copy of the items

        for sfc_id, value in altered_sfcs:
            try:
                sfc = self.substrate_network.get_sfc_by_id(sfc_id)
            except:
                continue
            time_out = 30 # configurável
            tempo_primeira_inicializacao = value['timestamp']
            tempo_agora = time.time()
            tempo_corrido = int(tempo_agora - tempo_primeira_inicializacao)
            self.altered_sfcs[sfc_id]['wait_time'] = tempo_corrido # Tempo de espera aumenta

            if tempo_corrido >= time_out:
                self.altered_sfcs[sfc_id]['flag'] = False
                self.send_back_to_qeue(sfc)

    def send_back_to_qeue(self,sfc,changed_location=False,new_location=False):
        sfc_id = sfc.id
        new_sfc_list = []
        location = sfc.dst.substrate_node if new_location == False else new_location
    
        duration = self.sfc_id_duration[sfc_id]

        self.undeploy_sfc(sfc_id)
        new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)

        if changed_location == True:
            if re.search('cache', sfc_id) is not None:
                old_loc = str(sfc.dst.substrate_node)
                new_loc = str(location)

                ma_old_key = 'MA_region_' + old_loc
                re_old_key = 'RE_region_' + old_loc

                if ma_old_key in self.sfcs_routing_info[sfc_id].keys():
                    stored_info = copy.deepcopy(
                        self.sfcs_routing_info[sfc_id][ma_old_key])
                    
                    del self.sfcs_routing_info[sfc_id][ma_old_key]

                    ma_new_key = re.sub(old_loc, new_loc,ma_old_key)

                    self.sfcs_routing_info[sfc_id][ma_new_key] = stored_info
                    new_vnfs_list_dict[1]['name'] = ma_new_key

                if re_old_key in self.sfcs_routing_info[sfc.id].keys():
                    stored_info = copy.deepcopy(
                        self.sfcs_routing_info[sfc_id][re_old_key])

                    del self.sfcs_routing_info[sfc_id][re_old_key]

                    re_new_key = re.sub(old_loc, new_loc,re_old_key)

                    self.sfcs_routing_info[sfc_id][re_new_key] = stored_info
                    new_vnfs_list_dict[2]['name'] = re_new_key

        new_sfc_dict = {}
        new_sfc_dict["name"] = sfc_id
        new_sfc_dict["vnf_list"] = new_vnfs_list_dict
        new_sfc_dict["bandwidth"] = sfc.input_throughput
        new_sfc_dict["src_node"] = sfc.src.substrate_node
        new_sfc_dict["dst_node"] = location
        new_sfc_dict["duration"] = duration
        new_sfc_dict["latency"] = sfc.latency_request

        new_sfc = SFCGenerator(new_sfc_dict).generate()
        new_sfc_list.append(new_sfc)
        self.sfc_queue.put_begin(new_sfc_list)
        #except Exception as e:
        #    print(f"Erro na tentativa de mandar de volta pra fila: {e}")

    def stop(self) -> None:
        """Stops the simulation."""
        self.is_stopped = True
        self.mobility_manager.stop_simulation()
        if self.timer:
            self.timer.cancel()

    def get_shareable_sfs(self, sfc) -> List[Tuple]:
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

    def deploy_success(self, sfc: object) -> None:
        """Print success message."""
        if self.verbose == True:
            print("deploy succeed, sfc: ", sfc.id)

    def deploy_failed(self, sfc: object) -> None:
        """Print failure message."""
        if self.verbose == True:
            print(" deploy FAILED, sfc: ", sfc.id)

    def submit_sfcs(self) -> None:
        """Submit the SFCs stored in the queue."""
        last_sf_mono = 'sfc_unique_p4_' + str(self.flows)
        last_sf_dec = 'sfc_mono_p4_' + str(self.flows)

        
        while not self.is_stopped:
            sfc_list = self.sfc_queue.peek_sfc()           
            print("queue_size: " + str(self.sfc_queue.qsize()))
            if self.max_queue_size < self.sfc_queue.qsize():
                self.max_queue_size = self.sfc_queue.qsize()
            
            player_sfc_id_list = []
            
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
                    self.stop()
                    sys.exit()
            self.players_sfc_list.append(player_sfc_id_list)

