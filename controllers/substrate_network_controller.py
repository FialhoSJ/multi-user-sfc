"""_summary_

"""

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
from controllers.sfc_queue import SFCQueue
from core.poisson_emitter import PoissonEmitter
from utils.k_shortest_paths import k_shortest_paths
import traceback
import math

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
        self.altered_sfcs = {}
        
        self.latency_interval = [6,10]
        self.allow_temporary_high_latency = False

        self.verbose= False
        self.shareable = False
        self.nodes = None
        self.edges = []
        self.tracer = 0
        self.crasher = 0
        self.mobility_activated = None
        self.crasher_activate = 1
        self.users_crashed = 0 
        self.servers_to_crash = []

        self.output_writter = 0
        
        self.substrate_network = nw
        self.node_info = {}
        self.sfc_list = []
        self.sfcs_routing_info = {}
        self.is_stopped = True
        self.update_interval = 1
        self.number_of_nodes = None
        self.cpu_threshold = 0.8
        self.over_threshold_nodes_list = []
        self.timer = None
        self.remaining_time = None
        self.sfc_queue = None
        self.sfc = None
        self.network_status = "offline"
        
        self.sfc_id_duration = {}
        
        self.edges_vnf = {}
        # self.existing_vnf = {}

        self.deploy_failure = 0
        self.flows = 0
        self.players = 0
        self.counter = 0
        self.alg = None
        self.alg_name = None
        self.players_sfc_list = []
        self.sfcs_crashed = {}
        self.sfcs_back_to_qeue_latency_diff = {}
        self.crash_moment = 0
        self.shareable_list = []

        self.sfcs_that_deployed = []
        self.max_queue_size = 0
        self.success = []
        self.crash_trials = 0
        #self.lock = threading.Lock()  # Inicializa o lock


        self.shareable_band = False # not being used
        #self.fail_resources = (1/3, 1/3, 1/3)
        self.costs_parameters = [1,1,1,1]

    def simulation_timer(self) -> int:
        """Stops the simulation when the time is over."""
        while self.remaining_time > 0:
            time.sleep(1)
            self.remaining_time -= 1

        # stops the simulation when the loop ends.
        self.stop()
        sys.exit()

    def start_tracer_thread(self):
        self.tracer.start_sumo_simulation()
        self.check_user_position_thread()

    def start(self) -> None:
        """Starts the simulation."""

        if not self.is_stopped:
            self.is_stopped = True
            time.sleep(2*self.update_interval)

        self.is_stopped = False
        self.update()
        self.check_sfc_duration()

        # If mobility is activated
        if self.mobility_activated != 0:
            self.start_tracer_thread()

        if self.allow_temporary_high_latency:
            self.start_check_altered_sfc_thread(interval = 5)

        # If crasher is activated
        # if self.crasher_activate != 0:
        #     self.start_crasher_thread(self.crasher.time_interval, trigger_value="online")

        #Run simulation
        _thread.start_new_thread(self.run, ())

    def output_network_resources(self,deploy_time):
        self.output_writter.output_cpu_utilization(self.substrate_network, self.crasher.crashed_nodes, deploy_time)
        self.output_writter.output_cache_utilization(self.substrate_network, self.crasher.crashed_nodes, deploy_time)
        self.output_writter.output_bandwidth_utilization(self.substrate_network, deploy_time)
        self.output_writter.output_nodes_sf_utilization(self.substrate_network, deploy_time)
        #self.output_utils.output_edges_sf_utilization(self.edges_vnf, self.existing_vnf, self.sfc_list, deploy_time, route_info, sfc)

    def output_flows(self,current_time, sfc, latency, run_duration, is_success,bw_transcode,backup_sfc_activated=0,latency_diff=None,wait_time=None):
        # if self.crasher.a_server_was_crashed == 1:
        #     self.crash_moment = self.crash_moment + 1
        sfc_to_remove = sfc.id
        backup_sfc = False
        if sfc.id in self.sfcs_crashed:
            crash_moment = 1
            if self.sfcs_crashed[sfc.id]['has_backup'] == True:
                latency_diff = self.sfcs_crashed[sfc.id]['latency_diff']
                backup_sfc = self.sfcs_crashed[sfc.id]['backup_sfc']
            else:
                old_latency = self.sfcs_crashed[sfc.id]['old_latency']
                new_latency = latency
                latency_diff = old_latency-new_latency if latency != None else None
        else:
            crash_moment = 0 

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
                                         self.users_crashed,
                                         self.sfcs_crashed,
                                         crash_moment,
                                         backup_sfc_activated,
                                         latency_diff,
                                         )

        if sfc.id in list(self.sfcs_crashed.keys()):
            self.sfcs_crashed.pop(sfc.id)
        # if backup_sfc in list(self.sfcs_crashed.keys()):
        #     self.sfcs_crashed.pop(sfc.id)

        self.crasher.a_server_was_crashed = 0 # Resets crash variable

    def start_check_altered_sfc_thread(self, interval):
        def task():
            while not self.is_stopped:  # Loop infinito para chamar a função repetidamente
                self.check_altered_sfcs()
                time.sleep(interval)

        thread_check_alt = threading.Thread(target=task)
        thread_check_alt.daemon = True
        thread_check_alt.start()

    # def start_crasher_thread(self, interval, trigger_value, mode='reliablity', poisson_lambda=100):
    #     def task():
    #         # Espera até que a variável network_status tenha o valor esperado
    #         while self.network_status != trigger_value:
    #              time.sleep(5)

    #         # Uma vez que o trigger value é atingido, inicia a tarefa
    #         with self.lock:
    #             if self.network_status != trigger_value:
    #                 self.start_crasher()
    #             #time.sleep(interval)

    #     thread = threading.Thread(target=task)
    #     thread.daemon = True
    #     thread.start()
        
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
        self.check_cpu_threshold()

    def check_cpu_threshold(self):
        """Information about over utilization of CPU resources."""

        self.over_threshold_nodes_list = []
        for node in self.substrate_network.nodes():
            self.check_node_cpu_threshold(node)

    def check_sfc_dst_v2(self,sfc):
        update_sfc = copy.deepcopy(sfc)
        
        old_loc = update_sfc.dst_node
        sfc_id = update_sfc.id
        duration = self.sfc_id_duration[sfc_id] if sfc_id in self.sfc_id_duration.keys() else update_sfc.duration

        player = sfc_id.split("_")[2][1]
        p_session = sfc_id.split("_")[3]
        user_id = int(player + p_session)

        dst_server_crashed = old_loc in self.crasher.crashed_nodes
        
        if dst_server_crashed:
            topology = self.tracer.topology 
            current_server_id = old_loc
            excluded_servers = self.crasher.crashed_nodes
            # Coordenadas do servidor atual
            current_position = topology[current_server_id]

            # Variáveis para armazenar o servidor mais próximo e a menor distância
            nearest_server_id = None
            nearest_distance = float('inf')

            distances = []
            # Coletar todas as distâncias e médias de consumo de banda para normalização
            all_distances = []
            all_bandwidths = []

            for server_id, position in topology.items():
                if server_id not in excluded_servers and server_id != current_server_id:
                    # Calcular a distância euclidiana
                    distance = math.sqrt((current_position[0] - position[0]) ** 2 + (current_position[1] - position[1]) ** 2)
                    links = self.substrate_network._adj[server_id]
                    average_bandwidth_used = sum(link['bandwidth_used'] for link in links.values()) / len(links)
                    
                    all_distances.append(distance)
                    all_bandwidths.append(average_bandwidth_used)
                    
                    distances.append((server_id, distance, average_bandwidth_used))
    
            # Pesos para distância e consumo de banda
            peso_distancia = 0.5
            peso_banda = 0.5

            # Lista de distâncias e custos
            distances = []

            for server_id, position in topology.items():
                if server_id not in excluded_servers and server_id != current_server_id:
                    # Calcular a distância euclidiana
                    distance = math.sqrt((current_position[0] - position[0]) ** 2 + (current_position[1] - position[1]) ** 2)
                    links = self.substrate_network._adj[server_id]
                    average_bandwidth_used = sum(link['bandwidth_used'] for link in links.values()) / len(links)
                    distances.append((server_id, distance, average_bandwidth_used))

            # Ordenar a lista de distâncias
            distances.sort(key=lambda x: x[1])

            # Obter os 10 servidores mais próximos
            nearest_servers = distances[:8]

            # Coletar todas as distâncias e médias de consumo de banda dos 10 mais próximos para normalização
            all_distances = [entry[1] for entry in nearest_servers]
            all_bandwidths = [entry[2] for entry in nearest_servers]

            # Normalizar as distâncias e os consumos de banda
            max_distance = max(all_distances)
            min_distance = min(all_distances)
            max_bandwidth = max(all_bandwidths)
            min_bandwidth = min(all_bandwidths)

            normalized_distances = []
            for server_id, distance, average_bandwidth_used in nearest_servers:
                normalized_distance = (distance - min_distance) / (max_distance - min_distance) if max_distance != min_distance else 0
                normalized_bandwidth = (average_bandwidth_used - min_bandwidth) / (max_bandwidth - min_bandwidth) if max_bandwidth != min_bandwidth else 0
                
                # Calcular o custo combinando distância e consumo de banda normalizados
                custo = peso_distancia * normalized_distance + peso_banda * normalized_bandwidth
                
                normalized_distances.append((server_id, distance, average_bandwidth_used, custo))

            # Ordenar a lista de servidores pelo custo
            normalized_distances.sort(key=lambda x: x[3])

            # Obter o servidor com o menor custo
            nearest_server_id, nearest_distance, _, nearest_custo = normalized_distances[0]

            location  =  nearest_server_id
            new_vnfs_list_dict = copy.deepcopy(update_sfc.vnfs_dict)

            if re.search('cache', sfc_id) is not None:
                old_loc = str(update_sfc.dst.substrate_node)
                new_loc = str(location)

                ma_old_key = 'MA_region_' + old_loc
                re_old_key = 'RE_region_' + old_loc
                if sfc_id in self.sfcs_routing_info:
                    if ma_old_key in self.sfcs_routing_info[sfc_id].keys():
                        stored_info = copy.deepcopy(self.sfcs_routing_info[sfc_id][ma_old_key])

                        del self.sfcs_routing_info[sfc_id][ma_old_key]

                        ma_new_key = re.sub(old_loc, new_loc,ma_old_key)

                        self.sfcs_routing_info[sfc_id][ma_new_key] = stored_info
                        new_vnfs_list_dict[1]['name'] = ma_new_key

                    if re_old_key in self.sfcs_routing_info[sfc.id].keys():
                        stored_info = copy.deepcopy(self.sfcs_routing_info[sfc_id][re_old_key])

                        del self.sfcs_routing_info[sfc_id][re_old_key]

                        re_new_key = re.sub(old_loc, new_loc,re_old_key)

                        self.sfcs_routing_info[sfc_id][re_new_key] = stored_info
                        new_vnfs_list_dict[2]['name'] = re_new_key
                else:
                    ma_new_key = re.sub(old_loc, new_loc,ma_old_key)
                    new_vnfs_list_dict[1]['name'] = ma_new_key

                    re_new_key = re.sub(old_loc, new_loc,re_old_key)
                    new_vnfs_list_dict[2]['name'] = re_new_key

            new_sfc_dict = {}
            new_sfc_dict["name"] = sfc_id
            new_sfc_dict["vnf_list"] = new_vnfs_list_dict
            new_sfc_dict["bandwidth"] = update_sfc.input_throughput
            new_sfc_dict["src_node"] = update_sfc.src.substrate_node
            new_sfc_dict["dst_node"] = location
            new_sfc_dict["duration"] = duration
            new_sfc_dict["latency"] = update_sfc.latency_request

            new_sfc = SFCGenerator(new_sfc_dict).generate()
            # if sfc_id in self.sfcs_that_deployed:
            #     pass
            # else:
            return new_sfc     
        else:
            return sfc
        
    def check_sfc_dst(self,sfc):
        update_sfc = copy.deepcopy(sfc)
        
        old_loc = update_sfc.dst_node
        sfc_id = update_sfc.id
        duration = self.sfc_id_duration[sfc_id] if sfc_id in self.sfc_id_duration.keys() else sfc.duration

        player = sfc_id.split("_")[2][1]
        p_session = sfc_id.split("_")[3]
        user_id = int(player + p_session)
        
        try:
            dst_server_crashed = old_loc in self.crasher.crashed_nodes

            if dst_server_crashed: 
                # Nós precisamos criar um veículo para o usuário, mesmo que a sfc venha a não ser instanciada corretamente, pois é necessária uma localização
                if not self.tracer.is_vehicle_created(user_id): 
                    self.tracer.create_vehicle(user_id,old_loc)

                self.tracer.check_sfc_for_user(user_id,sfc_id)
                location = self.tracer.get_closest_server(user_id,self.crasher.crashed_nodes)

                if location == -1:
                    topology = self.tracer.topology 
                    current_server_id = old_loc
                    excluded_servers = self.crasher.crashed_nodes
                    # Coordenadas do servidor atual
                    current_position = topology[current_server_id]

                    # Variáveis para armazenar o servidor mais próximo e a menor distância
                    nearest_server_id = None
                    nearest_distance = float('inf')

                    # Iterar sobre todos os servidores na topologia
                    for server_id, position in topology.items():
                        if server_id not in excluded_servers and server_id != current_server_id:
                            # Calcular a distância euclidiana
                            distance = math.sqrt((current_position[0] - position[0]) ** 2 + (current_position[1] - position[1]) ** 2)

                            # Verificar se a distância atual é menor que a menor distância encontrada
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest_server_id = server_id

                    # Imprimir o resultado
                    print(f"O servidor mais próximo é o ID {nearest_server_id} com uma distância de {nearest_distance:.2f} unidades.")
                    location  =  nearest_server_id

                new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)

                if re.search('cache', sfc_id) is not None:
                    old_loc = str(sfc.dst.substrate_node)
                    new_loc = str(location)

                    ma_old_key = 'MA_region_' + old_loc
                    re_old_key = 'RE_region_' + old_loc
                    if sfc_id in self.sfcs_routing_info:
                        if ma_old_key in self.sfcs_routing_info[sfc_id].keys():
                            stored_info = copy.deepcopy(self.sfcs_routing_info[sfc_id][ma_old_key])

                            del self.sfcs_routing_info[sfc_id][ma_old_key]

                            ma_new_key = re.sub(old_loc, new_loc,ma_old_key)

                            self.sfcs_routing_info[sfc_id][ma_new_key] = stored_info
                            new_vnfs_list_dict[1]['name'] = ma_new_key

                        if re_old_key in self.sfcs_routing_info[sfc.id].keys():
                            stored_info = copy.deepcopy(self.sfcs_routing_info[sfc_id][re_old_key])

                            del self.sfcs_routing_info[sfc_id][re_old_key]

                            re_new_key = re.sub(old_loc, new_loc,re_old_key)

                            self.sfcs_routing_info[sfc_id][re_new_key] = stored_info
                            new_vnfs_list_dict[2]['name'] = re_new_key
                    else:
                        ma_new_key = re.sub(old_loc, new_loc,ma_old_key)
                        new_vnfs_list_dict[1]['name'] = ma_new_key

                        re_new_key = re.sub(old_loc, new_loc,re_old_key)
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
                return new_sfc            
            else:
                return sfc
            
        except Exception as e:
            print(f"erro no check sfc  {e}")
            traceback.print_exc()
            return sfc           
            
            
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
        
        sfc = copy.deepcopy(self.check_sfc_dst_v2(sfc))
        
        alg = copy.deepcopy(self.alg)
        alg.clear_all()
        alg.install_substrate_network(self.substrate_network)
        alg.install_SFC(sfc)
        
        s = time.time() # Start measuring how long it takes to the alg run

        shareable_sfs = self.substrate_network.get_shareable_sfs()

        match self.alg_name:
            case 'ga' | 'osfem' | 'goku': # algs with active reuse and cost method
                alg.set_costs(self.costs_parameters)
                alg.start_algorithm(shareable_sfs=shareable_sfs)
            case _: # algs with passive reuse and no cost method
                alg.start_algorithm() 

        s2 = time.time()

        route_info = alg.get_route_info() # Routes choosen by the alg
        latency = alg.get_latency() # latency of the solution
        bit_rate_adjust =  1.0 if self.alg_name != 'osfem' else alg.get_bit_rate_used()
        bw_transcode =  sfc.vnfs_dict[-1]['out_bw'] if self.alg_name != 'osfem' else alg.get_transcode_bw()

        if (self.alg_name == 'musfico') and (sfc.id in self.sfcs_routing_info.keys()): # musfico exclusive methodology
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

            is_success = 1
            self.deploy_success(sfc)
            if sfc not in self.sfcs_routing_info.keys():
                self.sfcs_routing_info[sfc.id] = copy.deepcopy(route_info)
        else:
            self.deploy_failure = 1
            self.deploy_failed(sfc)
            
        self.update()
        is_success = self.check_resources_exceed(is_success,sfc) # Check if any fees exceed 100%
        self.counter += 1 # at this time all verifications are done. So we add 1 to counter of sfc

        # output of the simulation
        self.output_network_resources(deploy_time=current_time)
        self.output_flows(current_time,sfc,latency,run_duration,is_success,bw_transcode,latency_diff=None,backup_sfc_activated=0,wait_time=wait_time)

        actual_session =  int(sfc.id.split("_")[3])
        actual_player  =  int(sfc.id.split("_")[2][1:])
        session_break_crasher = self.flows/2

        if actual_session >= session_break_crasher  and self.crasher_activate != 0 and self.crash_trials == 0 :
            self.start_crasher()
            self.crash_trials = self.crash_trials + 1
            self.network_status = 'online' #flag for crasher thread start
        
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
        #with self.lock:
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

            if cpu_utilization > 1 or cache_utilization > 1 or bw_utilization > 1: # Check if any fees exceed 100%
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
    
    def check_node_cpu_threshold(self, node_id: str) -> None:
        """
        Returns information about CPU consumption for a given node.

        Args:
            node_id (str): The node to be verified.
        """

        cpu_used = self.substrate_network.get_node_cpu_used(node_id)
        cpu_capacity = self.substrate_network.get_node_cpu_capacity(node_id)
        if cpu_capacity == 0:
            return
        if float(cpu_used)/float(cpu_capacity) > self.cpu_threshold:
            self.over_threshold_nodes_list.append(node_id)

    def check_sfc_duration(self) -> None:
        """ 
        Implements a counter to remove SFCs whose durations
        are over.
        """
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
            player = sfc_id.split("_")[2][1]
            p_session = sfc_id.split("_")[3]
            user_id = int(player + p_session)
            self.tracer.set_sfc_status_for_user(user_id,sfc_id, new_status = 'completed')
            self.undeploy_sfc(sfc_id)


        self.check_altered_sfcs()

        time2 = time.time()
        if time2 - time1 > 1:
            if not self.is_stopped:
                self.timer = Timer(0, self.check_sfc_duration, ()).start()
        else:
            if not self.is_stopped:
                self.timer = Timer((self.update_interval \
                                    - (time2 - time1)), self.check_sfc_duration, ()).start()
    
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

            if self.allow_temporary_high_latency:
                if latency > self.latency_interval[0] and latency <= self.latency_interval[1]:  #latency bettwen 6 and 10 ms
                    altered_sfc = True

            # TODO bitrate implementation is not being done in this simulation
            # if bitrate != 1.0:
            #     altered_sfc = True

            if altered_sfc == True:
                if sfc.id not in list(self.altered_sfcs.keys()):
                    self.altered_sfcs[sfc.id] = {'wait_time':0,'timestamp':time.time(),'flag':True} # Primeira tentativa de alocação sem delay alto
                    return True, None # Na primeira passagem ele permite a SFC ter delay alto
                else:
                    allow_reroute = self.altered_sfcs[sfc.id]['flag']
                    if allow_reroute:
                        return True, self.altered_sfcs[sfc.id]['wait_time']
                    else:
                        return False, 30
            else:
                if sfc.id in self.altered_sfcs: # Nesse caso a sfc não está alterada agora, mas antes ela estava, o que quer dizer que ela saiu desse estado
                    wait_time = self.altered_sfcs[sfc.id]['wait_time']
                    self.altered_sfcs[sfc.id]
                    return True, wait_time
                else:
                    return True,None
        else:
            if sfc.id in list(self.altered_sfcs.keys()): # Retira
                del self.altered_sfcs[sfc.id]
                return True,None
            else:
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
            sfc_id = sfc.id
            player = sfc_id.split("_")[2][1]
            p_session = sfc_id.split("_")[3]
            user_id = int(player + p_session)
            self.tracer.set_sfc_status_for_user(user_id,sfc_id, new_status = 'completed')
        
        if route_info == False or latency ==  None:
            sfc_id = sfc.id
            player = sfc_id.split("_")[2][1]
            p_session = sfc_id.split("_")[3]
            user_id = int(player + p_session)
            vehicle_is_created = self.tracer.is_vehicle_created(user_id)
            if vehicle_is_created:
                self.tracer.set_sfc_status_for_user(user_id,sfc_id, new_status = 'completed')
        
        if route_info:
            #print(route_info)
            servers_used = list(set([server for key, servers in route_info.items() if key not in ['src', 'dst'] for server in servers]))
            crashed_nodes = self.crasher.crashed_nodes
            intersection = [x for x in servers_used if x in crashed_nodes]

            if len(intersection ) > 0 :
                print(f"Crashed Nodes: {crashed_nodes}")
                print(f"My route_info: {route_info}")
                print(f"Servers Used: {servers_used}")
                print(f"Servers that are being used that crashed: {intersection}")
                print("Already crash")
                route_info = False
                latency = None
        return route_info, latency

    def start_crasher(self):
        network = self.substrate_network
        servers_crashed = self.crasher.activate_crasher(network)
        sfc_id_duration = self.sfc_id_duration.items()
        
        if len(servers_crashed) != 0: 
            print(f"Servidores Crashados: {servers_crashed}")
        # Inicializa o dicionário sfcs_crashed se ele ainda não foi inicializado
        self.sfcs_crashed = {}
        sfc_ids = []
        for server in self.crasher.crashed_nodes:
            server_info = self.substrate_network.get_node_sfc_vnf_list(server)
            filtered_edges = {key: value for key, value in self.edges_vnf.items() if server in key}
        
            if server_info != []:
                sfc_ids = list(set([sfc[0] for sfc in server_info]))
                users_crashed = []
                pattern = re.compile(r'p\d+_\d+')

                for sfc_id in sfc_ids:
                    # Popula o dicionário sfcs_crashed
                    match = pattern.search(sfc_id)
                    if match:
                        users_crashed.append(match.group())
                users_crashed = list(set(users_crashed))

                self.users_crashed = self.users_crashed + len(users_crashed)
                #print(sfc_ids)
                for sfc_id in sfc_ids:
                    if sfc_id in sfc_id_duration:
                        # TODO: fazer alguma verificação pra garantir que aquela sfc está rodando
                        pass
                    #try:
                    if sfc_id not in self.substrate_network.sfc_dict:
                        continue
                    sfc_rf = network.sfc_route_info[sfc_id]
                    latency_sfc= sum((len(value) - 1) for key, value in sfc_rf.items() if key not in ('src', 'dst'))
                    self.sfcs_crashed[sfc_id] = {'fall_time':time.time(),'has_backup':False,'old_latency':latency_sfc}
                    #self.sfcs_back_to_qeue_latency_diff[sfc_id] = {'old_latency': latency_sfc}

                    self.send_back_to_qeue(self.substrate_network.get_sfc_by_id(sfc_id))

                    #except Exception as e :
#                         print()
#                         print()
                    #print(f"Erro no send back to qeue {e}")
#                         traceback.print_exc()
#                         print()
#                         print()
            self.substrate_network.set_node_cache_capacity(server, 0)
            self.substrate_network.set_node_cpu_capacity(server, 0)
            
            # if sfc_ids != []:
            #     for sfc_id in sfc_ids:
            #         # Popula o dicionário sfcs_crashed
            #         self.sfcs_crashed[sfc_id] = time.time()

            for link, sfc_vnf in filtered_edges.items():
                # if sfc_vnf == []:
                #     self.substrate_network.set_link_bandwidth_capacity(link[0], link[1], 0)
                #     self.substrate_network.set_link_latency(link[0], link[1], 0)
                #     continue
                # try:
                #     sfc_id = sfc_vnf[0]
                #     self.send_back_to_qeue(self.substrate_network.get_sfc_by_id(sfc_id))
                # except:
                #     #self.undeploy_sfc(sfc_id)
                #     print("CRASHER: Não foi possível realocar a sfc no link")

                self.substrate_network.set_link_bandwidth_capacity(link[0], link[1], 0)
                self.substrate_network.set_link_latency(link[0], link[1], 100)

    def check_user_position_thread(self):
        """
            Monitors and updates the mobility of users by interfacing with the SUMO simulation.

            This function periodically checks the positions of users in the SUMO simulation. If a user's position
            has changed, it reassigns their SFC (Service Function Chain) to a new location. It runs as a thread
            to continually monitor and update the positions.

            Steps:
            1. Iterate through the list of SFCs associated with players.
            2. For each SFC, retrieve its current location and the corresponding player's ID and session.
            3. Check if the player is already created in the SUMO simulation; if not, create a vehicle for the player.
            4. Get the closest server to the player and compare it with the current SFC location.
            5. If the player's location has changed, undeploy the current SFC and create a new one at the new location.
            6. Update the routing information and reassign the SFC to the new location.
            7. Re-deploy the SFC with updated location.

            The function uses a timer to periodically re-execute itself, ensuring continuous monitoring and updating.

            Raises:
                Exception: If there is an error in re-instantiating the SFC.

        """
        time1 = time.time()

        if len(self.players_sfc_list) != 0:
            for session in self.players_sfc_list:
                for sfc_id in session:
                    if sfc_id in self.sfc_id_duration and sfc_id in self.sfcs_routing_info:
                        
                        new_sfc_list = []
                        try:
                            sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                        except:
                            break
                        sfc_location = sfc.dst.substrate_node

                        player = sfc_id.split("_")[2][1]
                        p_session = sfc_id.split("_")[3]

                        # The user id in the tracer is given by concatenation of the id with the session number
                        user_id = int(player + p_session)

                        # Check if the player is already in the tracer, if not it creates a vehicle for it
                        # Check if the simulation is still running 
                        if not self.tracer.is_simulation_running():
                            break

                        if not self.tracer.is_vehicle_created(user_id):
                            self.tracer.create_vehicle(user_id,sfc_location)
                        
                        self.tracer.check_sfc_for_user(user_id,sfc_id)
                        
                        player_location = self.tracer.get_closest_server(user_id,self.crasher.crashed_nodes,sfc_location)
                        
                        if player_location != -1:
                            try:
                                if player_location != sfc_location:
                                    if self.verbose == True:
                                        print(f"sfc changed location from {sfc_location} to {player_location}")
                                    self.send_back_to_qeue(sfc,changed_location=True,new_location=player_location)
                            except:
                               print("Erro na reinstaciação da SFC") 
                        else:
                            player_location = sfc_location
                            
        time2 = time.time()
        
        if time2 - time1 > 1:
            if not self.is_stopped:
                self.timer = Timer(0, self.check_user_position_thread, ()).start()
        else:
            if not self.is_stopped:
                self.timer = Timer((self.update_interval \
                                    - (time2 - time1)), self.check_user_position_thread, ()).start()

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
                try:
                    self.undeploy_sfc(sfc)
                    del self.altered_sfcs[sfc.id]
                except:
                    pass # Só para não tratar sfcs do mesmo usuários passando por undeploy
            else:
                self.altered_sfcs[sfc_id]['flag'] = True
                self.send_back_to_qeue(sfc)

    
    def send_back_to_qeue(self,sfc,changed_location=False,new_location=False):
        sfc_id = sfc.id
        new_sfc_list = []
        location = sfc.dst.substrate_node if new_location == False else new_location
        duration = self.sfc_id_duration[sfc_id]

        #try:

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
        self.tracer.stop_sumo_simulation()
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

    def handle_cpu_over_threshold(self, alg: object) -> None:
        """Seems use to for test. """
        ## undeploy the sfc, redeploy sfc by disable the over threshold cpu
        self.stop()
        for node in self.over_threshold_nodes_list:
            # (sfc_id, vnf) = sn.get_node_sfc_vnf_list(node)
            sfc_vnf_list = self.substrate_network.get_node_sfc_vnf_list(node)
            for (sfc_id, vnf) in sfc_vnf_list:
                # TODO: here we should consider which sfc need to be undployed. 
                # May according to priority or some history data or SLA. or cost...
                sfc = self.substrate_network.get_sfc_by_id(sfc_id)

                self.undeploy_sfc(sfc_id)
                sn = copy.deepcopy(self.substrate_network)
                sn.set_node_cpu_capacity(node, 0)
                sn.set_node_cpu_free(node, 0)
                alg.install_substrate_network(sn)
                alg.install_SFC(sfc)
                alg.start_algorithm()
                route_info = alg.get_route_info()
                if sfc.id in self.sfc_list:
                    print("sfc has been deployed")
                    return
                self.substrate_network.deploy_sfc(sfc, route_info)
                self.sfc_list.append(sfc.id)
                self.substrate_network.update()

        self.start()

    def submit_sfcs(self) -> None:
        """Submit the SFCs stored in the queue."""
        last_sf_mono = 'sfc_unique_p4_' + str(self.flows)
        last_sf_dec = 'sfc_mono_p4_' + str(self.flows)

        
        while not self.is_stopped:
            sfc_list = self.sfc_queue.peek_sfc()    
            
            if self.verbose == True:       
                print("queue_size: " + str(self.sfc_queue.qsize()))
            
            if self.max_queue_size < self.sfc_queue.qsize():
                self.max_queue_size = self.sfc_queue.qsize()
            
            player_sfc_id_list = []
            
            for sfc in sfc_list:
                t_1 = time.time()
                #with self.lock:
                self.deploy_sfc(sfc)
                player_sfc_id_list.append(sfc.id)
                self.update()
                t_2 = time.time()
                if self.verbose ==  True:
                    print("          algorithm take time: ", round(t_2 - t_1,4))
                
                if sfc.id in (last_sf_mono, last_sf_dec):
                    print('Last SFC released')
                    print('Max queue size:', self.max_queue_size )
                    self.stop()
                    sys.exit()
            self.players_sfc_list.append(player_sfc_id_list)
