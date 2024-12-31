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
from typing import Optional
from controllers.modules.sfcs_manager import SFCManager
from controllers.sfc_generator import SFCGenerator

from controllers.modules.mobility_manager import MobilityManager
from controllers.modules.crasher import Crasher
from controllers.modules.resource_manager import ResourceManager

from controllers.sfc_queue import SFCQueue
from core.net import Net
from utils.manager_results import OutputWritter

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
    def __init__(self):
        # Rede
        self.substrate_network = Net()
        self.node_info = {}

        # Modules
        self.mobility_manager = MobilityManager()
        self.sfc_manager = SFCManager()
        self.crasher_manager = Crasher()

        # Status da Rede
        self.remaining_time = None
        self.update_interval = 1
        self.is_stopped = True
        
        # SFC (Service Function Chains)
        self.players_sfc_list = []
        self.sfc_queue = None
        self.timer_qeue_sfcs = {}
        self.sfcs_crash_affected = {}

        # Implementações extras
        self.verbose = False
        self.log_file = "backup_log.txt"
        self.crasher_activate = False
        self.mobility_activated = None
        self.latency_interval = [6, 10]
        self.allow_high_latency = False
        self.altered_sfcs = {}


        # Estatísticas
        self.success = []
        self.counter = 0
        # self.deploy_failure = 0

        # Outros
        self.lock = threading.Lock()
        self.flows = 0
        self.alg = None
        self.timer = None
        self.output_writter : Optional[OutputWritter] = None
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
        
        sequential = True
        
        if sequential:
            if self.mobility_activated:
                self.mobility_manager.start_simulation()
            
            _thread.start_new_thread(self.sequential_operation,())
            #self.sequential_operation(self)
        else:
            self.update()
            self.check_sfc_duration()

            # If mobility is activated
            if self.mobility_activated:
                self.mobility_manager.start_simulation()
                self.start_tracer(interval = 5)

            if self.alg.name in ['vegeta']:
                self.start_backup_manager(interval=20,threshold=0.5)
            
            if self.crasher_activate:
                self.start_crasher(interval=200)

            # if self.allow_high_latency:
            #     self.start_check_altered_sfc_thread(interval = 5)
            #Run simulation
            _thread.start_new_thread(self.run, ())

    def run(self) -> None:
        """Starts the simulation."""
        while not self.is_stopped:
            self.submit_sfcs()
    def stop(self) -> None:
        """Stops the simulation."""
        self.is_stopped = True
        self.mobility_manager.stop_simulation()
        if self.timer:
            self.timer.cancel()

    def check_sfc_duration(self) -> None:
        """
        Implements a counter to remove SFCs whose durations
        are over, using a thread.
        """
        def task():
            while not self.is_stopped:
                with self.lock:
                    time1 = time.time()
                    for sfc_id in self.sfc_manager.check_sfc_duration():
                        self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network)
                        self.mobility_manager.remove_sfc(sfc_id)

                    for sfc_id in self.sfc_manager.get_list_sfc_duration(threshold=20):
                        self.mobility_manager.remove_sfc(sfc_id)
                    
                    time2 = time.time()
                    elapsed_time = time2 - time1
                    # intervalo baseado no tempo de execução da função
                    if elapsed_time < self.update_interval:
                        sleep_time = self.update_interval - elapsed_time
                    else:
                        sleep_time = 0  # Se o tempo de execução for maior que o intervalo, reexecuta imediatamente

                if not self.is_stopped:
                    time.sleep(sleep_time)  # Aguarda o tempo restante para completar o intervalo

        thread = threading.Thread(target=task)
        thread.daemon = True  # Faz a thread rodar como daemon para que o programa termine sem bloqueios
        thread.start()
    
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
                                # try:
                                print(f"SFC {sfc.id} mudou de localização para {new_location}")
                                self.send_back_to_qeue(sfc, changed_location=True, new_location=new_location)
                                # except:
                                #     print("Erro na reinstanciação da SFC")
                    # end_time = time.time()  # Marca o tempo de fim
                    # elapsed_time = end_time - start_time

                    # # Abre o arquivo de log em modo append para não sobrescrever os dados existentes
                    # with open(self.log_file, "a") as log:
                    #     log.write(f"Tempo total de execução: {elapsed_time:.2f} segundos\n")
                time.sleep(interval)  # Intervalo entre as verificações

        # # Verifica se o arquivo já existe e apaga se necessário, depois cria um novo
        # with open(self.log_file, "w") as log:
        #     log.write("Log de Mobilidade\n")
        #     log.write("====================\n")

        # Inicia a thread
        thread_mob = threading.Thread(target=task_mob)
        thread_mob.daemon = True 
        thread_mob.start()

    def start_crasher(self,interval=25):
        def task_crash():
            time.sleep(interval)
            if not self.is_stopped:
                with self.lock:
                    nodes_to_crash = self.crasher_manager.activate_crasher(self.substrate_network)
                    sfcs_affected = self.crasher_manager.implement_crash(nodes_to_crash,self.substrate_network)
                    self.recover_sfcs(sfcs_affected)

                    #sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes()
                    for sfc_id in sfcs_affected:
                        try:
                            sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                        except:
                            continue
                        # try:
                        print(f"SFC {sfc.id} crashou")
                        self.send_back_to_qeue(sfc, changed_location=False)
                        # except:
                        #     print("Erro na reinstanciação da SFC")

        thread_mob = threading.Thread(target=task_crash)
        thread_mob.daemon = True 
        thread_mob.start()

    def recover_sfcs(self,sfc_affected):
        return
        # for sfc_id in sfc_affected:
        #     self.mobility_manager.remove_sfc(sfc_id) # TODO sfcs que foram afetadas pelo crasher não de movimentam mais por comodidade de código 
        #     sfc = self.substrate_network.get_sfc_by_id(sfc_id)
        #     sfcs_with_backup = list(self.sfc_manager.sfs_backup.keys())
        #     if sfc.id in sfcs_with_backup:
        #         self.sfc_manager.trigger_sfc_backup(sfc.id,self.substrate_network)
        #         self.update()
        #     else:
        #         self.send_back_to_qeue(sfc, changed_location=False,punishment=10)

    def start_backup_manager(self, interval=10,threshold=0.6):
        def task_backup():
            while not self.is_stopped:
                risk_servers = []
                with self.lock:                
                    start_time = time.time()  # Marca o tempo de início
                    nodes_rel = self.substrate_network.nodes_reliability.copy()

                    # Filtrando os nós com 'rel' maior que o 'threshold'
                    filtered_nodes = {node: rel for node, rel in nodes_rel.items() if rel > threshold}
                    top_1_node = dict(sorted(filtered_nodes.items(), key=lambda item: item[1], reverse=True)[:2])

                    self.risk_servers = list(top_1_node.keys())
                    backups = self.sfc_manager.set_risk_sfcs(top_1_node,self.substrate_network)
                    if backups != []:
                        for backup in backups:
                            self.sfc_queue.put_begin(backup)
                                    
                    end_time = time.time()  # Marca o tempo de fim
                    elapsed_time = end_time - start_time
                    print(elapsed_time)
                    with open(self.log_file, "a") as log:
                        log.write(f"Tempo total de execução: {elapsed_time:.2f} segundos\n")
                time.sleep(interval) 

        with open(self.log_file, "w") as log:
            log.write("Log de Backup\n")
            log.write("====================\n")

        # Inicia a thread
        backup_t = threading.Thread(target=task_backup)
        backup_t.daemon = True 
        backup_t.start()



    def send_back_to_qeue(self,sfc,changed_location=False,new_location=False,punishment=10):
        sfc_id = sfc.id
        new_sfc_list = []
        location = sfc.dst.substrate_node if new_location == False else new_location
        sfc_id_duration = self.sfc_manager.sfc_id_duration
        duration = sfc_id_duration[sfc_id]["duration"]-(time.time()-sfc_id_duration[sfc_id]["timer"]) 
        new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)
        if changed_location == True:
            # Se a sfc original se mover, faz undeploy do backup e deleta do dicionario de sfcs com backup

            if re.search('cache', sfc_id) is not None:
                old_loc = str(sfc.dst.substrate_node)
                new_loc = str(location)

                ma_old_key = 'MA_region_' + old_loc
                re_old_key = 'RE_region_' + old_loc
                
                routing_info = copy.deepcopy(self.sfc_manager.sfcs_routing_info)
                # new_routing_info = copy.deepcopy(self.sfc_manager.sfcs_routing_info)
                if ma_old_key in routing_info[sfc_id].keys():
                    # stored_info = old_routing_info[sfc_id][ma_old_key]
                    # del new_routing_info[sfc_id][ma_old_key]
                    ma_new_key = re.sub(old_loc,new_loc,ma_old_key)
                    # new_routing_info[sfc_id][ma_new_key] = stored_info
                    new_vnfs_list_dict[1]['name'] = ma_new_key

                if re_old_key in routing_info[sfc.id].keys():
                    # stored_info = old_routing_info[sfc_id][re_old_key]
                    # del self.sfc_manager.sfcs_routing_info[sfc_id][re_old_key]
                    re_new_key = re.sub(old_loc, new_loc,re_old_key)
                    #self.sfc_manager.sfcs_routing_info[sfc_id][re_new_key] = stored_info
                    new_vnfs_list_dict[2]['name'] = re_new_key

        new_sfc_dict = {}
        new_sfc_dict["name"] = sfc_id
        new_sfc_dict["vnf_list"] = new_vnfs_list_dict
        new_sfc_dict["bandwidth"] = sfc.input_throughput
        new_sfc_dict["src_node"] = sfc.src.substrate_node
        new_sfc_dict["dst_node"] = location
        new_sfc_dict["duration"] = duration
        new_sfc_dict["latency"] = sfc.latency_request
        self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network) # retira a sfc antiga    
        new_sfc = SFCGenerator(new_sfc_dict).generate() # gera uma nova e coloca de volta na fila
        new_sfc_list.append(new_sfc)
        self.sfc_queue.put_begin(new_sfc_list) # coloca a nova
        #except Exception as e:
        #    print(f"Erro na tentativa de mandar de volta pra fila: {e}")

    def get_node_information(self, node_id: int) -> Tuple:
        """f
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

    def update(self) -> None:
        """Updates the network state and check for resource overhead."""
        self.substrate_network.update()

    def get_nodes_information(self) -> None:
        """Get information for each node in the network."""
    
        for node in self.substrate_network.nodes():
            self.node_info[node] = self.get_node_information(node)

    def output_results(self, results_dict, sfc,res_output=False) -> None:
        def output_network_resources(deploy_time):
            self.output_writter.output_cpu_utilization(self.substrate_network, deploy_time,self.crasher_manager.nodes_crashed)
            self.output_writter.output_cache_utilization(self.substrate_network, deploy_time,self.crasher_manager.nodes_crashed)
            self.output_writter.output_bandwidth_utilization(self.substrate_network, deploy_time)
            self.output_writter.output_nodes_sf_utilization(self.substrate_network, deploy_time)
            #self.output_utils.output_edges_sf_utilization(self.edges_vnf, self.existing_vnf, self.sfc_list, deploy_time, route_info, sfc)

        def output_flows(current_time, sfc, latency, run_duration, is_success,backup_sfc,latency_diff=None, wait_time=None):
            bw_transcode = 0
            self.output_writter.output_flows(
                self.substrate_network,
                wait_time,
                self.sfc_manager.get_running_players_sessions(),
                self.counter,
                self.remaining_time,
                current_time,
                sfc,
                latency,
                run_duration,
                is_success,
                backup_sfc,
                bw_transcode,
                latency_diff
            )
        def resilient_output(sfc_id,info):
            self.output_writter.resilient_output(sfc_id,info)

        """Updates the network state and check for resource overhead."""
        if results_dict:
            # output of the simulation
            if not res_output:
                output_network_resources(deploy_time=results_dict['current_time'])
                output_flows(results_dict['current_time'],sfc,
                            results_dict['latency'], results_dict['run_duration'],
                            results_dict['is_success'],results_dict['backup_sfc'])
                if not results_dict['backup_sfc']:
                    self.success.append(results_dict['is_success'])
            

        sfcs_crash_aff = copy.deepcopy(list(self.sfcs_crash_affected.keys())) 
        if sfc.id in sfcs_crash_aff:
            if not self.sfcs_crash_affected[sfc.id]['backup_success'] == True: #
                if results_dict:
                    self.sfcs_crash_affected[sfc.id]["recover_success"] = results_dict['is_success']
                    time_to_recover = time.time() - self.sfcs_crash_affected[sfc.id]["time_to_recover"] if results_dict['is_success'] else None
                    self.sfcs_crash_affected[sfc.id]["time_to_recover"] = time_to_recover
                    latency_diff = results_dict['latency'] - self.sfcs_crash_affected[sfc.id]["old_latency"] if results_dict['is_success'] else None
                    self.sfcs_crash_affected[sfc.id]["latency_diff"] = latency_diff
                else:
                    self.sfcs_crash_affected[sfc.id]["recover_success"] = False
                    self.sfcs_crash_affected[sfc.id]["time_to_recover"] = None
                    self.sfcs_crash_affected[sfc.id]["latency_diff"] = None
            info = copy.deepcopy(self.sfcs_crash_affected[sfc.id])
            resilient_output(sfc.id,info)
            del self.sfcs_crash_affected[sfc.id]
                    
        # else:
        #     if not results_dict['backup_sfc']:
        #         self.success.append(False)
        
        if self.verbose:
            print("__________________________________________")
            self.output_writter.print_output_info(self.substrate_network, self.success) # print log information
            print("") 

        #self.substrate_network.update()

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





















    # def seq_recover_sfcs(self,sfc_affected):
    #     for sfc_id in sfc_affected:
    #         self.mobility_manager.remove_sfc(sfc_id,all=True) # TODO sfcs que foram afetadas pelo crasher não de movimentam mais por comodidade de código 
    #         sfc = self.substrate_network.get_sfc_by_id(sfc_id)
    #         sfcs_with_backup = list(self.sfc_manager.sfs_backup.keys())
    #         # if sfc.id in sfcs_with_backup:
    #         #     self.send_back_to_qeue(sfc, changed_location=False,punishment=10)
    #         #     # self.sfc_manager.trigger_sfc_backup(sfc.id,self.substrate_network)
    #         #     # self.update()
    #         # else:
    #         self.send_back_to_qeue(sfc, changed_location=False,punishment=10)

#   def sequential_backup(self,threshold=0.5):
#         nodes_rel = self.substrate_network.nodes_reliability.copy()

#         # Filtrando os nós com 'rel' maior que o 'threshold'
#         filtered_nodes = {node: rel for node, rel in nodes_rel.items() if rel > threshold}
#         top_1_node = dict(sorted(filtered_nodes.items(), key=lambda item: item[1], reverse=True)[:1])

#         self.risk_servers = list(top_1_node.keys())
#         backups = self.sfc_manager.set_risk_sfcs(top_1_node,self.substrate_network)
#         if backups != []:
#             for backup in backups:
#                 self.sfc_queue.put_begin(backup)


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
        # with self.lock:
        results_dict = self.sfc_manager.deploy_sfc(sfc,self.substrate_network,self.alg)
        self.output_results(results_dict,sfc)
        if results_dict != False:
            if not results_dict['backup_sfc'] and (results_dict['is_success'] == True): # If not a backup sfc and successful deployed
                self.mobility_manager.add_vehicle(sfc)
            else:
                self.mobility_manager.remove_sfc(sfc.id,all=True)
        # else:
        #     return False


    def sequential_submit_sfcs(self):
        self.check_timer_qeue()
        last_sf_mono = 'sfc_unique_p4_' + str(self.flows)
        last_sf_dec = 'sfc_mono_p4_' + str(self.flows)

        sfc_list = self.sfc_queue.peek_sfc()           
        print("queue_size: " + str(self.sfc_queue.qsize()))
        if self.max_queue_size < self.sfc_queue.qsize():
            self.max_queue_size = self.sfc_queue.qsize()
        last_sfc_release = False
        player_sfc_id_list = []
        for sfc in sfc_list:
            t_1 = time.time()
            self.deploy_sfc(sfc)
            player_sfc_id_list.append(sfc.id)
            #self.update()
            t_2 = time.time()
            print("          algorithm take time: ", round(t_2 - t_1,4))
            if sfc.id in (last_sf_mono, last_sf_dec):
                print('Last SFC released')
                print('Max queue size:', self.max_queue_size )
                last_sfc_release = True
                self.stop()
                sys.exit()
        self.players_sfc_list.append(player_sfc_id_list)
        self.update()
        return last_sfc_release

    def sequential_crasher(self,interval=25):
        nodes_to_crash = self.crasher_manager.activate_crasher(self.substrate_network,self.sfc_manager)
        if nodes_to_crash == False:
            return False
        sfcs_affected = self.crasher_manager.implement_crash(nodes_to_crash,self.substrate_network)
        
        sfcs_with_backup = list(self.sfc_manager.sfs_backup.keys())
        to_send_back_to_qeue = []
        #sfcs_backup_success = []
        for sfc_id in sfcs_affected:
            sfc = self.substrate_network.get_sfc_by_id(sfc_id)
            if self.sfc_manager.is_Backup(sfc_id=sfc_id): # Se for de backup simplesmente retira
                self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network)
                continue
            if sfc_id in sfcs_with_backup:
                backup_vnf = self.sfc_manager.sfs_backup[sfc_id]['vnf_id']
                if backup_vnf in sfcs_affected[sfc_id]['vnf_ids']:
                    name = self.sfc_manager.sfs_backup[sfc_id]['sfc_backup_id']
                    try:
                        backup_sfc = self.substrate_network.get_sfc_by_id(name)
                    except:
                        self.sfc_manager.take_off_backup(sfc_id,self.substrate_network)
                        to_send_back_to_qeue.append(sfc)
                        continue
                    node_crash = self.crasher_manager.nodes_crashed[0]
                    self.sfc_manager.trigger_sfc_backup(sfc,backup_sfc,self.substrate_network,node_id=node_crash)
                    #sfcs_backup_success.append(sfc)
                    o_rf = self.substrate_network.sfc_route_info[sfc_id]
                    o_latency = self.sfc_manager.calculate_latency(o_rf)

                    self.sfcs_crash_affected[sfc_id] = {"recover_success":True,"backup_success":True,"latency_diff":0,"old_latency":o_latency,"time_to_recover":random.uniform(1,2)}
                    self.output_results(results_dict=False,sfc=sfc,res_output=True)
                    #self.update()
                else: # A sfc tem backup mas não é dessa sf
                    to_send_back_to_qeue.append(sfc)
                    continue
            else: # A sfc não tem backup
                to_send_back_to_qeue.append(sfc)
                continue        

        for sfc in to_send_back_to_qeue:        
            sfc_id = sfc.id
            o_rf = self.substrate_network.sfc_route_info[sfc_id]
            o_latency = self.sfc_manager.calculate_latency(o_rf)
            new_sfc_list = []
            location = sfc.dst.substrate_node
            sfc_id_duration = self.sfc_manager.sfc_id_duration 
            duration = sfc_id_duration[sfc_id]["duration"]-(time.time()-sfc_id_duration[sfc_id]["timer"])   + 15 # bonus
            new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)
            new_sfc_dict = {}
            new_sfc_dict["name"] = sfc_id
            new_sfc_dict["vnf_list"] = new_vnfs_list_dict
            new_sfc_dict["bandwidth"] = sfc.input_throughput
            new_sfc_dict["src_node"] = sfc.src.substrate_node
            new_sfc_dict["dst_node"] = location
            new_sfc_dict["duration"] = duration
            new_sfc_dict["latency"] = sfc.latency_request
            self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network) # retira a sfc antiga  
              
            new_sfc = SFCGenerator(new_sfc_dict).generate() # gera uma nova e coloca de volta na fila
            new_sfc_list.append(new_sfc)
            self.timer_qeue_sfcs[sfc_id] = {"new_sfc_list":new_sfc_list,"timer":time.time()}
            self.sfcs_crash_affected[sfc_id] = {"recover_success":None,"backup_success":False,"latency_diff":None,"old_latency":o_latency,"time_to_recover":time.time()}
        print()

    def sequential_recovery(self,interval=200):
        self.crasher_manager.recover_from_crash(self.substrate_network)

    def check_timer_qeue(self):
        if self.timer_qeue_sfcs != {}:
            final_time = time.time()
            copy_timer_qeue = copy.deepcopy(self.timer_qeue_sfcs)
            for sfc_id,info in copy_timer_qeue.items():
                time_elapsed = final_time - info['timer']
                if  time_elapsed >= random.randint(4,6):
                    self.sfc_queue.put_begin(info["new_sfc_list"])
                    del self.timer_qeue_sfcs[sfc_id]

    def sequential_backup(self, threshold=0.0):
        nodes_rel = self.substrate_network.nodes_reliability.copy()

        # Filtrando os nós com 'rel' maior que o 'threshold'
        filtered_nodes = {node: rel for node, rel in nodes_rel.items() if rel > threshold}

        # Ordenando os servidores por confiabilidade (reliability) em ordem crescente
        sorted_nodes = sorted(filtered_nodes.items(), key=lambda item: item[1])

        # os servidores com menores 'reliability'
        top_5_nodes = dict(sorted_nodes[:5])
        top_1 = dict(sorted_nodes[-1:])
        if not top_5_nodes:
            print("Nenhum servidor disponível com confiabilidade adequada.")
            return
        
        # # Escolhendo aleatoriamente 2 servidores dos 5 com menor 'reliability'
        # servers = list(top_5_nodes.keys())

        chosen_servers = random.sample(list(top_1.keys()), 1)

        # Chamando a função 'set_risk_sfcs' com os servidores escolhidos
        backups = self.sfc_manager.set_risk_sfcs(chosen_servers, self.substrate_network)
        
        # Adicionando os backups na fila, se existirem
        if backups:
            for backup in backups:
                self.sfc_queue.put_begin(backup)

    def sequential_mobility(self, interval=5):
        if len(self.players_sfc_list) != 0:
            sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes()
            for sfc_list, new_location in zip(sfcs_moved, new_locations):
                for sfc_id in sfc_list:

                    sfc = self.substrate_network.get_sfc_by_id(sfc_id)

                    print(f"SFC {sfc.id} mudou de localização para {new_location}")
                    self.send_back_to_qeue(sfc, changed_location=True, new_location=new_location)

    def sequential_check_duration(self):
        remove_list = self.sfc_manager.check_sfc_duration()
        for sfc_id in remove_list:
            self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network)
            self.mobility_manager.remove_sfc(sfc_id,all=True)
        self.update()


    def sequential_operation(self):
        mobility_interval = 5
        crasher_interruption_time = 200
        crasher_recovery_time = 400
        crasher_activated = False
        recovery_activated = False
        backup_interval_creation = 25
        
        start_timer = time.time()
        last_mobility_time = start_timer
        last_backup_time = start_timer
        i = 0
        
        while not self.is_stopped:
            last_sfc_release = self.sequential_submit_sfcs()
            if last_sfc_release:
                break
                       
            current_time = time.time()
            
            if self.mobility_activated:
                decorrido = current_time - last_mobility_time
                if decorrido >= mobility_interval:
                    self.sequential_check_duration() 
                    self.sequential_mobility()
                    self.sequential_submit_sfcs()
                    last_mobility_time = time.time()
                    self.sequential_check_duration()

            if self.alg.name in ['vegeta']:
                decorrido = current_time - last_backup_time
                if decorrido >= backup_interval_creation:
                    self.sequential_check_duration() 
                    self.sequential_backup()  # criação de backups
                    self.sequential_submit_sfcs()
                    last_backup_time = time.time()
                    self.sequential_check_duration() 
            
            if self.crasher_activate:
                # Aciona o crasher se o tempo total atingir 200 segundos, apenas uma vez
                if not crasher_activated and current_time - start_timer >= crasher_interruption_time:
                    self.sequential_check_duration() 
                    retorno = self.sequential_crasher()  # Substitua pelo método de ativação do crasher
                    if retorno != False:
                        self.sequential_submit_sfcs()
                        crasher_activated = True
                        crasher_activation_time = time.time()
                        self.sequential_check_duration()

                                # Recupera o sistema após o tempo de recuperação
                if crasher_activated and not recovery_activated:
                    recovery_elapsed = current_time - crasher_activation_time
                    if recovery_elapsed >= crasher_recovery_time:
                        self.sequential_recovery()  # Substitua pelo método de recuperação
                        recovery_activated = True
            self.sequential_check_duration()    
            self.check_timer_qeue()
            print(i)
            i = i + 1
    # def start_check_altered_sfc_thread(self, interval):
    #     def task():
    #         while not self.is_stopped:
    #             altered_sfcs = list(self.altered_sfcs.items())  # Create a list copy of the items

    #             for sfc_id, value in altered_sfcs:
    #                 try:
    #                     sfc = self.substrate_network.get_sfc_by_id(sfc_id)
    #                 except:
    #                     continue
    #                 time_out = 30 # configurável
    #                 tempo_primeira_inicializacao = value['timestamp']
    #                 tempo_agora = time.time()
    #                 tempo_corrido = int(tempo_agora - tempo_primeira_inicializacao)
    #                 self.altered_sfcs[sfc_id]['wait_time'] = tempo_corrido # Tempo de espera aumenta
    #         if tempo_corrido >= time_out:
    #             self.altered_sfcs[sfc_id]['flag'] = False
    #             self.send_back_to_qeue(sfc)
    #             time.sleep(interval)
    #     thread_check_alt = threading.Thread(target=task)
    #     thread_check_alt.daemon = True
    #     thread_check_alt.start()


    # def check_latency_and_bitrate(self, sfc,route_info, latency: int,bitrate :int) -> bool:
    #     """
    #     Checks if the latency is within acceptable limits and handles high latency scenarios.
    #     Checks if Trascoding bitrate was adjusted

    #     Args:
    #         sfc (object): The service function chain (SFC) object containing the SF details.
    #         latency (int): The measured latency for the SFC.
    #         bitrate (int): The bitrate factor used
    #     Returns:
    #         bool: True if the latency and bitrate are acceptable, False otherwise.
    #     """
    #     # if not isinstance(latency, int):
    #     #     if isinstance(latency,float):
    #     #         pass
    #     #     else:
    #     #         return False
            
    #     if route_info:
    #         altered_sfc = False
    #         wait_time = None

    #         if self.allow_high_latency:
    #             if latency > self.latency_interval[0] and latency <= self.latency_interval[1]:  #latency bettwen 6 and  ms
    #                 altered_sfc = True

    #         # TODO bitrate implementation is not being done in this simulation
    #         # if bitrate != 1.0:
    #         #     altered_sfc = True

    #         if altered_sfc == True:
    #             if sfc.id not in list(self.altered_sfcs.keys()):
    #                 self.altered_sfcs[sfc.id] = {'wait_time':0,'timestamp':time.time(),'flag':True} # Primeira tentativa de alocação sem delay alto
    #                 return True, None 
                
    #             else:
    #                 allow_reroute = self.altered_sfcs[sfc.id]['flag']
    #                 if allow_reroute:
    #                     return True,None#self.altered_sfcs[sfc.id]['wait_time']
    #                 else:
    #                     return False,30
    #         else:
    #             # Caso não alterado
    #             if sfc.id in self.altered_sfcs: # Nesse caso a sfc não está alterada agora, mas antes ela estava, o que quer dizer que ela saiu desse estado
    #                 wait_time = self.altered_sfcs[sfc.id]['wait_time']
    #                 self.altered_sfcs[sfc.id]
    #                 del self.altered_sfcs[sfc.id]
    #                 return True, wait_time
    #             else:   
    #                 return True,None
                
    #     else:
    #         if sfc.id in list(self.altered_sfcs.keys()): # Retira
    #             del self.altered_sfcs[sfc.id]
    #         return False, None
        
    # def validate_solution(self,route_info, latency, sfc, sfc_mode):
    #     # TODO otimizar essas verificações
    #     """
    #     Validates the route information and latency.

    #     Args:
    #         route_info (dict or bool): The route information to validate.
    #         latency (int, float, or None): The measured latency to validate.
    #         sfc (object): The service function chain (SFC) object containing the SF details.
    #         sfc_mode (str): The mode of the SFC ('on' or 'off').

    #     Returns:
    #         tuple: A tuple containing the validated route_info and latency.
    #     """
    #     sfc_id = sfc.id
    #     # player = sfc_id.split("_")[2][1]
    #     # p_session = sfc_id.split("_")[3]
    #     # user_id = int(player + p_session)

    #     # if route_info:
    #     #     if (len(route_info.keys()) != 6 and sfc_mode == 'on') or (len(route_info.keys()) != 4 and sfc_mode == 'off'):
    #     #         latency = None
    #     #         route_info = False

    #     if latency is not None:
    #         if isinstance(latency, (int, float)):
    #             if latency > sfc.get_latency_request() or latency < 0:
    #                 route_info = False
    #                 latency = None
                
    #     # if is_sfc_acceptable == False: #sfc's denied because is the 2nd time it could not fit
    #     #     latency = None
    #     #     route_info = False   
    #     return route_info, latency