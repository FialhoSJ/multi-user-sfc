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
from controllers.modules.backup_manager import BackupManager
from controllers.modules.sfcs_manager import SFCManager
from controllers.sfc_generator import SFCGenerator
from collections import deque
from typing import Optional

from controllers.modules.mobility_manager import MobilityManager
from controllers.modules.crasher import Crasher
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
        self.mobility_manager: Optional[MobilityManager] = 0  
        self.sfc_manager: Optional[SFCManager] = 0
        self.crasher_manager: Optional[Crasher] = 0
        self.backup_manager: Optional[BackupManager] = 0
        # Status da Rede
        self.remaining_time = None
        self.update_interval = 1
        self.is_stopped = True
        
        # SFC (Service Function Chains)
        self.players_sfc_list = []
        self.sfc_queue = None
        self.timer_qeue_sfcs = []
        self.sfcs_crash_affected = {}

        # Implementações extras
        self.verbose = False
        self.log_file = "backup_log.txt"
        self.latency_interval = [7, 7]
        self.allow_high_latency = False
        self.altered_sfcs = {}
        # self.backups_data = {}

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
            if self.mobility_manager.activated:
                self.mobility_manager.start_simulation()
            
            _thread.start_new_thread(self.sequential_operation,())
            #self.sequential_operation(self)
        else:
            self.update()
            self.check_sfc_duration()

            # If mobility is activated
            if self.mobility_manager.activated:
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

    def send_back_to_qeue(self,sfc_list,changed_location=False,new_location=False,punishment=10):
        # A duração deve ser a mesma para as duas 
        sfc_id_duration = self.sfc_manager.sfc_id_duration
        first_sfc = sfc_list[0]
        duration = sfc_id_duration[first_sfc.id]["duration"]-(time.time()-sfc_id_duration[first_sfc.id]["timer"]) 
        new_sfc_list = []
        for sfc in sfc_list:
            sfc_id = sfc.id
            location = sfc.dst.substrate_node if new_location == False else new_location
            new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)
            if changed_location == True: # Checar depois se esse procedimento não é necessário também para Unique
                if re.search('cache', sfc_id) is not None: 
                    old_loc = str(sfc.dst.substrate_node)
                    new_loc = str(location)

                    ma_old_key = 'MA_region_' + old_loc
                    re_old_key = 'RE_region_' + old_loc
                    
                    old_routing_info = copy.deepcopy(self.sfc_manager.sfcs_routing_info)
                    #routing_info = self.sfc_manager.sfcs_routing_info
                    
                    if ma_old_key in old_routing_info[sfc_id].keys():
                        stored_info = old_routing_info[sfc_id][ma_old_key]
                        del self.sfc_manager.sfcs_routing_info[sfc_id][ma_old_key]
                        ma_new_key = re.sub(old_loc,new_loc,ma_old_key)
                        self.sfc_manager.sfcs_routing_info[sfc_id][ma_new_key] = stored_info
                        new_vnfs_list_dict[1]['name'] = ma_new_key

                    if re_old_key in old_routing_info[sfc.id].keys():
                        stored_info = old_routing_info[sfc_id][re_old_key]
                        del self.sfc_manager.sfcs_routing_info[sfc_id][re_old_key]
                        re_new_key = re.sub(old_loc, new_loc,re_old_key)
                        self.sfc_manager.sfcs_routing_info[sfc_id][re_new_key] = stored_info
                        new_vnfs_list_dict[2]['name'] = re_new_key
                    
                    src_path = old_routing_info[sfc_id]['src']
                    del self.sfc_manager.sfcs_routing_info[sfc_id]['src']
                    del self.sfc_manager.sfcs_routing_info[sfc_id]['dst']
                    self.sfc_manager.sfcs_routing_info[sfc_id]['src'] = src_path
                    self.sfc_manager.sfcs_routing_info[sfc_id]['dst'] = []

            new_sfc_dict = {}
            new_sfc_dict["name"] = sfc_id
            new_sfc_dict["vnf_list"] = new_vnfs_list_dict
            new_sfc_dict["bandwidth"] = sfc.input_throughput
            new_sfc_dict["src_node"] = sfc.src.substrate_node
            new_sfc_dict["dst_node"] = location
            new_sfc_dict["duration"] = duration
            new_sfc_dict["latency"] = sfc.latency_request
            self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network) # retira a sfc antig
            #self.backup_manager.take_off_backup_if_exist([sfc_id])    
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

    def output_results(self, results_dict, sfc_id,res_output=False) -> None:
        def output_network_resources(deploy_time):
            self.output_writter.output_cpu_utilization(self.substrate_network, deploy_time,self.crasher_manager.nodes_crashed)
            self.output_writter.output_cache_utilization(self.substrate_network, deploy_time,self.crasher_manager.nodes_crashed)
            self.output_writter.output_bandwidth_utilization(self.substrate_network, deploy_time)
            self.output_writter.output_nodes_sf_utilization(self.substrate_network, deploy_time)
            #self.output_utils.output_edges_sf_utilization(self.edges_vnf, self.existing_vnf, self.sfc_list, deploy_time, route_info, sfc)

        def output_flows(current_time, sfc_id, latency, run_duration, is_success,backup_sfc,alg_name=self.alg.name,fail_reason=None,latency_diff=None, wait_time=None):
            bw_transcode = 0
            self.output_writter.output_flows(
                self.substrate_network,
                wait_time,
                self.sfc_manager.get_running_players_sessions(),
                self.counter,
                self.remaining_time,
                current_time,
                sfc_id,
                latency,
                run_duration,
                is_success,
                fail_reason,
                backup_sfc,
                bw_transcode,
                latency_diff,
                len(self.crasher_manager.nodes_crashed)!=0
            )
        def resilient_output(sfc_id,info):
            self.output_writter.resilient_output(sfc_id,info)

        """Updates the network state and check for resource overhead."""
        if results_dict:
            # output of the simulation
            if not res_output:
                output_network_resources(deploy_time=results_dict['current_time'])
                output_flows(results_dict['current_time'],sfc_id,results_dict['latency'],
                            results_dict['run_duration'],results_dict['is_success'],
                           results_dict['backup_sfc'], results_dict['fail_reason'])
                
                if not results_dict['backup_sfc']:
                    self.success.append(results_dict['is_success'])
            
        sfcs_crash_aff = copy.deepcopy(list(self.sfcs_crash_affected.keys())) 
        if sfc_id in sfcs_crash_aff:
            if not self.sfcs_crash_affected[sfc_id]['backup_success'] == True: #
                time_to_recover = None
                latency_diff = None
                latency_factor = None
                resource_factor = None
                if results_dict:
                    self.sfcs_crash_affected[sfc_id]["recover_success"] = results_dict['is_success']
                    
                    if results_dict['is_success']:
                        time_to_recover = time.time() - self.sfcs_crash_affected[sfc_id]["fall_time"] #if results_dict['is_success'] else None
                        latency_diff = results_dict['latency'] - self.sfcs_crash_affected[sfc_id]["old_latency"] # if results_dict['is_success'] else None
                        latency_factor = (results_dict['latency'] - self.sfcs_crash_affected[sfc_id]["old_latency"])/self.sfcs_crash_affected[sfc_id]["old_latency"] #if results_dict['is_success'] else None
                        old_resource_reuse = self.sfcs_crash_affected[sfc_id]["resource_info"] # Taxa antiga
        
                        if old_resource_reuse != 0 and old_resource_reuse != 0.0:
                            resource_factor = (old_resource_reuse - results_dict['resource_info'])/old_resource_reuse
                        else:
                            resource_factor = 0#(old_resource_reuse - results_dict['resource_info'])/old_resource_reuse
                    if resource_factor != None:        
                        if resource_factor > 2:
                            resource_factor = 1
                        if resource_factor < -1:
                            resource_factor = -1
                    self.sfcs_crash_affected[sfc_id]["latency_diff"] = latency_diff
                    self.sfcs_crash_affected[sfc_id]["latency_degrad"] = latency_factor
                    self.sfcs_crash_affected[sfc_id]["resource_degrad"] = resource_factor
                    self.sfcs_crash_affected[sfc_id]["time_to_recover"] = time_to_recover
                else:
                    self.sfcs_crash_affected[sfc_id]["recover_success"] = False
                    self.sfcs_crash_affected[sfc_id]["time_to_recover"] = None
                    self.sfcs_crash_affected[sfc_id]["latency_diff"] = None

            info = copy.deepcopy(self.sfcs_crash_affected[sfc_id])
            resilient_output(sfc_id,info)
            del self.sfcs_crash_affected[sfc_id]
                    
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
    def backup_in_qeue(self):
        sfc_deque = self.sfc_queue.queue
        for index, sfc_list in enumerate(sfc_deque):
            for sfc in sfc_list:
                is_backup = (sfc.id.split("_")[2])
                if is_backup == 'backup':
                    return True
        return False


    def deploy_sfc(self, sfc: object) -> bool:
        # with self.lock:
        results_dict = self.sfc_manager.deploy_sfc(sfc,self.substrate_network)
        # if results_dict['backup_sfc']:
        #     if results_dict['is_success']:  
        #         try:
        #             self.backups_data[sfc.id]['status'] = 'instantiated'
        #         except:
        #             results_dict['is_success'] = False
        #             self.sfc_manager.undeploy_sfc(sfc.id,self.substrate_network)
        #     else:
        #         if sfc.id in self.backups_data:
        #             del self.backups_data[sfc.id]

        if results_dict != False:
            #if not results_dict['backup_sfc']: #and (results_dict['is_success'] == True):
            return results_dict
        return False  
                #self.mobility_manager.add_vehicle(sfc) # Se não é Backup e tiver sido um sucesso, adiciona um veículo para essa SFC;
            # else:
            #     self.mobility_manager.remove_sfc(sfc.id,all=True) # Se não remove ()
        # else:
        #     return False

    def sequential_submit_sfcs(self):
        self.check_timer_qeue()
        self.sequential_check_duration() 
        
        last_sf_mono = 'sfc_unique_p4_' + str(self.flows)
        last_sf_dec = 'sfc_mono_p4_' + str(self.flows)

        sfc_list = self.sfc_queue.peek_sfc()           
        print("queue_size: " + str(self.sfc_queue.qsize()))
        if self.max_queue_size < self.sfc_queue.qsize():
            self.max_queue_size = self.sfc_queue.qsize()
        last_sfc_release = False
        player_sfc_id_list = []

        deploy_log = {}  # Indicador para verificar se todas as SFCs foram implantadas com sucesso 
        all_success = True
        t_1 = time.time()
        for sfc in sfc_list:
            result_dict = self.deploy_sfc(sfc)
            deploy_log[sfc.id] = result_dict
            if not result_dict['is_success']:
                all_success = False
            player_sfc_id_list.append(sfc.id)
            #self.update()

        if not result_dict['backup_sfc']:
            if all_success:
                self.mobility_manager.add_vehicle(sfc_list)
                sfc_duration_timer = time.time()
                # for sfc in sfc_list:
                #     self.sfc_manager.sfc_id_duration[sfc.id] = {"duration":sfc.duration,"timer":sfc_duration_timer}
            else: # Se todas não foram um sucesso, retira quem foi um sucesso
                for sfc_id, result_dict in deploy_log.items():
                    if result_dict['is_success']:
                        deploy_log[sfc_id]['is_success'] = False
                        deploy_log[sfc_id]['latency'] = None
                        self.sfc_manager.undeploy_sfc(sfc_id, self.substrate_network)
                    self.mobility_manager.remove_sfc(sfc_id, all=True)
        
        for sfc_id, result_dict in deploy_log.items():
            self.output_results(sfc_id=sfc_id,results_dict=result_dict)
        
        self.players_sfc_list.append(player_sfc_id_list)
        self.update()
        
        t_2 = time.time()
        print(f"          algorithm {self.alg.name} take time: {round(t_2 - t_1,4)}")
        if sfc.id in (last_sf_mono, last_sf_dec):
            print('Last SFC released')
            print('Max queue size:', self.max_queue_size )
            last_sfc_release = True
            self.stop()
            sys.exit()
        return last_sfc_release


    def sequential_crasher(self,interval=25):
        #if not backup activated
        self.sequential_check_duration() 
        nodes_to_crash = self.crasher_manager.activate_crasher(self.substrate_network,self.sfc_manager,self.alg.name)
        self.sfc_manager.crashed_servers = self.crasher_manager.nodes_crashed
        print(f"Servidores Crashados: {nodes_to_crash}")
        
        vnfs_backup_instantiate = 0
        backups = self.sfc_manager.backup_manager.backups_sfc_instantiated
        
        vnfs_backup_util = 0
        if self.alg.name == 'ga':
            for backup_id,original_sfc in backups.items():
                sfc_backups = self.sfc_manager.backup_manager.sfcs_backups_instatiated[original_sfc]
                vnfs_backup_instantiate = len(sfc_backups) + vnfs_backup_instantiate
        else:
            vnfs_backup_instantiate = len(list(backups.keys())) * 4

        
        print("número de vnfs de backup  instanciadas: ",vnfs_backup_instantiate)
        # ------------Coleta das SFC's caídas---------------#
        sfcs_crashed_ids = {}
        for server in nodes_to_crash:
            server_info = self.substrate_network.get_node_sfc_vnf_list(server)
            self.substrate_network.set_node_cache_capacity(server, -0.0000001)
            self.substrate_network.set_node_cpu_capacity(server, -0.0000001)
            self.substrate_network.set_node_cache_free(server, 0)
            self.substrate_network.set_node_cpu_free(server, 0)
            for sfc_vnf in server_info: # Aqui serve para tirarmos as sfcs de backup da análise e também montarmos uam estrutura boa.
                if self.sfc_manager.is_Backup(sfc_vnf[0]):
                    self.sfc_manager.remove_backup_by_id(sfc_vnf[0],self.substrate_network)
                    continue
                if sfc_vnf[0] not in sfcs_crashed_ids:
                    sfcs_crashed_ids[sfc_vnf[0]] = []  # Inicializa a chave corretamente
                sfcs_crashed_ids[sfc_vnf[0]].append(sfc_vnf[1])  # Adiciona à lista existente

        
        # ------------Checagem do Backup---------------#
        sfcs_with_backup = []
        for sfc_id, vnfs in sfcs_crashed_ids.items():
            self.mobility_manager.remove_sfc(sfc_id)
            backup_found = self.sfc_manager.check_backups(sfc_id,vnfs,self.substrate_network)
            if backup_found:   
                o_rf = self.substrate_network.sfc_route_info[sfc_id]
                
                if self.alg.name == 'ga':
                    vnfs_backup_util += 1
                    new_rf = copy.deepcopy(o_rf)
                    vnf_id = backup_found['vnf_id'].removesuffix("_b") 
                    new_rf[vnf_id] =  backup_found['route_info'][backup_found['vnf_id']]
                    keys = list(new_rf)
                    if vnf_id in keys[:-1]:  
                        new_rf[keys[keys.index(vnf_id) + 1]] = backup_found['route_info']['source']
                    time_to_r = random.uniform(2,3)
                else:
                    vnfs_backup_util += 4
                    #self.sfc_manager.swap_sfcs(sfc_id,backup_found,self.substrate_network)
                    self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network,take_out_backup=False) # não retira o backup, pois ele foi ativado

                    new_rf = backup_found['route_info']
                    vnf_id = None
                    time_to_r = random.uniform(4,5)
                    
                o_latency = self.sfc_manager.calculate_latency(o_rf)
                new_latency = self.sfc_manager.calculate_latency(new_rf) + 1 # Compensar bug
                l_diff = new_latency - o_latency
                
                latency_degrad = l_diff/o_latency #if results_dict['is_success'] else None
                resource_degrad = 1
                backup_efficient = round((vnfs_backup_util/vnfs_backup_instantiate)*100,2)
                self.sfcs_crash_affected[sfc_id] = {"vnf_id":vnf_id,"recover_success":True,"backup_success":True,
                                                    "latency_diff":l_diff,"old_latency":o_latency,'latency_degrad':latency_degrad,'resource_degrad':resource_degrad,
                                                    "time_to_recover":time_to_r,'backup_efficient':backup_efficient}
                
                self.output_results(results_dict=False,sfc_id=sfc_id,res_output=True)     
                sfcs_with_backup.append(sfc_id)
        
        backup_efficient = round((vnfs_backup_util/vnfs_backup_instantiate)*100,2)
        print(backup_efficient)
        # Remove os SFCs que possuem backup do dicionário sfcs_crashed_ids
        for sfc_id in sfcs_with_backup:
            sfcs_crashed_ids.pop(sfc_id, None)  # Evita erro caso a chave já tenha sido removida
        
        # sfcs_list = []
        # for sfc_id in list(sfcs_crashed_ids.keys()):
        #     new_sfc_list = self.find_sfc_pair(self.players_sfc_list,sfc_id)
        #     sfcs_list.append(new_sfc_list)
        # sfcs_list = [list(t) for t in set(tuple(sublist) for sublist in sfcs_list)]
        novas = []
        for sfc_id in list(sfcs_crashed_ids.keys()):
            new_sfc_list = []
            sfc_id_duration = self.sfc_manager.sfc_id_duration 
            duration = sfc_id_duration[sfc_id]["duration"]-(time.time()-sfc_id_duration[sfc_id]["timer"]) + 3 # bonus

            # self.mobility_manager.set_vehicle_status(sfc_id,new_status='moving')         
            sfc = self.substrate_network.get_sfc_by_id(sfc_id)

            o_rf = self.substrate_network.sfc_route_info[sfc_id]
            o_latency = self.sfc_manager.calculate_latency(o_rf)
            
            resource_saved = 1#self.sfc_manager.sfc_reuse[sfc_id]
            resource_info = self.sfc_manager.calculate_latency(o_rf)

            location = sfc.dst.substrate_node
            
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

            self.sfcs_crash_affected[sfc_id] = {"vnf_id":None,"recover_success":None,"backup_success":False,"latency_diff":None,
                                                "old_latency":o_latency,"resource_info":resource_saved,"time_to_recover":None,
                                                'fall_time':time.time(),'backup_efficient':backup_efficient}
            
            novas.append(new_sfc_list)
            self.timer_qeue_sfcs.append({"new_sfc_list":new_sfc_list,"timer":time.time()})
        time.sleep(7)
        for sfc_list in novas:
            self.sfc_queue.put_begin(sfc_list)

    def find_sfc_pair(self,player_sfc_id_list, sfc_key):
        for sfc_list in player_sfc_id_list:
            if sfc_key in sfc_list:
                return sfc_list  # Retorna a lista onde a chave está presente
        return None  # Retorna None se a chave não for encontrada

    def sequential_recovery(self,interval=200):
        node = self.crasher_manager.nodes_crashed[0]
        self.substrate_network.reset_node_cpu_capacity(node,100)
        self.substrate_network.reset_node_cache_capacity(node,100)
        self.sfc_manager.crashed_servers =  self.crasher_manager.recover_from_crash(self.substrate_network)

    def check_timer_qeue(self):
        pass
        # if self.timer_qeue_sfcs:
        #     final_time = time.time()
        #     new_timer_qeue_sfcs = []  # Nova lista para armazenar itens que não atendem à condição
            
        #     for entry in self.timer_qeue_sfcs:
        #         time_elapsed = final_time - entry['timer']
        #         if time_elapsed >= random.uniform(3, 4):
        #             self.sfc_queue.put_begin(entry["new_sfc_list"])  # Coloca na fila
        #         else:
        #             new_timer_qeue_sfcs.append(entry)  # Mantém na lista se não atender à condição
            
        #     # Atualiza a lista original com os itens restantes
        #     self.timer_qeue_sfcs = new_timer_qeue_sfcs

    def sequential_backup(self, threshold=0.0):
        backups = self.sfc_manager.create_backups(self.substrate_network)

    def sequential_mobility(self, interval=5):
        self.sequential_check_duration() 
        if len(self.players_sfc_list) != 0: # Se não houver mais SFC's não faz nada
            sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes() # Checagem dos Veh que mudaram de posição 
            for sfc_list, new_location in zip(sfcs_moved, new_locations):
                print(f"SFCs moved: {sfc_list} | New Location: {new_location}")
                # try:
                obj_sfc_list = [self.substrate_network.get_sfc_by_id(sfc_id) for sfc_id in sfc_list]
                self.send_back_to_qeue(obj_sfc_list, changed_location=True, new_location=new_location)
                # except:
                #     continue

    def sequential_check_duration(self):
        remove_list = self.sfc_manager.check_sfc_duration()
        for sfc_id in remove_list:
            self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network)
           #self.mobility_manager.remove_sfc(sfc_id,all=True) # Remove o Veículo associado
            self.mobility_manager.remove_sfc(sfc_id)#(sfc_id,new_status='blocked')
            #self.backup_manager.take_off_backup_if_exist([sfc_id])
            # for backup_id in list(self.backups_data.keys()):
            #     if self.backups_data[backup_id]['original_sfc'] == sfc_id:
            #         del self.backups_data[backup_id]
        self.update()

    def sequential_operation(self):
        mobility_interval = 5
        crasher_interval = self.crasher_manager.fail_interval #260  # Intervalo em segundos para ativar o Crasher repetidamente
        crashs_trials = 0
        crash_limit = self.crasher_manager.number_of_fails 

        fail_recovery_time = (1000 - (self.crasher_manager.availability)*1000)*2 # fator de segurança
                
        if self.alg.name == 'msf' or self.alg.name == 'greedyb':
            backup_interval_creation = 25
        else:
            backup_interval_creation = 5
        
        start_timer = time.time()
        last_mobility_time = start_timer
        last_backup_time = start_timer
        last_crasher_time = start_timer

        i = 0
        
        while not self.is_stopped:
            self.sequential_submit_sfcs()
                       
            current_time = time.time()
            if self.mobility_manager.activated:
                decorrido = current_time - last_mobility_time
                if decorrido >= mobility_interval:    
                    self.sequential_mobility()
                    last_mobility_time = time.time()

            self.sequential_submit_sfcs()
            
            if self.sfc_manager.backup_manager.backup_activated:
                decorrido = current_time - last_backup_time
                if decorrido >= backup_interval_creation:
                    self.sequential_backup()  # criação de backups
                    last_backup_time = time.time()
            
            self.sequential_submit_sfcs()

            if self.crasher_manager.activated:
                #Recupera o sistema após o tempo de recuperação
                if self.crasher_manager.nodes_crashed != []:
                    recovery_elapsed = current_time - last_crasher_time
                    if recovery_elapsed >= fail_recovery_time:
                        self.sequential_recovery()  # Substitua pelo método de recuperação        

                if (current_time - last_crasher_time >= crasher_interval) and not self.backup_in_qeue() and crashs_trials < crash_limit:
                    #self.sequential_backup()
                    retorno = self.sequential_crasher()  
                    # if retorno != False:
                    #     self.sequential_submit_sfcs()
                    last_crasher_time = time.time()
                    crashs_trials = crashs_trials + 1 
            self.sequential_submit_sfcs()
            print(i)
            i = i + 1