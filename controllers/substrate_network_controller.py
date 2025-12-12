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
from controllers.modules.sfcs_instatiator import SFCInstatiator
from controllers.modules.sfcs_manager import SFCManager
from controllers.sfc_generator import SFCGenerator
from collections import deque
from typing import Optional
import itertools

from controllers.modules.mobility_manager import MobilityManager
from controllers.modules.crasher import Crasher
from controllers.sfc_queue import SFCQueue
from core.net import Net
from core.net_v2 import Net2
from utils.manager_results import OutputWritter
from utils.network_utils import EnergyCalculator

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
# logger.debug('debug message')
# logger.info('info message')
# logger.warn('warn message')
# logger.error('error message')
# logger.critical('critical message')

class SubstrateNetworkController():
    def __init__(self):
        # Rede
        self.substrate_network = Net2()
        self.node_info = {}

        # Modules
        self.mobility_manager: Optional[MobilityManager] = 0  
        self.sfc_manager: Optional[SFCManager] = 0
        self.sfc_instantiator: Optional[SFCInstatiator] = 0
        self.fail_manager: Optional[Crasher] = 0
        self.backup_manager: Optional[BackupManager] = 0
        self.energy_calculator = EnergyCalculator()


        # Status da Rede
        self.remaining_time = None
        self.update_interval = 1
        self.is_stopped = True
        
        # SFC (Service Function Chains)
        self.sfc_queue = None
        self.timer_qeue_sfcs = []
        self.sfcs_crash_affected = {}
        self.crashs_trials  = 0
        self.failure_schedule = []
        self.active_failures = []

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

    def start(self) -> None:
        """Starts the simulation."""

        if not self.is_stopped:
            self.is_stopped = True
            time.sleep(2*self.update_interval)

        self.is_stopped = False

        if self.mobility_manager.activated:
            self.mobility_manager.start_simulation()
        
        _thread.start_new_thread(self.sequential_operation,())
            #self.sequential_operation(self)

    def stop(self) -> None:
        """Stops the simulation."""
        self.is_stopped = True
        self.mobility_manager.stop_simulation()
        if self.timer:
            self.timer.cancel()
        sys.exit()
# 13 novo e 22 antigo
    def send_back_to_qeue(self,sfc_list,changed_location=False,new_location=False,punishment=10):
        # A duração deve ser a mesma para as duas 
        sfcs_tracker_info = self.sfc_manager.sfcs_tracker[sfc_list[0].dst_node]
        duration = sfcs_tracker_info["duration"]-(time.time()-sfcs_tracker_info["timer"]) 
        new_sfc_list = []
        for sfc in sfc_list:
            sfc_id = sfc.id
            location = new_location if changed_location else sfc.closer_router
            new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)
            
            if changed_location and 'cache' in sfc_id: # TODO Checar depois se esse procedimento não é necessário também para Unique
                old_loc = str(sfc.closer_router)
                new_loc = str(new_location)

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
            new_sfc_dict["dst_node"] = sfc.dst_node
            new_sfc_dict["duration"] = duration
            new_sfc_dict["closer_router"] = location
            new_sfc_dict["latency"] = sfc.latency_request
            #self.backup_manager.take_off_backup_if_exist([sfc_id])    
            new_sfc_list.append(new_sfc_dict)

        self.sfc_manager.undeploy_sfc(sfc_list[0].dst_node, self.substrate_network)

        new_sfcs = [SFCGenerator(sfc_dict).generate() for sfc_dict in new_sfc_list]

        # Define o tempo de entrada na fila AGORA
        current_enqueue_time = time.time()
        for sfc in new_sfcs:
            sfc.enqueue_time = current_enqueue_time

        self.sfc_queue.put_begin(new_sfcs)

    def update(self) -> None:
        """Updates the network state and check for resource overhead."""
        self.substrate_network.update()

    def output_results(self, results_dict, sfc_id,is_success,res_output=False, wait_time = None) -> None:
        current_time = time.time()
        def output_network_resources(current_time):
            self.output_writter.output_cpu_utilization(self.substrate_network, current_time,self.fail_manager.nodes_crashed)
            self.output_writter.output_gpu_utilization(self.substrate_network, current_time,self.fail_manager.nodes_crashed)
            self.output_writter.output_cache_utilization(self.substrate_network, current_time,self.fail_manager.nodes_crashed)
            self.output_writter.output_bandwidth_utilization(self.substrate_network, current_time)
            self.output_writter.output_nodes_sf_utilization(self.substrate_network, current_time)
            #self.output_utils.output_edges_sf_utilization(self.edges_vnf, self.current_time, self.sfc_list, deploy_time, route_info, sfc)

        def output_flows(current_time, sfc_id, latency, comp_latency, comm_latency,
                         run_duration, is_success,alg_name=self.alg,fail_reason=None,latency_diff=None, wait_time=None):
            bw_transcode = 0

            
            server_energy_consumption = self.energy_calculator.calculate_total_server_power(self.substrate_network)

            mobile_energy_consumption = self.energy_calculator.calculate_total_mobile_device_power(self.substrate_network)
            
            total_energy_consumption = server_energy_consumption + mobile_energy_consumption

            self.output_writter.output_flows(
                self.substrate_network,
                wait_time,
                self.sfc_manager.get_running_players_sessions(),
                self.counter,
                self.remaining_time,
                current_time,
                sfc_id,
                latency,
                comp_latency,  
                comm_latency, 
                run_duration,
                is_success,
                fail_reason,
                bw_transcode,
                self.substrate_network.get_acceptance_rate(self.success),
                server_energy_consumption, mobile_energy_consumption,total_energy_consumption,
                latency_diff,
                len(self.fail_manager.nodes_crashed)!=0
            )
        def resilient_output(sfc_id,info):
            self.output_writter.resilient_output(sfc_id,info,self.crashs_trials)

        """Updates the network state and check for resource overhead."""
        #if results_dict:
            # output of the simulation
        if not res_output:
            #if not results_dict['backup_sfc']:
            self.success.append(is_success)

            output_network_resources(current_time=current_time)
            output_flows(current_time,sfc_id,results_dict['latency'],
                        results_dict.get('comp_latency'),
                        results_dict.get('comm_latency'), 
                        results_dict['run_duration'],is_success,
                        wait_time=wait_time)
            
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

    def backup_in_qeue(self):
        sfc_deque = self.sfc_queue.queue
        for index, sfc_list in enumerate(sfc_deque):
            for sfc in sfc_list:
                is_backup = (sfc.id.split("_")[2])
                if is_backup == 'backup':
                    return True
        return False

    def deploy_sfc_list(self, sfc_list) -> bool:
        # with self.lock:
        mob_player_id = self.create_mobile_user(sfc_list)
        solution,is_success = self.sfc_instantiator.search_solution(sfc_list, self.substrate_network)
        if is_success:
            self.sfc_manager.submit_solution(sfc_list,solution,self.substrate_network)
        else:
            self.remove_mobile_user(mob_player_id)
        return solution,is_success

    def server_fail_operation(self) -> Tuple[List[str], list]:
        """Triggers a failure and returns the crashed nodes and affected SFCs."""
        servers_failed = self.fail_manager.activate_crasher(self.substrate_network,self.sfc_manager,self.alg)
        self.sfc_manager.crashed_servers = self.fail_manager.nodes_crashed
        print(f"Servidores Crashados: {servers_failed}")
        
        # ------------Coleta das SFC's caídas---------------#
        fallen_sfcs_list = []
        for server in servers_failed:
            mscs_affected = self.substrate_network.get_node_sfcs(server)
            tracker_id_list = [self.substrate_network.get_sfc_by_id(sfc_id).dst_node for sfc_id in mscs_affected]
            tracker_unique = list(set(tracker_id_list))
            
            for tracker_id in tracker_unique:
                self.mobility_manager.mark_vehicle_as_redeploying(tracker_id)
                sfc_list = self.sfc_manager.sfcs_tracker[tracker_id]['sfc_list']
                self.send_back_to_qeue(sfc_list, changed_location=False)
        
        for server in servers_failed:
            self.substrate_network.set_node_down(server) 
        return servers_failed, fallen_sfcs_list
   
    def recover_sfcs(self,fallen_sfcs_list):
        pass
        
        # vnfs_backup_instantiate = self.sfc_manager.backup_manager.get_backups_instantiated_q()
        # vnfs_backup_util = 0
        # # ------------Checagem do Backup---------------#
        # sfcs_with_backup = []
        # for sfc_id, vnfs in fallen_sfcs.items():
        #     self.mobility_manager.remove_sfc(sfc_id)
        #     backup_found = self.sfc_manager.check_backups(sfc_id,vnfs,self.substrate_network)
        #     if backup_found:   
        #         o_rf = self.substrate_network.sfc_route_info[sfc_id]
                
        #         if self.alg == 'ga':
        #             vnfs_backup_util += 1
        #             new_rf = copy.deepcopy(o_rf)
        #             vnf_id = backup_found['vnf_id'].removesuffix("_b") 
        #             new_rf[vnf_id] =  backup_found['route_info'][backup_found['vnf_id']]
        #             keys = list(new_rf)
        #             if vnf_id in keys[:-1]:  
        #                 new_rf[keys[keys.index(vnf_id) + 1]] = backup_found['route_info']['source']
        #             time_to_r = random.uniform(1,3)
        #         else:
        #             vnfs_backup_util += 4
        #             #self.sfc_manager.swap_sfcs(sfc_id,backup_found,self.substrate_network)
        #             self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network,take_out_backup=False) # não retira o backup, pois ele foi ativado

        #             new_rf = backup_found['route_info']
        #             vnf_id = None
        #             time_to_r = random.uniform(3,5)
                    
        #         o_latency = self.sfc_manager.calculate_latency(o_rf)
        #         new_latency = self.sfc_manager.calculate_latency(new_rf) + 1 # Compensar bug
        #         l_diff = new_latency - o_latency
                
        #         latency_degrad = l_diff/o_latency #if results_dict['is_success'] else None
        #         resource_degrad = 1
        #         backup_efficient = round((vnfs_backup_util/vnfs_backup_instantiate)*100,2)
        #         self.sfcs_crash_affected[sfc_id] = {"vnf_id":vnf_id,"recover_success":True,"backup_success":True,
        #                                             "latency_diff":l_diff,"old_latency":o_latency,'latency_degrad':latency_degrad,'resource_degrad':resource_degrad,
        #                                             "time_to_recover":time_to_r,'backup_efficient':backup_efficient}
                
        #         self.output_results(results_dict=False,sfc_id=sfc_id,res_output=True)     
        #         sfcs_with_backup.append(sfc_id)
        
        # backup_efficient = round((vnfs_backup_util/vnfs_backup_instantiate)*100,2)
        # print(backup_efficient)
        # # Remove os SFCs que possuem backup do dicionário sfcs_crashed_ids
        # for sfc_id in sfcs_with_backup:
        #     fallen_sfcs.pop(sfc_id, None)  # Evita erro caso a chave já tenha sido removida
        
        # # sfcs_to_qeue = []
        # for sfc_id in list(fallen_sfcs.keys()):
        #     sfc = self.substrate_network.get_sfc_by_id(sfc_id)

        #     o_rf = self.substrate_network.sfc_route_info[sfc_id]
        #     o_latency = self.sfc_manager.calculate_latency(o_rf)
            
        #     resource_saved = 1#self.sfc_manager.sfc_reuse[sfc_id]
        #     resource_info = self.sfc_manager.calculate_latency(o_rf)
        #     self.sfc_manager.undeploy_sfc(sfc_id,self.substrate_network) # retira a sfc antiga  
            
        #     self.sfcs_crash_affected[sfc_id] = {"vnf_id":None,"recover_success":None,"backup_success":False,"latency_diff":None,
        #                                         "old_latency":o_latency,"resource_info":resource_saved,"time_to_recover":None,
        #                                         'fall_time':time.time(),'backup_efficient':backup_efficient}
            
        #     self.timer_qeue_sfcs.append({"new_sfc_list":[sfc],"timer":time.time()})

    def server_recovery_operation(self, nodes_to_recover: List[str]):
        """Recovers a specific list of crashed nodes."""
        for node in nodes_to_recover:
            if node in self.fail_manager.nodes_crashed:
                # --- CORREÇÃO AQUI ---
                self.substrate_network.restore_node(node)
                # ---------------------
                self.fail_manager.nodes_crashed.remove(node)
                print(f"Recuperando servidor: {node}")
        # Atualiza a lista de servidores crashados no sfc_manager
        self.sfc_manager.crashed_servers = self.fail_manager.nodes_crashed

    def check_timer_qeue(self):
        if self.timer_qeue_sfcs:
            final_time = time.time()
            time_elapsed = final_time - self.timer_qeue_sfcs[0]['timer']
            if time_elapsed >= random.uniform(5,6):

                current_enqueue_time = time.time()

                for entry in self.timer_qeue_sfcs:
                    for sfc in entry["new_sfc_list"]:
                        sfc.enqueue_time = current_enqueue_time
                    self.sfc_queue.put_begin(entry["new_sfc_list"])  # Coloca na fila
                self.timer_qeue_sfcs = {}

    def check_mobility(self, interval=5):
        if self.sfc_manager.sfcs_tracker != {}:# Se não houver mais SFC's não faz nada
            sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes() # Checagem dos Veh que mudaram de posição 
            for sfc_list, new_location in zip(sfcs_moved, new_locations):
                print(f"SFCs moved: {sfc_list} | New Location: {new_location}")
                obj_sfc_list = [self.substrate_network.get_sfc_by_id(sfc_id) for sfc_id in sfc_list]
                self.send_back_to_qeue(obj_sfc_list, changed_location=True, new_location=new_location)

    def create_mobile_user(self,sfc_list):
        group_id = sfc_list[0].dst_node
        closer_router = sfc_list[0].closer_router
        sfc_id_list = [sfc.id for sfc in sfc_list]

        self.mobility_manager.add_player(group_id,closer_router,sfc_id_list)
        # TODO futuramente essa posição vai ser importante para que o algoritmo decida ativamente o roteador.
        distance = self.mobility_manager.get_md_distance_from_router(group_id,closer_router)
        
        # ips = 0.1 * random.uniform(0.1, 0.2)
        ips = 0.1
        self.substrate_network.add_node(group_id, 'mobile_device', cpu_capacity=25.00, cache_capacity=10.00,ips=ips,position=distance)
        return group_id
    
    def remove_mobile_user(self,sfc_list_id):
        self.mobility_manager.remove_player(sfc_list_id)
        self.substrate_network.remove_node(sfc_list_id)

    def check_duration(self):
        #TODO Continuar daqui#
        remove_list = []
        current_time = time.time()
        for sfc_list_id, info in list(self.sfc_manager.sfcs_tracker.items()):
            elapsed_time = current_time - info["timer"] 
            if elapsed_time >= info["duration"]: # Tempo do user acabou
                self.sfc_manager.undeploy_sfc(sfc_list_id,self.substrate_network)
                self.remove_mobile_user(sfc_list_id) # Definitivo
        pass
        #self.update()

    def sequential_operation(self):
        """
        Controla a execução sequencial das operações de mobilidade, backups e falhas.
        """
        self.initialize_timers()
        
        while not self.is_stopped:
            self.check_duration() 
            self.handle_mobility()
            self.handle_backups()
            self.handle_fails()
            processed_sfcs = self.submit_sfcs()
            if self.check_simulation_end(processed_sfcs):
                self.stop()
            self.iteration_counter += 1

    def initialize_timers(self):
        """Inicializa variáveis de controle e intervalos de tempo."""
        self.mobility_interval = 5
        self.backup_interval_creation = 40 if self.alg in ['msf', 'greedyb'] else 5
        
        self.start_time = time.time()
        self.last_mobility_time = self.start_time
        self.last_backup_time = self.start_time
        
        # Converte a lista de falhas para uma deque para processamento eficiente
        self.failure_schedule = deque(self.failure_schedule)
        
        self.iteration_counter = 0

    def handle_mobility(self):
        """Gerencia a mobilidade de acordo com o intervalo definido."""
        if self.mobility_manager.activated:
            if time.time() - self.last_mobility_time >= self.mobility_interval:
                self.check_mobility() # Verifica se os usuários mudaram de servidor
                self.last_mobility_time = time.time()

    def handle_backups(self):
        """Gerencia a criação de backups."""
        if self.sfc_manager.backup_manager.backup_activated:
            if time.time() - self.last_backup_time >= self.backup_interval_creation:
                self.sfc_manager.create_backups(self.substrate_network)
                self.last_backup_time = time.time()

    def handle_fails(self):
        """Gerencia a ativação e recuperação de falhas (Nós e Links) com base no cronograma."""
        if not self.fail_manager.activated:
            return

        elapsed_time = time.time() - self.start_time

        # -------------------------------------------------
        # 1. Lidar com recuperações pendentes (Nós e Links)
        # -------------------------------------------------
        # Itera sobre uma cópia (list(...)), pois podemos remover itens da lista original durante o loop
        for failure in list(self.active_failures):
            if elapsed_time >= failure['recovery_time']:
                
                # --- Recuperação de Nó ---
                if failure.get('type') == 'node':
                    # Suporta chave 'nodes' (legado) ou 'target'
                    nodes_to_recover = failure.get('nodes', failure.get('target'))
                    self.server_recovery_operation(nodes_to_recover=nodes_to_recover)
                
                # --- Recuperação de Link ---
                elif failure.get('type') == 'link':
                    self.link_recovery_operation(link_tuple=failure['target'])
                
                # Remove da lista de falhas ativas após recuperar
                self.active_failures.remove(failure)

        # -------------------------------------------------
        # 2. Lidar com novas falhas agendadas
        # -------------------------------------------------
        if self.failure_schedule and elapsed_time >= self.failure_schedule[0]['start']:
            event = self.failure_schedule.popleft()
            
            # === Tipo: NÓ (Server) ===
            if event['type'] == 'node':
                start_time = event['start']
                duration = event['duration']
                
                print(f"\n>>> [EVENTO] Falha de NÓ agendada para {start_time:.2f}s (Duração: {duration:.2f}s).")

                # Executa a lógica de falha de servidor (Roleta de servidores)
                crashed_nodes, sfcs_affected = self.server_fail_operation()
                
                # Tenta recuperar/logar SFCs afetadas (específico para lógica de nós/backup)
                self.recover_sfcs(sfcs_affected)

                # Agenda a recuperação apenas se a duração for positiva
                if duration > 0 and crashed_nodes: 
                    recovery_time = elapsed_time + duration
                    self.active_failures.append({
                        'type': 'node',
                        'nodes': crashed_nodes,   # Mantém compatibilidade
                        'target': crashed_nodes,  # Padrão novo
                        'recovery_time': recovery_time
                    })
            
            # === Tipo: LINK ===
            elif event['type'] == 'link':
                start_time = event['start']
                duration = event['duration']
                
                print(f"\n>>> [EVENTO] Falha de LINK agendada para {start_time:.2f}s (Duração: {duration:.2f}s).")
                
                # Executa a lógica de falha de link (Roleta de links)
                link_crashed, _ = self.link_fail_operation()
                
                # Agenda a recuperação apenas se um link foi derrubado e duração > 0
                if link_crashed and duration > 0:
                    recovery_time = elapsed_time + duration
                    self.active_failures.append({
                        'type': 'link',
                        'target': link_crashed, # Tupla (u, v)
                        'recovery_time': recovery_time
                    })

    def should_trigger_fail(self):
        """Verifica se uma nova falha pode ser ativada.
            Exemplo: Limite da falhas e tempo entre falhas """
        return (
            time.time() - self.last_crasher_time >= self.crasher_interval
            and self.crashs_trials < self.crash_limit)

    def check_simulation_end(self,sfc_list):
        last_sf_mono = 'sfc_unique_p4_' + str(self.flows)
        last_sf_dec = 'sfc_mono_p4_' + str(self.flows)
        for sfc_id  in sfc_list:
            if sfc_id in (last_sf_mono, last_sf_dec):
                print('Last SFC released')
                print('Max queue size:', self.max_queue_size )
                return True
        return False
    
    def submit_sfcs(self):
        """Executa a submissão de SFCs."""
        self.check_timer_qeue()

        if self.max_queue_size < self.sfc_queue.qsize():
            self.max_queue_size = self.sfc_queue.qsize()
        processed_sfcs = []
        
        start_time = time.time()
        while self.sfc_queue.qsize() != 0:
            if time.time() - start_time > 1: # se passou 1s, sair do loop
                break
            
            sfc_list = self.sfc_queue.peek_sfc()

            # --- calculo tempo de fila ---
            dequeue_time = time.time()
            wait_time = 0 # Valor padrão

            # Verifica se o atributo 'enqueue_time' que criamos existe
            if hasattr(sfc_list[0], 'enqueue_time'):
                wait_time = dequeue_time - sfc_list[0].enqueue_time
            
            # (Opcional) Fallback para SFCs iniciais que podem não ter passado
            # pelas funções acima, mas têm 'arrival_time'
            elif hasattr(sfc_list[0], 'arrival_time'):
                wait_time = dequeue_time - sfc_list[0].arrival_time
            # ---------------------------                                  
            for sfc in sfc_list:
                if sfc.dst_node in self.sfc_manager.sfcs_tracker:
                    raise ValueError(f"SFC já submetida")

            log,is_success = self.deploy_sfc_list(sfc_list)
            
            for sfc_id, result_dict in log.items():
                processed_sfcs.append(sfc_id)
                self.output_results(sfc_id=sfc_id,results_dict=result_dict,
                                    is_success=is_success, wait_time=wait_time)    
        return processed_sfcs
    

    def link_fail_operation(self):
        """Executa o ciclo de vida de uma falha de link."""
        
        # 1. Roleta: Escolhe qual link falhar baseado no stress
        link_failed = self.fail_manager.activate_link_crasher(self.substrate_network)
        
        if not link_failed:
            print(">>> [CRASHER] Nenhum link disponível para falhar (Roleta vazia).")
            return None, []

        u, v = link_failed
        print(f">>> [CRASHER] Link Sorteado na Roleta: {u} <-> {v}")

        # 2. Identifica SFCs afetadas (quem passa por esse link?)
        affected_sfcs = []
        # Itera sobre o routing info de todas as SFCs
        for sfc_id, routing_info in self.sfc_manager.sfcs_routing_info.items():
            path_broken = False
            for vnf, path in routing_info.items():
                if vnf in ['src', 'dst'] or not path: 
                    continue
                
                # Verifica se o par {u, v} está contido em algum segmento do caminho
                for i in range(len(path) - 1):
                    node_a, node_b = path[i], path[i+1]
                    # Verifica se é o link crashado (independente da ordem)
                    if {node_a, node_b} == {u, v}:
                        path_broken = True
                        break
                if path_broken:
                    break
            
            if path_broken:
                affected_sfcs.append(sfc_id)

        # 3. Derruba o link na topologia
        self.substrate_network.set_link_down(u, v)

        # 4. Trata as SFCs afetadas (Reenfileira para re-roteamento)
        fallen_sfcs_objects = []
        
        # Agrupa por Session/Group ID para usar a lógica de tracker existente
        affected_groups = set()
        for sfc_id in affected_sfcs:
            try:
                # Assume que sfc_id existe no tracking. Se não, ignora.
                sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                affected_groups.add(sfc_obj.dst_node)
            except:
                pass

        for group_id in affected_groups:
            if group_id in self.sfc_manager.sfcs_tracker:
                self.mobility_manager.mark_vehicle_as_redeploying(group_id)
                sfc_list = self.sfc_manager.sfcs_tracker[group_id]['sfc_list']
                
                # Envia de volta para a fila. 
                # Como o link está com latência INFINITA, o algoritmo buscará outra rota.
                self.send_back_to_qeue(sfc_list, changed_location=False)
                fallen_sfcs_objects.extend(sfc_list)

        return link_failed, fallen_sfcs_objects
    
    def link_recovery_operation(self, link_tuple):
        """Recupera o link físico."""
        u, v = link_tuple
        self.substrate_network.restore_link(u, v)
        print(f">>> [RECOVERY] Link Restaurado: {u} <-> {v}")
