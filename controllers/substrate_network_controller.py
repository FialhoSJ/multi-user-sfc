"""
Substrate Network Controller
Summary: Manages the simulation lifecycle, including SFC deployment, mobility, 
failures, backups, and results output.
"""

# --- Standard Library Imports ---
import sys
import re
import time
import logging
import copy
import random
import threading
import _thread
from queue import Queue
from collections import deque
from typing import Tuple, List, Optional

# --- Third Party Imports ---
import numpy as np

# --- Local Module Imports ---
from core.net import Net
from core.net_v2 import Net2
from controllers.sfc_generator import SFCGenerator
from controllers.sfc_queue import SFCQueue
from controllers.modules.backup_manager import BackupManager
from controllers.modules.sfcs_instatiator import SFCInstatiator
from controllers.modules.sfcs_manager import SFCManager
from controllers.modules.mobility_manager import MobilityManager
from controllers.modules.crasher import Crasher
from utils.manager_results import OutputWritter
from utils.network_utils import EnergyCalculator
from algorithms.environments.environment import SFC_AllocationEnv

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('./logs/substrate_network_controller.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class SubstrateNetworkController():
    def __init__(self):
        # --- Network Initialization ---
        self.substrate_network = Net2()
        self.node_info = {}

        # --- Modules ---
        # Initialized as 0 based on original code, likely acting as placeholders until instantiation
        self.mobility_manager: Optional[MobilityManager] = 0  
        self.sfc_manager: Optional[SFCManager] = 0
        self.sfc_instantiator: Optional[SFCInstatiator] = 0
        self.fail_manager: Optional[Crasher] = 0
        self.backup_manager: Optional[BackupManager] = 0
        self.energy_calculator = EnergyCalculator()
        self.output_writter: Optional[OutputWritter] = None

        # --- Simulation Status ---
        self.remaining_time = None
        self.update_interval = 1
        self.is_stopped = True
        self.start_time = 0
        self.iteration_counter = 0
        
        # --- Timers & Intervals ---
        self.timer = None
        self.mobility_interval = 5
        self.backup_interval_creation = 5
        self.last_mobility_time = 0
        self.last_backup_time = 0
        self.last_crasher_time = 0
        self.crasher_interval = 0 

        # --- SFC & Queue Management ---
        self.sfc_queue = None
        self.timer_qeue_sfcs = []
        self.max_queue_size = 0
        self.flows = 0
        self.alg = None
        
        # --- Failure & Recovery State ---
        self.sfcs_crash_affected = {}
        self.crashs_trials  = 0
        self.crash_limit = 0
        self.failure_schedule = []
        self.active_failures = []

        # --- Statistics & Logging ---
        self.success = []
        self.counter = 0
        self.verbose = False
        self.log_file = "backup_log.txt"
        
        # --- Configuration/Extras ---
        self.latency_interval = [7, 7]
        self.allow_high_latency = False
        self.altered_sfcs = {}
        self.lock = threading.Lock()

    ###########################################################################
    #                           LIFECYCLE METHODS                             #
    ###########################################################################

    def start(self) -> None:
        """Starts the simulation."""
        if not self.is_stopped:
            self.is_stopped = True
            time.sleep(2 * self.update_interval)

        self.is_stopped = False

        if self.mobility_manager.activated:
            self.mobility_manager.start_simulation()
        
        _thread.start_new_thread(self.sequential_operation, ())

    def stop(self) -> None:
        """Stops the simulation."""
        self.is_stopped = True
        self.mobility_manager.stop_simulation()
        if self.timer:
            self.timer.cancel()
        sys.exit()

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

    ###########################################################################
    #                       MAIN LOOP & ORCHESTRATION                         #
    ###########################################################################

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
        """Gerencia o ciclo de vida (Crash -> Espera -> Recovery) baseado no Schedule."""
        if not self.fail_manager.activated:
            return

        elapsed_time = time.time() - self.start_time

        # 1. Checar Recuperações Pendentes (Lista Ativa)
        # Usamos uma cópia da lista [:] para poder remover itens seguramente durante iteração
        for failure in self.active_failures[:]:
            if elapsed_time >= failure['recovery_time']:
                if failure['type'] == 'node':
                    self.server_recovery_operation(failure['target'])
                elif failure['type'] == 'link':
                    self.link_recovery_operation(failure['target'])
                
                self.active_failures.remove(failure)

        # 2. Checar Novas Falhas Agendadas (Schedule do Main)
        if self.failure_schedule and elapsed_time >= self.failure_schedule[0]['start']:
            event = self.failure_schedule.popleft() 
            duration = event['duration'] 
            
            if event['type'] == 'node':
                crashed_nodes, _ = self.server_fail_operation()
                # Agenda recuperação APENAS se houver nós derrubados e duração > 0
                if crashed_nodes and duration > 0:
                    recovery_time = elapsed_time + duration
                    self.active_failures.append({
                        'type': 'node',
                        'target': crashed_nodes,
                        'recovery_time': recovery_time
                    })
                    print(f"   -> Recuperação agendada para T={recovery_time:.2f}s (Daqui a {duration}s)")

            elif event['type'] == 'link':
                link_crashed, _ = self.link_fail_operation()
                if link_crashed and duration > 0:
                    recovery_time = elapsed_time + duration
                    self.active_failures.append({
                        'type': 'link',
                        'target': link_crashed,
                        'recovery_time': recovery_time
                    })
                    print(f"   -> Recuperação agendada para T={recovery_time:.2f}s")

    def update(self) -> None:
        """Updates the network state and check for resource overhead."""
        self.substrate_network.update()

    ###########################################################################
    #                   QUEUE & DEPLOYMENT MANAGEMENT                         #
    ###########################################################################
    
    # Na classe SubstrateNetworkController

    def ensure_reliability_target(self, sfc_list, target_reliability=0.99):
        """
        Loop Proativo: Verifica confiabilidade e implanta réplicas usando DRL se necessário.
        """
        if not self.sfc_manager.backup_manager.backup_activated:
            return

        for sfc in sfc_list:
            # 1. Verifica confiabilidade atual
            current_r, weak_vnf, weak_node = self.sfc_manager.calculate_sfc_reliability(sfc.id, self.substrate_network)
            
            # Controle de Loop
            attempts = 0
            max_replicas = 2 # Evita loops infinitos
            
            while current_r < target_reliability and attempts < max_replicas:
                if not weak_vnf: break # Não deve acontecer se R < 1.0

                print(f"--- [PROACTIVE] SFC {sfc.id} Reliability: {current_r:.4f} < {target_reliability}. Replicating {weak_vnf}...")

                # 2. Cria Mini-SFC Contextual
                mini_sfc = self.sfc_manager.backup_manager.create_contextual_mini_sfc(
                    self.substrate_network, sfc, weak_vnf, weak_node
                )
                
                if not mini_sfc:
                    break

                # 3. Configura Ambiente DRL para esta tarefa específica
                # Reutilizamos o grafo mas criamos uma instância temporária de env
                # Importante: Nós válidos devem corresponder à visão do controlador principal
                env = SFC_AllocationEnv(
                    valid_nodes=[n for n in self.substrate_network.graph.nodes if self.substrate_network.graph.nodes[n]['type'] != 'router'],
                    list_graph=[self.substrate_network.graph],
                    list_sfc=[mini_sfc],
                    is_training=False
                )
                
                # 4. Aplica Mascaramento (Nó Proibido = Nó Primário)
                env.set_forbidden_nodes([weak_node])
                
                # 5. Executa Agente DRL (Kuririn)
                # Assumindo que self.sfc_instantiator.alg é a instância do Kuririn
                agent = self.sfc_instantiator.alg
                
                # Executa alocação (chama start_algorithm internamente que faz o loop de predição)
                # Precisamos adaptar já que start_algorithm geralmente recebe a configuração completa do env.
                # Aqui chamamos a lógica de decisão do agente diretamente usando o env que preparamos.
                success = agent.start_algorithm(env)

                if success:
                    # 6. Efetiva a Réplica
                    route_info = agent.get_route_info()
                    
                    # Converte para formato padrão esperado por deploy_sfc
                    self.substrate_network.deploy_sfc(mini_sfc, route_info)
                    
                    # Registra no BackupManager
                    if sfc.id not in self.sfc_manager.backup_manager.sfcs_backups_instatiated:
                         self.sfc_manager.backup_manager.sfcs_backups_instatiated[sfc.id] = []
                    
                    self.sfc_manager.backup_manager.sfcs_backups_instatiated[sfc.id].append({
                        "sfc_backup_id": mini_sfc.id,
                        "vnf_id": weak_vnf,
                        "route_info": route_info
                    })
                    
                    print(f"--- [PROACTIVE] Success! Replica for {weak_vnf} deployed at {route_info[weak_vnf+'_b'][0]}.")
                    
                    # Recalcula para próxima iteração
                    current_r, weak_vnf, weak_node = self.sfc_manager.calculate_sfc_reliability(sfc.id, self.substrate_network)
                    attempts += 1
                else:
                    print(f"--- [PROACTIVE] Failed to place replica for {weak_vnf}. Agent could not find solution.")
                    break

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
            # (Opcional) Fallback para SFCs iniciais
            elif hasattr(sfc_list[0], 'arrival_time'):
                wait_time = dequeue_time - sfc_list[0].arrival_time
            
            for sfc in sfc_list:
                if sfc.dst_node in self.sfc_manager.sfcs_tracker:
                    raise ValueError(f"SFC já submetida")

            log, is_success = self.deploy_sfc_list(sfc_list)
            
            for sfc_id, result_dict in log.items():
                processed_sfcs.append(sfc_id)
                self.output_results(sfc_id=sfc_id, results_dict=result_dict,
                                    is_success=is_success, wait_time=wait_time)    
        return processed_sfcs

    def deploy_sfc_list(self, sfc_list) -> bool:
        # with self.lock:
        mob_player_id = self.create_mobile_user(sfc_list)
        solution, is_success = self.sfc_instantiator.search_solution(sfc_list, self.substrate_network)
        if is_success:
            self.sfc_manager.submit_solution(sfc_list, solution, self.substrate_network)
            target_r = 0.99 
            self.ensure_reliability_target(sfc_list, target_reliability=target_r)
        else:
            self.remove_mobile_user(mob_player_id)
        return solution, is_success

    def send_back_to_qeue(self, sfc_list, changed_location=False, new_location=False, punishment=10):
        # A duração deve ser a mesma para as duas 
        sfcs_tracker_info = self.sfc_manager.sfcs_tracker[sfc_list[0].dst_node]
        duration = sfcs_tracker_info["duration"] - (time.time() - sfcs_tracker_info["timer"]) 
        new_sfc_list = []
        
        for sfc in sfc_list:
            sfc_id = sfc.id
            location = new_location if changed_location else sfc.closer_router
            new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)
            
            if changed_location and 'cache' in sfc_id: 
                old_loc = str(sfc.closer_router)
                new_loc = str(new_location)

                ma_old_key = 'MA_region_' + old_loc
                re_old_key = 'RE_region_' + old_loc
                
                old_routing_info = copy.deepcopy(self.sfc_manager.sfcs_routing_info)
                
                if ma_old_key in old_routing_info[sfc_id].keys():
                    stored_info = old_routing_info[sfc_id][ma_old_key]
                    del self.sfc_manager.sfcs_routing_info[sfc_id][ma_old_key]
                    ma_new_key = re.sub(old_loc, new_loc, ma_old_key)
                    self.sfc_manager.sfcs_routing_info[sfc_id][ma_new_key] = stored_info
                    new_vnfs_list_dict[1]['name'] = ma_new_key

                if re_old_key in old_routing_info[sfc.id].keys():
                    stored_info = old_routing_info[sfc_id][re_old_key]
                    del self.sfc_manager.sfcs_routing_info[sfc_id][re_old_key]
                    re_new_key = re.sub(old_loc, new_loc, re_old_key)
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

    def check_timer_qeue(self):
        if self.timer_qeue_sfcs:
            final_time = time.time()
            time_elapsed = final_time - self.timer_qeue_sfcs[0]['timer']
            if time_elapsed >= random.uniform(5, 6):
                current_enqueue_time = time.time()
                for entry in self.timer_qeue_sfcs:
                    for sfc in entry["new_sfc_list"]:
                        sfc.enqueue_time = current_enqueue_time
                    self.sfc_queue.put_begin(entry["new_sfc_list"])
                self.timer_qeue_sfcs = {}

    def backup_in_qeue(self):
        sfc_deque = self.sfc_queue.queue
        for index, sfc_list in enumerate(sfc_deque):
            for sfc in sfc_list:
                is_backup = (sfc.id.split("_")[2])
                if is_backup == 'backup':
                    return True
        return False

    ###########################################################################
    #                           MOBILITY LOGIC                                #
    ###########################################################################

    def check_mobility(self, interval=5):
        if self.sfc_manager.sfcs_tracker != {}: # Se não houver mais SFC's não faz nada
            # Checagem dos Veh que mudaram de posição
            sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes()  
            for sfc_list, new_location in zip(sfcs_moved, new_locations):
                print(f"SFCs moved: {sfc_list} | New Location: {new_location}")
                obj_sfc_list = [self.substrate_network.get_sfc_by_id(sfc_id) for sfc_id in sfc_list]
                self.send_back_to_qeue(obj_sfc_list, changed_location=True, new_location=new_location)

    def create_mobile_user(self, sfc_list):
        group_id = sfc_list[0].dst_node
        closer_router = sfc_list[0].closer_router
        sfc_id_list = [sfc.id for sfc in sfc_list]

        self.mobility_manager.add_player(group_id, closer_router, sfc_id_list)
        # TODO futuramente essa posição vai ser importante para que o algoritmo decida ativamente o roteador.
        distance = self.mobility_manager.get_md_distance_from_router(group_id, closer_router)
        
        # ips = 0.1 * random.uniform(0.1, 0.2)
        ips = 0.1
        self.substrate_network.add_node(group_id, 'mobile_device', cpu_capacity=25.00, cache_capacity=10.00, ips=ips, position=distance)
        return group_id
    
    def remove_mobile_user(self, sfc_list_id):
        self.mobility_manager.remove_player(sfc_list_id)
        self.substrate_network.remove_node(sfc_list_id)

    ###########################################################################
    #                      FAILURE & RECOVERY LOGIC                           #
    ###########################################################################

    def should_trigger_fail(self):
        """Verifica se uma nova falha pode ser ativada."""
        return (
            time.time() - self.last_crasher_time >= self.crasher_interval
            and self.crashs_trials < self.crash_limit)

    def server_fail_operation(self) -> Tuple[List[str], list]:
        """
        Dispara a falha e tenta recuperar via réplicas. 
        LOGA EXPLICITAMENTE SE A RECUPERAÇÃO FALHAR.
        """
        # 1. Executa a roleta do Crasher
        servers_failed = self.fail_manager.activate_crasher(
            self.substrate_network, 
            self.sfc_manager, 
            self.alg
        )
        
        if not servers_failed:
            return [], []

        print(f"\n>>> [CRASH] Servidores derrubados: {servers_failed}")
        
        # 2. Atualiza o estado da rede (físico)
        for server in servers_failed:
            self.substrate_network.set_node_down(server)

        # 3. Identifica SFCs afetadas
        fallen_sfcs_list = []
        processed_servers = set(servers_failed)
        
        # Dicionário para evitar duplicatas ao processar falhas
        processed_sfc_ids = set()

        for server in processed_servers:
            sfcs_in_node = self.substrate_network.get_node_sfcs(server)
            
            for sfc_id in sfcs_in_node:
                if sfc_id in processed_sfc_ids:
                    continue
                processed_sfc_ids.add(sfc_id)

                # --- TENTATIVA DE RECUPERAÇÃO ---
                recovered = self.sfc_manager.attempt_recovery_by_replica(
                    sfc_id, 
                    server, 
                    self.substrate_network
                )
                
                # --- INSTRUMENTAÇÃO PARA CSV DE RESILIÊNCIA ---
                # Prepara os dados para o log
                info_log = {
                    "recover_success": recovered,
                    "backup_success": recovered, # Se recuperou, o backup funcionou
                    "backup_efficient": 1 if recovered else 0,
                    "latency_diff": 0, # Calculado depois se sucesso
                    "time_to_recover": 0,
                    "vnf_id": "unknown", # Poderia refinar buscando qual VNF caiu
                    "latency_degrad": 0,
                    "resource_degrad": 0
                }
                
                # Se RECUPEROU: Adiciona à lista de monitoramento para logar métricas de latência no próximo ciclo
                if recovered:
                    self.sfcs_crash_affected[sfc_id] = {
                        "fall_time": time.time(),
                        "old_latency": 0, # Idealmente capturar latência antiga
                        "resource_info": 0,
                        "backup_success": True
                    }
                    # O log de sucesso será feito em output_results na próxima iteração
                
                # Se FALHOU (DROP): Loga IMEDIATAMENTE antes de destruir a SFC
                else:
                    self.output_writter.resilient_output(sfc_id, info_log, self.crashs_trials)
                    
                    # Lógica de Redeploy (envia para fila)
                    try:
                        sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                        tracker_id = sfc_obj.dst_node
                        
                        if tracker_id in self.sfc_manager.sfcs_tracker:
                            self.mobility_manager.mark_vehicle_as_redeploying(tracker_id)
                            sfc_list = self.sfc_manager.sfcs_tracker[tracker_id]['sfc_list']
                            
                            # Verifica se já não mandamos essa lista para redeploy
                            if sfc_list[0] not in fallen_sfcs_list:
                                self.send_back_to_qeue(sfc_list, changed_location=False)
                                fallen_sfcs_list.extend(sfc_list)
                    except Exception as e:
                        print(f"Erro ao processar falha da SFC {sfc_id}: {e}")

        # Atualiza lista de falhas no Manager
        self.sfc_manager.crashed_servers = self.fail_manager.nodes_crashed
        self.crashs_trials += 1 # Incrementa contador de trials
        
        return servers_failed, fallen_sfcs_list

    def server_recovery_operation(self, nodes_to_recover: List[str]):
        """Recupera nós específicos delegando ao Crasher."""
        if not nodes_to_recover:
            return

        recovered_count = 0
        for node in nodes_to_recover:
            # Chama o método seguro do Crasher
            if self.fail_manager.recover_specific_node(self.substrate_network, node):
                print(f">>> [RECOVERY] Servidor recuperado: {node}")
                recovered_count += 1
        
        # Sincroniza estado com manager se houve mudança
        if recovered_count > 0:
            self.sfc_manager.crashed_servers = self.fail_manager.nodes_crashed

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
        for sfc_id, routing_info in self.sfc_manager.sfcs_routing_info.items():
            path_broken = False
            for vnf, path in routing_info.items():
                if vnf in ['src', 'dst'] or not path: 
                    continue
                
                # Verifica se o par {u, v} está contido em algum segmento do caminho
                for i in range(len(path) - 1):
                    node_a, node_b = path[i], path[i+1]
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
        affected_groups = set()
        for sfc_id in affected_sfcs:
            try:
                sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                affected_groups.add(sfc_obj.dst_node)
            except:
                pass

        for group_id in affected_groups:
            if group_id in self.sfc_manager.sfcs_tracker:
                self.mobility_manager.mark_vehicle_as_redeploying(group_id)
                sfc_list = self.sfc_manager.sfcs_tracker[group_id]['sfc_list']
                self.send_back_to_qeue(sfc_list, changed_location=False)
                fallen_sfcs_objects.extend(sfc_list)

        return link_failed, fallen_sfcs_objects
    
    def link_recovery_operation(self, link_tuple):
        """Recupera o link físico."""
        u, v = link_tuple
        self.substrate_network.restore_link(u, v)
        print(f">>> [RECOVERY] Link Restaurado: {u} <-> {v}")

    def recover_sfcs(self, fallen_sfcs_list):
        """
        Placeholder/Legacy code for recovering specific SFCs.
        Currently inactive in the flow.
        """
        pass
        # Original logic commented out to preserve file structure without cluttering
        # (See original file for the commented block if restoration is needed)

    ###########################################################################
    #                      MAINTENANCE & REPORTING                            #
    ###########################################################################

    def check_duration(self):
        # TODO Continuar daqui
        remove_list = []
        current_time = time.time()
        for sfc_list_id, info in list(self.sfc_manager.sfcs_tracker.items()):
            elapsed_time = current_time - info["timer"] 
            if elapsed_time >= info["duration"]: # Tempo do user acabou
                self.sfc_manager.undeploy_sfc(sfc_list_id, self.substrate_network)
                self.remove_mobile_user(sfc_list_id) # Definitivo
        pass

    def check_simulation_end(self, sfc_list):
        last_sf_mono = 'sfc_unique_p4_' + str(self.flows)
        last_sf_dec = 'sfc_mono_p4_' + str(self.flows)
        for sfc_id  in sfc_list:
            if sfc_id in (last_sf_mono, last_sf_dec):
                print('Last SFC released')
                print('Max queue size:', self.max_queue_size )
                return True
        return False

    def output_results(self, results_dict, sfc_id, is_success, res_output=False, wait_time=None) -> None:
        current_time = time.time()
        
        def output_network_resources(current_time):
            self.output_writter.output_cpu_utilization(self.substrate_network, current_time, self.fail_manager.nodes_crashed)
            self.output_writter.output_gpu_utilization(self.substrate_network, current_time, self.fail_manager.nodes_crashed)
            self.output_writter.output_cache_utilization(self.substrate_network, current_time, self.fail_manager.nodes_crashed)
            self.output_writter.output_bandwidth_utilization(self.substrate_network, current_time)
            self.output_writter.output_nodes_sf_utilization(self.substrate_network, current_time)

        def output_flows(current_time, sfc_id, latency, comp_latency, comm_latency,
                         run_duration, is_success, alg_name=self.alg, fail_reason=None, latency_diff=None, wait_time=None):
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
                server_energy_consumption, mobile_energy_consumption, total_energy_consumption,
                latency_diff,
                len(self.fail_manager.nodes_crashed) != 0
            )

        def resilient_output(sfc_id, info):
            self.output_writter.resilient_output(sfc_id, info, self.crashs_trials)

        # Updates the network state and check for resource overhead.
        if not res_output:
            self.success.append(is_success)
            output_network_resources(current_time=current_time)
            output_flows(current_time, sfc_id, results_dict['latency'],
                        results_dict.get('comp_latency'),
                        results_dict.get('comm_latency'), 
                        results_dict['run_duration'], is_success,
                        wait_time=wait_time)
            
        sfcs_crash_aff = copy.deepcopy(list(self.sfcs_crash_affected.keys())) 
        if sfc_id in sfcs_crash_aff:
            if not self.sfcs_crash_affected[sfc_id]['backup_success'] == True:
                time_to_recover = None
                latency_diff = None
                latency_factor = None
                resource_factor = None
                
                if results_dict:
                    self.sfcs_crash_affected[sfc_id]["recover_success"] = results_dict['is_success']
                    if results_dict['is_success']:
                        time_to_recover = time.time() - self.sfcs_crash_affected[sfc_id]["fall_time"] 
                        latency_diff = results_dict['latency'] - self.sfcs_crash_affected[sfc_id]["old_latency"] 
                        latency_factor = (results_dict['latency'] - self.sfcs_crash_affected[sfc_id]["old_latency"])/self.sfcs_crash_affected[sfc_id]["old_latency"] 
                        old_resource_reuse = self.sfcs_crash_affected[sfc_id]["resource_info"] 
        
                        if old_resource_reuse != 0 and old_resource_reuse != 0.0:
                            resource_factor = (old_resource_reuse - results_dict['resource_info'])/old_resource_reuse
                        else:
                            resource_factor = 0
                    
                    if resource_factor is not None:        
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
            resilient_output(sfc_id, info)
            del self.sfcs_crash_affected[sfc_id]
                    
        if self.verbose:
            print("__________________________________________")
            self.output_writter.print_output_info(self.substrate_network, self.success) 
            print("")