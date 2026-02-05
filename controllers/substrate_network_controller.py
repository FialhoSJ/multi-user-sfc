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
from algorithms.environments.env_sbrc import SFC_AllocationEnv

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
        self.players = 6
        
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
        """Gerencia a criação de backups delegando ao Manager."""
        # Se o backup não estiver ativo na config, sai
        if not self.sfc_manager.backup_manager.backup_activated:
            return

        # Verifica o intervalo de tempo
        if time.time() - self.last_backup_time >= self.backup_interval_creation:
            
            # --- MUDANÇA: Identifica o Agente (se houver) ---
            agent = None
            if self.alg == 'SBRCMASKABLEPPO':
                agent = self.sfc_instantiator.alg
            
            # Repassa a responsabilidade (e o agente) para o SFC Manager
            self.sfc_manager.create_backups(self.substrate_network, agent_ref=agent)
            
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
    
    def check_network_health(self):
        """Verifica se a carga da rede está abaixo de 75%."""
        utilization = self.substrate_network.get_processing_network_used_precise()
        return utilization < 0.75

    def ensure_reliability_target(self, sfc_list, target_reliability=0.99):
        """
        Garante a confiabilidade alvo gastando o MÍNIMO de recursos possível,
        aplicando a regra de 'Grupo Atômico' para nós consolidados.
        """
        # Travas de segurança e saúde da rede
        if self.alg != 'SBRCMASKABLEPPO': return
        if not self.sfc_manager.backup_manager.backup_activated: return
        if not self.check_network_health(): return

        for sfc in sfc_list:
            # Loop de Tentativas (Evita loops infinitos se não conseguir atingir a meta)
            # Geralmente 1 ou 2 iterações bastam para corrigir o nó gargalo.
            max_iterations = 3 
            for _ in range(max_iterations):
                
                # 1. Mede a Confiabilidade Atual
                current_r, weak_vnf_id, weak_node = self.sfc_manager.calculate_sfc_reliability(sfc.id, self.substrate_network)
                
                # [ECONOMIA MÁXIMA] Se já bateu a meta, PARE AGORA. Não gaste mais nada.
                if current_r >= target_reliability:
                    break
                
                # Se não tem nó fraco identificado (erro de topologia?), aborta.
                if not weak_node:
                    break

                # 2. Identifica o 'Grupo de Risco' (Todas as VNFs neste nó fraco específico)
                vnfs_no_no_fraco = []
                route_info = self.sfc_manager.sfcs_routing_info.get(sfc.id, {})
                
                for vnf_id, path in route_info.items():
                    if vnf_id in ['src', 'dst']: continue
                    # Verifica se a VNF está hospedada EXATAMENTE no nó problemático
                    if path and path[0] == weak_node:
                        vnfs_no_no_fraco.append(vnf_id)

                # Se por algum motivo a lista estiver vazia, evita loop infinito
                if not vnfs_no_no_fraco:
                    break

                # 3. Aplica a correção ATÔMICA (Conserta o nó inteiro de uma vez)
                backup_created_in_cycle = False
                
                for target_vnf in vnfs_no_no_fraco:
                    # Verifica se JÁ existe backup para essa VNF específica
                    # (Evita duplicar backup para a mesma VNF)
                    already_has_backup = False
                    if sfc.id in self.sfc_manager.backup_manager.sfcs_backups_instatiated:
                        for backup in self.sfc_manager.backup_manager.sfcs_backups_instatiated[sfc.id]:
                            if backup['vnf_id'] == target_vnf:
                                already_has_backup = True
                                break
                    
                    if already_has_backup:
                        continue

                    # Criação da Mini-SFC de Backup
                    mini_sfc = self.sfc_manager.backup_manager.create_contextual_mini_sfc(
                        self.substrate_network, sfc, target_vnf, weak_node
                    )
                    
                    if mini_sfc:
                        # Configuração do Ambiente RL (SBRC)
                        all_servers = [n for n, d in self.substrate_network.graph.nodes(data=True) if d.get('type') != 'router']
                        all_servers.append(getattr(mini_sfc, 'mobile_node', None))
                        
                        env = SFC_AllocationEnv(
                            valid_nodes=all_servers,
                            list_graph=[self.substrate_network.graph],
                            list_sfc=[mini_sfc],
                            is_training=False
                        )
                        
                        # Proíbe o nó original (falho) para garantir diversidade
                        forbidden = [weak_node]
                        if isinstance(weak_node, (int, float)):
                            forbidden.append(weak_node - 0.1 if weak_node % 1 == 0.1 else weak_node + 0.1)
                        env.set_forbidden_nodes(forbidden)
                        
                        agent = self.sfc_instantiator.alg
                        agent.install_SFC(mini_sfc)
                        success = agent.start_algorithm(env)

                        if success:
                            route_info_backup = agent.get_route_info()
                            self.substrate_network.deploy_sfc(mini_sfc, route_info_backup)

                            
                            # Registra o Backup
                            if sfc.id not in self.sfc_manager.backup_manager.sfcs_backups_instatiated:
                                self.sfc_manager.backup_manager.sfcs_backups_instatiated[sfc.id] = []
                            
                            self.sfc_manager.backup_manager.sfcs_backups_instatiated[sfc.id].append({
                                "sfc_backup_id": mini_sfc.id,
                                "vnf_id": target_vnf,
                                "route_info": route_info_backup
                            })
                            backup_created_in_cycle = True
                            
                            if self.verbose:
                                print(f"[OTIMIZAÇÃO] VNF {target_vnf} protegida. Nó fraco: {weak_node}")

                # Se não conseguiu criar nenhum backup neste ciclo (falta de recursos?), pare para não travar.
                if not backup_created_in_cycle:
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
            # target_r = 0.95
            # self.ensure_reliability_target(sfc_list, target_reliability=target_r)
        else:
            self.remove_mobile_user(mob_player_id)
        return solution, is_success

    def send_back_to_qeue(self, sfc_list, changed_location=False, new_location=False, punishment=10):
        # A duração deve ser a mesma para as duas 
        sfcs_tracker_info = self.sfc_manager.sfcs_tracker.get(sfc_list[0].dst_node)
        
        # Proteção caso a SFC já tenha sido limpa do tracker
        if not sfcs_tracker_info:
            return

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
        if self.sfc_manager.sfcs_tracker != {}: 
            sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes()  
            
            for sfc_list, new_location in zip(sfcs_moved, new_locations):
                obj_sfc_list = []
                valid_move = True
                
                # 2. Tentamos buscar os objetos SFC na rede
                for sfc_id in sfc_list:
                    try:
                        # AQUI OCORRIA O ERRO: get_sfc_by_id falhava se a SFC tivesse sofrido undeploy
                        sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                        obj_sfc_list.append(sfc_obj)
                    except KeyError:
                        # 3. Tratamento Silencioso: 
                        # Se não achou (está no limbo/fila), apenas abortamos a mobilidade deste ciclo
                        # Não removemos do mobility, pois o usuário ainda existe.
                        if self.verbose:
                            print(f"[MOBILITY] Ignorando movimento da SFC {sfc_id} (SFC em recuperação/offline).")
                        valid_move = False
                        break # Aborta processamento desta lista específica
                
                # Só prossegue se TODAS as SFCs do usuário estiverem vivas na rede
                if valid_move and obj_sfc_list:
                    print(f"SFCs moved: {sfc_list} | New Location: {new_location}")
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
    
    def _calculate_sfc_path_latency(self, sfc_id: str) -> float:
        """
        Calcula a latência atual de uma SFC somando os delays dos links em sua rota.
        """
        if sfc_id not in self.substrate_network.sfc_route_info:
            return 0.0
            
        route_info = self.substrate_network.sfc_route_info[sfc_id]
        total_latency = 0.0
        
        # Percorre todos os caminhos definidos na rota (ex: src->vnf1, vnf1->vnf2...)
        # A estrutura esperada de route_info é {'vnf_id': [path_nodes], ...}
        for vnf, path in route_info.items():
            if not path or len(path) < 2:
                continue
            
            # Soma a latência de cada aresta no caminho
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if self.substrate_network.graph.has_edge(u, v):
                    # Tenta pegar 'latency' ou 'delay', assume 0 se não encontrar
                    edge_data = self.substrate_network.graph[u][v]
                    total_latency += edge_data.get('latency', edge_data.get('delay', 0))
        
        return total_latency

    def server_fail_operation(self) -> Tuple[List[str], list]:
        """
        Orquestra o processo completo de crash de servidores e análise de impacto.
        """

        servers_failed = self._trigger_crash()
        if not servers_failed:
            return [], []

        affected_sfc_ids, sfc_failed_nodes_map = self._map_affected_sfcs(servers_failed)

        high_risk, med_risk, low_risk = self._classify_crash_risk(
            servers_failed, affected_sfc_ids
        )

        pre_crash_latencies, sfc_owners_map = self._audit_pre_crash(affected_sfc_ids)

        self._crash_servers(servers_failed)

        fallen_sfcs_list, post_crash_latencies = self._recover_sfcs(
            affected_sfc_ids,
            sfc_failed_nodes_map,
            pre_crash_latencies,
            sfc_owners_map
        )

        self._compute_crash_metrics(
            servers_failed,
            affected_sfc_ids,
            high_risk,
            med_risk,
            low_risk,
            pre_crash_latencies,
            post_crash_latencies
        )

        self.sfc_manager.crashed_servers = self.fail_manager.nodes_crashed
        self.crashs_trials += 1

        return servers_failed, fallen_sfcs_list
    
    def _trigger_crash(self) -> List[str]:
        servers_failed = self.fail_manager.activate_crasher(
            self.substrate_network,
            self.sfc_manager,
            self.alg
        )

        if servers_failed:
            print(f"\n>>> [CRASH] Servidores a serem derrubados: {servers_failed}")

        return servers_failed
    
    def _map_affected_sfcs(self, servers_failed):
        affected_sfc_ids = set()
        sfc_failed_nodes_map = {}

        for server in servers_failed:
            sfcs_in_node = self.substrate_network.get_node_sfcs(server)
            for sfc_id in sfcs_in_node:
                affected_sfc_ids.add(sfc_id)
                sfc_failed_nodes_map.setdefault(sfc_id, []).append(server)

        return affected_sfc_ids, sfc_failed_nodes_map
    
    def _classify_crash_risk(self, servers_failed, affected_sfc_ids):
        high_risk = med_risk = low_risk = 0

        if not servers_failed:
            return high_risk, med_risk, low_risk

        target_node = servers_failed[0]
        base_id = str(target_node).split('.')[0]

        server_reliability = self.substrate_network.get_node_reliability(base_id)
        total_victims = len(affected_sfc_ids)

        if server_reliability < 0.90:
            high_risk = total_victims
        elif 0.90 <= server_reliability <= 0.95:
            med_risk = total_victims
        else:
            low_risk = total_victims

        print(f"   -> Crash Source Reliability: {server_reliability:.4f}")
        print(f"   -> Impact bucket: {'High' if high_risk else 'Med' if med_risk else 'Low'} Risk Node")

        return high_risk, med_risk, low_risk
    
    def _audit_pre_crash(self, affected_sfc_ids):
        pre_crash_latencies = {}
        sfc_owners_map = {}

        for sfc_id in affected_sfc_ids:
            try:
                sfc_obj_temp = self.substrate_network.get_sfc_by_id(sfc_id)
                sfc_owners_map[sfc_id] = sfc_obj_temp.dst_node
            except Exception:
                pass

            pre_crash_latencies[sfc_id] = self._calculate_sfc_path_latency(sfc_id)

        return pre_crash_latencies, sfc_owners_map
    
    def _crash_servers(self, servers_failed):
        for server in servers_failed:
            self.substrate_network.set_node_down(server)
            
    # Em controllers/substrate_network_controller.py

    def _recover_sfcs(self, affected_sfc_ids, sfc_failed_nodes_map, pre_crash_latencies, sfc_owners_map):
        fallen_sfcs_list = []
        post_crash_latencies = {}

        for sfc_id in sorted(affected_sfc_ids):
            # 1. Identificar o nó que falhou para esta SFC
            failed_nodes = sfc_failed_nodes_map.get(sfc_id, [])
            relevant_server_down = failed_nodes[0] if failed_nodes else None
            
            # 2. Snapshot: Salvar Objeto e Rota antes de destruir
            try:
                sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                old_route_info = copy.deepcopy(self.sfc_manager.sfcs_routing_info.get(sfc_id))
            except:
                # Se não conseguir pegar o objeto, não tem como recuperar
                continue

            # 3. UNDEPLOY IMEDIATO: Libera todos os recursos (Banda e CPU)
            # Isso garante que a rede esteja limpa para a tentativa de realocação
            self.sfc_manager.undeploy_sfc(sfc_owners_map.get(sfc_id), self.substrate_network, take_out_backup=False)

            # Marca o tempo inicial da tentativa
            if sfc_id in self.substrate_network.sfc_dict:
                debug = 1

            recovery_start_time = time.time()
            recovered = False

            # 4. Tenta reconstruir e fazer o Deploy novamente
            if relevant_server_down and old_route_info:
                # Chama a nova função de "Costura e Deploy"
                recovered = self.sfc_manager.reconstruct_and_redeploy(
                    sfc_obj, 
                    relevant_server_down, 
                    old_route_info, 
                    self.substrate_network
                )

            if recovered:
                # Sucesso: Calcula métricas
                new_lat = self._calculate_sfc_path_latency(sfc_id)
                post_crash_latencies[sfc_id] = new_lat
                old_lat = pre_crash_latencies.get(sfc_id, 0)
                
                self.sfcs_crash_affected[sfc_id] = {
                    "fall_time": time.time(),
                    "old_latency": old_lat,
                    "latency_diff": new_lat - old_lat,
                    "latency_degrad": new_lat - old_lat,
                    "time_to_recover": time.time() - recovery_start_time,
                    "resource_info": 0,
                    "resource_degrad": 0,
                    "backup_success": True,
                    "recover_success": True
                }

                print(f"")
            else:
                # 5. Falha no Backup: Manda para a fila (Redeploy completo/Migração)
                self.sfcs_crash_affected[sfc_id] = {
                    "fall_time": time.time(),
                    "old_latency": pre_crash_latencies.get(sfc_id, 0),
                    "resource_info": 0,
                    "backup_success": False,
                    "crash_trial": self.crashs_trials
                }
                
                tracker_id = sfc_owners_map.get(sfc_id)
                if tracker_id and tracker_id in self.sfc_manager.sfcs_tracker:
                    self.mobility_manager.mark_vehicle_as_redeploying(tracker_id)
                    sfc_list = self.sfc_manager.sfcs_tracker[tracker_id]['sfc_list']
                    
                    # Como já demos undeploy lá em cima (passo 3), 
                    # só precisamos mandar para a fila se a lista ainda existir
                    if sfc_list:
                        self.send_back_to_qeue(sfc_list, changed_location=False)
                        fallen_sfcs_list.extend(sfc_list)

        return fallen_sfcs_list, post_crash_latencies
    
    def _compute_crash_metrics(
        self,
        servers_failed,
        affected_sfc_ids,
        high_risk,
        med_risk,
        low_risk,
        pre_crash_latencies,
        post_crash_latencies
    ):
        total_affected = len(affected_sfc_ids)
        total_active_sfcs = len(self.substrate_network.sfc_dict)

        avg_lat_before = np.mean(list(pre_crash_latencies.values())) if pre_crash_latencies else 0.0
        avg_lat_after = np.mean(list(post_crash_latencies.values())) if post_crash_latencies else 0.0

        lat_diffs = [
            post_crash_latencies[sfc_id] - pre_crash_latencies[sfc_id]
            for sfc_id in post_crash_latencies
            if sfc_id in pre_crash_latencies
        ]

        avg_lat_diff = np.mean(lat_diffs) if lat_diffs else 0.0

        affected_pct = (total_affected / total_active_sfcs * 100) if total_active_sfcs > 0 else 0.0

        self.output_writter.output_crash_impact(
            self.crashs_trials,
            len(servers_failed),
            total_affected,
            high_risk,
            med_risk,
            low_risk,
            avg_lat_before,
            avg_lat_after,
            affected_pct,
            avg_lat_diff
        )


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

    ###########################################################################
    #                      MAINTENANCE & REPORTING                            #
    ###########################################################################

    def check_duration(self):
        # TODO Continuar daqui
        remove_list = []
        current_time = time.time()
        
        list_aux = list(self.sfc_manager.sfcs_tracker.items())
        for sfc_list_id, info in list_aux:
            elapsed_time = current_time - info["timer"] 
            if elapsed_time >= info["duration"]: # Tempo do user acabou
                self.sfc_manager.undeploy_sfc(sfc_list_id, self.substrate_network)
                self.remove_mobile_user(sfc_list_id) # Definitivo

        pass

    def check_simulation_end(self, sfc_list):
        """
        Verifica se a ÚLTIMA SFC da ÚLTIMA SESSÃO foi processada.
        """
        # [FIX 2] Constrói a ID baseada no ÚLTIMO player (ex: p6) e não hardcoded p4
        last_sf_mono = f'sfc_unique_p{self.players}_{self.flows}'
        last_sf_dec = f'sfc_mono_p{self.players}_{self.flows}'
        
        for sfc_id in sfc_list:
            # Verifica se a ID atual é exatamente a última esperada
            if sfc_id == last_sf_mono or sfc_id == last_sf_dec:
                print(f'[INFO] Last SFC released detected: {sfc_id}')
                print('Max queue size:', self.max_queue_size)
                
                # Pequeno sleep de segurança para garantir I/O de disco
                time.sleep(1) 
                return True
        return False
    

    def get_effective_system_reliability(self) -> float:
        """
        Calcula a confiabilidade média REAL do sistema.
        Considera:
        1. Rota Principal (Série)
        2. Backups Instanciados (Paralelo/Redundância)
        
        Fórmula por VNF: R_estagio = 1 - ((1 - R_main) * (1 - R_backup))
        """
        total_reliability = 0.0
        active_count = 0
        
        # Acesso seguro aos backups (dicionário {sfc_id: [lista_backups]})
        backups_dict = {}
        if self.sfc_manager and self.sfc_manager.backup_manager:
            backups_dict = self.sfc_manager.backup_manager.sfcs_backups_instatiated

        for sfc_id, sfc in self.substrate_network.sfc_dict.items():
            # 1. Filtros: Ignorar SFCs sem rota ou que sejam próprias de backup
            if sfc_id not in self.substrate_network.sfc_route_info:
                continue
            
            # Se a string 'backup' estiver no ID, é uma mini-SFC de proteção, não conta na média
            if "backup" in sfc_id: 
                continue

            route_info = self.substrate_network.sfc_route_info[sfc_id]
            
            # 2. Identificar onde cada VNF está rodando na rota PRINCIPAL
            # Estrutura típica route_info: {'src':..., 'vnf1': [node_id], ...}
            vnf_placement = {}
            for vnf_id, path in route_info.items():
                if vnf_id not in ['src', 'dst'] and path:
                    vnf_placement[vnf_id] = path[0]

            if not vnf_placement:
                continue

            # 3. Calcular confiabilidade da SFC (Estágio por Estágio)
            sfc_effective_rel = 1.0
            
            for vnf_id, main_node in vnf_placement.items():
                # Confiabilidade do Nó Principal
                r_main = self.substrate_network.get_node_reliability(main_node)
                r_backup = 0.0
                
                # Verificar se existe Backup para esta VNF específica
                if sfc_id in backups_dict:
                    for bk_info in backups_dict[sfc_id]:
                        # bk_info normalmente tem: {'vnf_id': '...', 'route_info': ...}
                        if bk_info.get('vnf_id') == vnf_id:
                            bk_route = bk_info.get('route_info', {})
                            # Descobrir em qual nó o backup está rodando
                            for bk_k, bk_path in bk_route.items():
                                if bk_k not in ['src', 'dst'] and bk_path:
                                    bk_node = bk_path[0]
                                    r_backup = self.substrate_network.get_node_reliability(bk_node)
                                    break
                
                # CÁLCULO DA REDUNDÂNCIA (PARALELO)
                # Se r_backup for 0 (sem backup), a fórmula vira apenas r_main.
                # Se tiver backup, a confiabilidade sobe drasticamente.
                stage_rel = 1.0 - ((1.0 - r_main) * (1.0 - r_backup))
                
                # Multiplica pela confiabilidade acumulada da SFC
                sfc_effective_rel *= stage_rel

            # Só conta se a SFC estiver viva (> 0.0)
            if sfc_effective_rel > 0.0001:
                total_reliability += sfc_effective_rel
                active_count += 1

        # Retorna a média
        return total_reliability / active_count if active_count > 0 else 0.0

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

            # [NOVO] 1. Calcula a confiabilidade correta aqui no Controller
            real_reliability = self.get_effective_system_reliability()

            # [MODIFICADO] 2. Passa 'real_reliability' para o writter
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
                len(self.fail_manager.nodes_crashed) != 0,
                
                # ARGUMENTO NOVO AQUI:
                avg_sfc_reliability_override=real_reliability
            )

        # --- FUNÇÃO INTERNA (CORRIGIDA) ---
        def resilient_output(sfc_id, info):
            trial_id = info.get("crash_trial", self.crashs_trials)
            self.output_writter.resilient_output(sfc_id, info, trial_id)

        # Updates the network state and check for resource overhead.
        if not res_output:
            self.success.append(is_success)
            output_network_resources(current_time=current_time)
            output_flows(current_time, sfc_id, results_dict['latency'],
                        results_dict.get('comp_latency'),
                        results_dict.get('comm_latency'), 
                        results_dict['run_duration'], is_success,
                        wait_time=wait_time)
            
        # --- LÓGICA DE CRASH/RESILIÊNCIA (CORRIGIDA) ---
        sfcs_crash_aff = copy.deepcopy(list(self.sfcs_crash_affected.keys())) 
        if sfc_id in sfcs_crash_aff:
            stored_data = self.sfcs_crash_affected[sfc_id]
            
            # Verifica se NÃO foi um sucesso de backup imediato (é uma reinstanciação)
            if not stored_data.get('backup_success'):
                
                if results_dict:
                    # Define se a recuperação foi bem sucedida baseada no parametro da função
                    stored_data["recover_success"] = is_success 
                    
                    if is_success: 
                        # Cálculos de tempo
                        time_to_recover = time.time() - stored_data["fall_time"] 
                        
                        # --- CÁLCULO DE LATÊNCIA BRUTA ---
                        # latency_diff = Nova - Antiga
                        # Se positivo: Piorou X ms. Se negativo: Melhorou X ms.
                        latency_diff = results_dict['latency'] - stored_data["old_latency"] 
                        latency_factor = latency_diff # Valor BRUTO
                        
                        # --- CÁLCULO DE RECURSO BRUTO ---
                        old_resource_reuse = stored_data["resource_info"] 
                        
                        # resource_factor = Antigo - Novo
                        # Se positivo: Economizou X recursos. Se negativo: Gastou X a mais.
                        resource_factor = old_resource_reuse - results_dict['resource_info']
                        
                        # (O bloco de limitação > 2 ou < -1 foi removido para manter o valor real)
                        
                        # Atualiza o dicionário com os dados finais
                        stored_data["latency_diff"] = latency_diff
                        stored_data["latency_degrad"] = latency_factor
                        stored_data["resource_degrad"] = resource_factor
                        stored_data["time_to_recover"] = time_to_recover
                    else:
                        # Falhou, mantém para próxima tentativa
                        pass
                else:
                    stored_data["recover_success"] = False
            
            # --- SALVAMENTO E LIMPEZA ---
            if stored_data.get("recover_success") or stored_data.get("backup_success"):
                resilient_output(sfc_id, stored_data)
                del self.sfcs_crash_affected[sfc_id]
                    
        if self.verbose:
            print("__________________________________________")
            self.output_writter.print_output_info(self.substrate_network, self.success) 
            print("")

