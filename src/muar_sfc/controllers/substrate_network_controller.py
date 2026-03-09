# --- Standard Library Imports ---
import _thread
import copy
import logging
import random
import re
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any

# --- Third Party Imports ---
import numpy as np

from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.crasher import Crasher
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.modules.sfcs_instatiator import SFCInstatiator
from muar_sfc.controllers.modules.sfcs_manager import SFCManager
from muar_sfc.controllers.sfc_generator import SFCGenerator

# --- Local Module Imports ---
from muar_sfc.core.net_v2 import Net2
from muar_sfc.utils.manager_results import OutputWritter
from muar_sfc.utils.network_utils import EnergyCalculator

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Garante que a pasta logs exista ou trata erro em produção
try:
    ch = logging.FileHandler("./logs/substrate_network_controller.log")
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
except FileNotFoundError:
    pass  # Ignora se rodando em ambiente sem pasta logs criada


class SubstrateNetworkController:
    def __init__(self):
        # --- Network Initialization ---
        # Instancia a nova Net2 com sistema de métricas refatorado
        self.substrate_network = Net2()
        self.node_info = {}

        # --- Modules ---
        self.mobility_manager: MobilityManager | None = 0
        self.sfc_manager: SFCManager | None = 0
        self.sfc_instantiator: SFCInstatiator | None = 0
        self.fail_manager: Crasher | None = 0
        self.backup_manager: BackupManager | None = 0
        self.energy_calculator = EnergyCalculator()
        self.output_writter: OutputWritter | None = None

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
        self.crashs_trials = 0
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
        self.backup_interval_creation = 40 if self.alg in ["msf", "greedyb"] else 5

        self.start_time = time.time()
        self.last_mobility_time = self.start_time
        self.last_backup_time = self.start_time

        # Converte a lista de falhas para uma deque para processamento eficiente
        self.failure_schedule = deque(self.failure_schedule)
        self.iteration_counter = 0

    ###########################################################################
    #                       MAIN LOOP & ORCHESTRATION                         #
    ###########################################################################

    def sequential_operation(self) -> None:
        """
        Controla a execução sequencial das operações.
        Usa o Garbage Collector centralizado para limpar recursos da Net2.
        """
        self.initialize_timers()

        while not self.is_stopped:
            # [Net2 Update] Limpeza centralizada que respeita a consistência de métricas
            self.handle_resources_cleanup()

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
                self.check_mobility()
                self.last_mobility_time = time.time()

    def handle_backups(self) -> None:
        """
        Gerencia o ciclo de criação e implantação de backups.

        Orquestra a chamada ao gerenciador para criação lógica e, subsequentemente,
        realiza o deploy físico na rede para garantir consistência de estado.
        """
        # Guard Clause para evitar aninhamento excessivo e execução desnecessária [cite: 75]
        if not self.sfc_manager.backup_manager.backup_activated:
            return

        current_time = time.time()

        # Verificação de intervalo temporal
        if current_time - self.last_backup_time < self.backup_interval_creation:
            return

        # Definição do Agente (se aplicável)
        agent = None
        if self.alg == "REPLICMASKABLEPPO" or "ga":
            agent = self.sfc_instantiator.alg

        # 1. Criação Lógica (Factory)
        # O Manager decide "O QUE" criar, mas não altera o estado da rede física.
        created_backup_groups, strategy = self.sfc_manager.create_backups(
            self.substrate_network, agent_ref=agent
        )

        # 2. Persistência de Estado e Alocação (Orchestration)
        # O Controller garante que "O QUE" foi criado seja registrado "ONDE" (Net2).
        if created_backup_groups:
            self._deploy_backups_to_network(created_backup_groups)

            # Observabilidade Estruturada (Log)
            if self.verbose:
                count = sum(len(group) for group in created_backup_groups)
                print(
                    f"[BACKUP] Ciclo concluído. {count} novos backups registrados via {strategy}."
                )

        self.last_backup_time = current_time

    def _deploy_backups_to_network(self, backup_groups: list[list[Any]]) -> None:
        """
        Método auxiliar para registrar backups na rede física.

        Isola a lógica de iteração e tratamento de erros de deploy,
        mantendo o método principal limpo (Clean Code).
        """
        for group in backup_groups:
            for backup_sfc in group:
                try:
                    # Recuperação segura de atributos (EAFP)
                    # Backups via RL possuem rota pré-calculada; Greedy pode não ter.
                    route_info = getattr(backup_sfc, "pre_calculated_route", {})

                    # Ação Crítica:
                    # O método deploy_sfc do Net2 popula o self.sfc_dict.
                    # Sem isso, o get_sfc_by_id falha durante o recovery.
                    self.substrate_network.deploy_sfc(backup_sfc, route_info)

                except Exception as e:
                    print(f"Erro ao implantar backup {backup_sfc.id}: {e}")

    def handle_fails(self):
        """Gerencia o ciclo de vida (Crash -> Espera -> Recovery)."""
        if not self.fail_manager.activated:
            return

        elapsed_time = time.time() - self.start_time

        # 1. Recuperações Pendentes
        for failure in self.active_failures[:]:
            if elapsed_time >= failure["recovery_time"]:
                if failure["type"] == "node":
                    self.server_recovery_operation(failure["target"])
                elif failure["type"] == "link":
                    self.link_recovery_operation(failure["target"])

                self.active_failures.remove(failure)

        # 2. Novas Falhas Agendadas
        if self.failure_schedule and elapsed_time >= self.failure_schedule[0]["start"]:
            event = self.failure_schedule.popleft()
            duration = event["duration"]

            if event["type"] == "node":
                crashed_nodes, _ = self.server_fail_operation()
                if crashed_nodes and duration > 0:
                    recovery_time = elapsed_time + duration
                    self.active_failures.append(
                        {"type": "node", "target": crashed_nodes, "recovery_time": recovery_time}
                    )
                    print(
                        f"   -> Recuperação agendada para T={recovery_time:.2f}s (Daqui a {duration}s)"
                    )

            elif event["type"] == "link":
                link_crashed, _ = self.link_fail_operation()
                if link_crashed and duration > 0:
                    recovery_time = elapsed_time + duration
                    self.active_failures.append(
                        {"type": "link", "target": link_crashed, "recovery_time": recovery_time}
                    )
                    print(f"   -> Recuperação agendada para T={recovery_time:.2f}s")

    def update(self) -> None:
        """Updates the network state."""
        self.substrate_network.update()

    ###########################################################################
    #                   QUEUE & DEPLOYMENT MANAGEMENT                         #
    ###########################################################################

    def submit_sfcs(self):
        """Executa a submissão de SFCs."""
        self.check_timer_qeue()

        if self.max_queue_size < self.sfc_queue.qsize():
            self.max_queue_size = self.sfc_queue.qsize()
        processed_sfcs = []

        start_time = time.time()
        while self.sfc_queue.qsize() != 0:
            if time.time() - start_time > 1:
                break

            sfc_list = self.sfc_queue.peek_sfc()

            dequeue_time = time.time()
            wait_time = 0

            if hasattr(sfc_list[0], "enqueue_time"):
                wait_time = dequeue_time - sfc_list[0].enqueue_time
            elif hasattr(sfc_list[0], "arrival_time"):
                wait_time = dequeue_time - sfc_list[0].arrival_time

            for sfc in sfc_list:
                if sfc.dst_node in self.sfc_manager.sfcs_tracker:
                    raise ValueError("SFC já submetida")

            log, is_success = self.deploy_sfc_list(sfc_list)

            for sfc_id, result_dict in log.items():
                processed_sfcs.append(sfc_id)
                self.output_results(
                    sfc_id=sfc_id,
                    results_dict=result_dict,
                    is_success=is_success,
                    wait_time=wait_time,
                )
        return processed_sfcs

    def deploy_sfc_list(self, sfc_list) -> bool:
        mob_player_id = self.create_mobile_user(sfc_list)
        # Passa a Net2 atualizada para o instanciador
        solution, is_success = self.sfc_instantiator.search_solution(
            sfc_list, self.substrate_network
        )
        if is_success:
            self.sfc_manager.submit_solution(sfc_list, solution, self.substrate_network)
        else:
            self.remove_mobile_user(mob_player_id)
        return solution, is_success

    def send_back_to_qeue(
        self,
        sfc_list: list,
        changed_location: bool = False,
        new_location=False,
        punishment: int = 10,
    ) -> None:
        """
        Recicla uma lista de SFCs. Usa métodos seguros de undeploy da Net2.
        """
        if not sfc_list:
            return

        session_id = sfc_list[0].dst_node
        sfcs_tracker_info = self.sfc_manager.sfcs_tracker.get(session_id)

        # Se não há tracker, a sessão já morreu logicamente. Abortar.
        if not sfcs_tracker_info:
            return

        # CÁLCULO DE TEMPO ABSOLUTO (Referência Lógica)
        # Usamos o timer DA SESSÃO, não do objeto SFC (que pode ter sido reiniciado)
        session_start_time = sfcs_tracker_info["timer"]
        original_duration = sfcs_tracker_info["duration"]

        time_elapsed_absolute = time.time() - session_start_time
        remaining_duration = original_duration - time_elapsed_absolute

        # Validação Crítica: Se o tempo acabou, force a limpeza e não re-enfileire.
        if remaining_duration <= 1.0:  # Margem de segurança de 1s
            if self.verbose:
                print(f"[Queue] Sessão {session_id} expirou durante falha. Cancelando re-deploy.")
            self._force_cleanup_session(session_id)
            return

        new_sfc_list_dicts = []

        for sfc in sfc_list:
            sfc_id = sfc.id
            location = new_location if changed_location else sfc.closer_router
            new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)

            if changed_location and "cache" in sfc_id:
                old_loc = str(sfc.closer_router)
                new_loc = str(new_location)

                ma_old_key = "MA_region_" + old_loc
                re_old_key = "RE_region_" + old_loc

                old_routing_info = self.sfc_manager.sfcs_routing_info

                if sfc_id in old_routing_info:
                    current_sfc_routes = old_routing_info[sfc_id]
                    if ma_old_key in current_sfc_routes:
                        ma_new_key = re.sub(old_loc, new_loc, ma_old_key)
                        new_vnfs_list_dict[1]["name"] = ma_new_key

                    if re_old_key in current_sfc_routes:
                        re_new_key = re.sub(old_loc, new_loc, re_old_key)
                        new_vnfs_list_dict[2]["name"] = re_new_key

            new_sfc_dict = {
                "name": sfc_id,
                "vnf_list": new_vnfs_list_dict,
                "bandwidth": sfc.input_throughput,
                "src_node": sfc.src.substrate_node,
                "dst_node": sfc.dst_node,
                "duration": remaining_duration,  # O novo objeto nasce sabendo que tem pouco tempo
                "closer_router": location,
                "latency": sfc.latency_request,
            }
            new_sfc_list_dicts.append(new_sfc_dict)

        sfc_ids_to_remove = self.sfc_manager.cleanup_session_state(session_id)

        for old_sfc_id in sfc_ids_to_remove:
            self._force_remove_sfc_and_backups(old_sfc_id)

        new_sfcs_objects = [SFCGenerator(d).generate() for d in new_sfc_list_dicts]

        current_enqueue_time = time.time()
        for sfc in new_sfcs_objects:
            sfc.enqueue_time = current_enqueue_time

        self.sfc_queue.put_begin(new_sfcs_objects)

    def _force_cleanup_session(self, session_id: str):
        ids = self.sfc_manager.cleanup_session_state(session_id)
        for sfc_id in ids:
            self._force_remove_sfc_and_backups(sfc_id)

    def check_timer_qeue(self):
        if self.timer_qeue_sfcs:
            final_time = time.time()
            time_elapsed = final_time - self.timer_qeue_sfcs[0]["timer"]
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
                is_backup = sfc.id.split("_")[2]
                if is_backup == "backup":
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

                for sfc_id in sfc_list:
                    try:
                        sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                        obj_sfc_list.append(sfc_obj)
                    except KeyError:
                        if self.verbose:
                            print(f"[MOBILITY] Ignorando movimento da SFC {sfc_id} (SFC offline).")
                        valid_move = False
                        break

                if valid_move and obj_sfc_list:
                    print(f"SFCs moved: {sfc_list} | New Location: {new_location}")
                    self.send_back_to_qeue(
                        obj_sfc_list, changed_location=True, new_location=new_location
                    )

    def create_mobile_user(self, sfc_list):
        group_id = sfc_list[0].dst_node
        closer_router = sfc_list[0].closer_router
        sfc_id_list = [sfc.id for sfc in sfc_list]

        self.mobility_manager.add_player(group_id, closer_router, sfc_id_list)
        distance = self.mobility_manager.get_md_distance_from_router(group_id, closer_router)

        ips = 0.1
        self.substrate_network.add_node(
            group_id,
            "mobile_device",
            cpu_capacity=25.00,
            cache_capacity=10.00,
            ips=ips,
            position=distance,
        )
        return group_id

    def remove_mobile_user(self, sfc_list_id):
        self.mobility_manager.remove_player(sfc_list_id)
        self.substrate_network.remove_node(sfc_list_id)

    ###########################################################################
    #                      FAILURE & RECOVERY LOGIC                           #
    ###########################################################################

    def should_trigger_fail(self):
        return (
            time.time() - self.last_crasher_time >= self.crasher_interval
            and self.crashs_trials < self.crash_limit
        )

    def _calculate_sfc_path_latency(self, sfc_id: str) -> float:
        """Delega o cálculo complexo ponta-a-ponta para a infraestrutura física."""
        return self.substrate_network.calculate_sfc_total_latency(sfc_id)

    def server_fail_operation(self) -> tuple[list[str], list]:
        servers_failed = self._trigger_crash()
        if not servers_failed:
            return [], []

        affected_sfc_ids, sfc_failed_nodes_map = self._map_affected_sfcs(servers_failed)

        high_risk, med_risk, low_risk = self._classify_crash_risk(servers_failed, affected_sfc_ids)

        pre_crash_latencies, sfc_owners_map = self._audit_pre_crash(affected_sfc_ids)

        self._crash_servers(servers_failed)

        fallen_sfcs_list, post_crash_latencies = self._recover_sfcs(
            affected_sfc_ids, sfc_failed_nodes_map, pre_crash_latencies, sfc_owners_map
        )

        self._compute_crash_metrics(
            servers_failed,
            affected_sfc_ids,
            high_risk,
            med_risk,
            low_risk,
            pre_crash_latencies,
            post_crash_latencies,
        )

        self.sfc_manager.crashed_servers = self.fail_manager.nodes_crashed
        self.crashs_trials += 1

        return servers_failed, fallen_sfcs_list

    def _trigger_crash(self) -> list[str]:
        servers_failed = self.fail_manager.activate_crasher(
            self.substrate_network, self.sfc_manager, self.alg
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
                if "backup" in sfc_id:
                    continue

                affected_sfc_ids.add(sfc_id)
                sfc_failed_nodes_map.setdefault(sfc_id, []).append(server)

        return affected_sfc_ids, sfc_failed_nodes_map

    def _classify_crash_risk(self, servers_failed, affected_sfc_ids):
        high_risk = med_risk = low_risk = 0

        if not servers_failed:
            return high_risk, med_risk, low_risk

        target_node = servers_failed[0]
        base_id = str(target_node).split(".")[0]

        server_reliability = self.substrate_network.get_node_reliability(base_id)
        total_victims = len(affected_sfc_ids)

        if server_reliability < 0.8666:
            high_risk = total_victims
        elif 0.8666 <= server_reliability <= 0.9333:
            med_risk = total_victims
        else:
            low_risk = total_victims

        print(f"   -> Crash Source Reliability: {server_reliability:.4f}")
        print(
            f"   -> Impact bucket: {'High' if high_risk else 'Med' if med_risk else 'Low'} Risk Node"
        )

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

    def _get_real_risk_with_backups(self, sfc_id: str) -> str:
        """
        Calcula o risco REAL da SFC no momento do crash, considerando a
        confiabilidade combinada da rota primária + backups ativos.
        """
        network = self.substrate_network

        # 1. Recupera Backups Ativos
        backups_dict = {}
        if self.sfc_manager.backup_manager and not isinstance(
            self.sfc_manager.backup_manager, int
        ):
            backups_dict = self.sfc_manager.backup_manager.sfcs_backups_instatiated

        # Se não tem rota (já caiu antes ou erro), assume risco médio por segurança
        if sfc_id not in network.sfc_route_info:
            return "Medium"

        route_info = network.sfc_route_info[sfc_id]

        # 2. Mapeia Confiabilidade dos Backups (VNF -> Reliability)
        backup_reliability_map = {}
        if sfc_id in backups_dict:
            for b in backups_dict[sfc_id]:
                vnf_id = b.get("vnf_id")
                bk_route = b.get("route_info", {})
                # Acha o nó onde o backup está (ex: vnf1_b está no node 5)
                bk_node_list = next(
                    (v for k, v in bk_route.items() if k.endswith("_b") and v), None
                )

                if vnf_id and bk_node_list:
                    node_id = bk_node_list[0]
                    backup_reliability_map[vnf_id] = network.get_node_reliability(node_id)

        # 3. Agrupa Primários por Nó Físico (Domínio de Falha)
        node_groups = defaultdict(list)
        for vnf_id, path in route_info.items():
            if vnf_id in ["src", "dst"] or not path:
                continue
            node_groups[path[0]].append(vnf_id)

        # 4. Cálculo Matemático (RBD: Reliability Block Diagram)
        total_reliability = 1.0

        for node_id, vnfs_list in node_groups.items():
            # Confiabilidade do nó primário
            try:
                rel_primary = network.get_node_reliability(node_id)
            except KeyError:  # <-- CORRIGIDO: Captura estrita da anomalia do dicionário
                rel_primary = 1.0

            # Verifica redundância paralela para este grupo
            all_vnfs_protected = True
            prod_fail_backups = 1.0  # Produtório das falhas dos backups

            for vnf_id in vnfs_list:
                rel_bk = backup_reliability_map.get(vnf_id, 0.0)
                if rel_bk > 0.0:
                    prod_fail_backups *= 1.0 - rel_bk
                else:
                    all_vnfs_protected = False

            if all_vnfs_protected:
                # Sistema Paralelo: 1 - (Prob_Falha_Pri * Prob_Falha_Bks)
                prob_fail_primary = 1.0 - rel_primary
                group_rel = 1.0 - (prob_fail_primary * prod_fail_backups)
            else:
                # Sistema Série (Sem backup completo): Confiabilidade do nó primário
                group_rel = rel_primary

            total_reliability *= group_rel

        # 5. Classificação baseada nos seus thresholds
        if total_reliability < 0.8666:
            return "High"
        elif 0.8666 <= total_reliability <= 0.9333:
            return "Medium"
        else:
            return "Low"

    def _recover_sfcs(
        self,
        affected_sfc_ids: set,
        sfc_failed_nodes_map: dict,
        pre_crash_latencies: dict,
        sfc_owners_map: dict,
    ) -> tuple[list, dict]:

        fallen_sfcs_list = []
        post_crash_latencies = {}

        for sfc_id in sorted(affected_sfc_ids):
            # 1. Calcular Nível de Risco (Antes de mexer na rota)
            risk_level = self._get_real_risk_with_backups(sfc_id)

            failed_nodes = sfc_failed_nodes_map.get(sfc_id, [])
            relevant_server_down = failed_nodes[0] if failed_nodes else None

            try:
                sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                old_route_info = copy.deepcopy(self.sfc_manager.sfcs_routing_info.get(sfc_id))
            except KeyError:
                continue

            affected_vnf_id = None
            if old_route_info:
                for vnf, path in old_route_info.items():
                    if (
                        (vnf not in ["src", "dst"] and "virt" not in vnf)
                        and path
                        and path[0] == relevant_server_down
                    ):
                        affected_vnf_id = vnf
                        break

            # [TRECHO MANTIDO] Identificação de Backup Viável
            has_viable_backup = False
            aux = self.sfc_manager.backup_manager.sfcs_backups_instatiated
            if affected_vnf_id and sfc_id in aux:
                backups_list = self.sfc_manager.backup_manager.sfcs_backups_instatiated[sfc_id]
                for backup_entry in backups_list:
                    b_vnf_clean = backup_entry["vnf_id"].replace("_b", "")
                    if b_vnf_clean == affected_vnf_id:
                        has_viable_backup = True
                        break

            # CAMINHO 1: Sem Backup -> Fila
            if not has_viable_backup:
                if self.verbose:
                    if "backup" not in sfc_id:
                        print(
                            f"⚠️ [FAIL-FAST] SFC {sfc_id} perdeu VNF {affected_vnf_id} e NÃO tem backup. Enviando para fila."
                        )

                self.sfcs_crash_affected[sfc_id] = {
                    "fall_time": time.time(),
                    "old_latency": pre_crash_latencies.get(sfc_id, 0),
                    "resource_info": 0,
                    "backup_success": False,
                    "crash_trial": self.crashs_trials,
                    "recover_success": False,
                    "risk_level": risk_level,  # <--- SALVA O RISCO
                    "final_status": "Processing",  # Será atualizado no output_results
                }

                tracker_id = sfc_owners_map.get(sfc_id)
                if tracker_id and tracker_id in self.sfc_manager.sfcs_tracker:
                    self.mobility_manager.mark_vehicle_as_redeploying(tracker_id)
                    sfc_list_tracker = self.sfc_manager.sfcs_tracker[tracker_id]["sfc_list"]

                    if sfc_list_tracker:
                        self.send_back_to_qeue(sfc_list_tracker, changed_location=False)
                        fallen_sfcs_list.extend(sfc_list_tracker)
                continue

            # CAMINHO 2: Tem Backup -> Tenta Stitching
            try:
                self.substrate_network.undeploy_specific_vnf_context(sfc_id, affected_vnf_id)
            except Exception as e:
                print(f"Erro no undeploy pré-recuperação: {e}")

            recovery_start_time = time.time()
            recovered = False

            try:
                recovered = self.sfc_manager.reconstruct_and_redeploy(
                    sfc_obj, relevant_server_down, old_route_info, self.substrate_network
                )
            except Exception as e:
                print(f"Erro crítico no stitching: {e}")
                recovered = False

            if recovered:
                # RECUPERAÇÃO RÁPIDA (BACKUP)
                new_lat = self._calculate_sfc_path_latency(sfc_id)
                post_crash_latencies[sfc_id] = new_lat
                old_lat = pre_crash_latencies.get(sfc_id, 0)

                info_log = {
                    "crash_trial": self.crashs_trials,
                    "recover_success": True,
                    "backup_success": True,
                    "backup_efficient": "Yes",
                    "latency_before": old_lat,  # <--- NOVO
                    "latency_after": new_lat,  # <--- NOVO
                    "latency_diff": new_lat - old_lat,
                    "time_to_recover": time.time() - recovery_start_time,
                    "vnf_id": affected_vnf_id,
                    "latency_degrad": new_lat - old_lat,
                    "resource_degrad": 0,
                    "risk_level": risk_level,
                    "final_status": "Fast Recover",
                }

                # Escrita Direta no Arquivo
                self.output_writter.resilient_output(sfc_id, info_log, self.crashs_trials)

            else:
                # Falhou o Stitching -> Vai para a fila (Tentativa Lenta)
                self.sfcs_crash_affected[sfc_id] = {
                    "fall_time": time.time(),
                    "old_latency": pre_crash_latencies.get(sfc_id, 0),
                    "resource_info": 0,
                    "backup_success": False,
                    "crash_trial": self.crashs_trials,
                    "recover_success": False,
                    "risk_level": risk_level,  # <--- SALVA O RISCO
                    "final_status": "Processing",
                }

                tracker_id = sfc_owners_map.get(sfc_id)
                if tracker_id:
                    self._requeue_session(tracker_id, fallen_sfcs_list)

        return fallen_sfcs_list, post_crash_latencies

    def _requeue_session(self, tracker_id: str, fallen_list: list) -> None:
        if tracker_id in self.sfc_manager.sfcs_tracker:
            self.mobility_manager.mark_vehicle_as_redeploying(tracker_id)
            sfc_list_tracker = self.sfc_manager.sfcs_tracker[tracker_id]["sfc_list"]
            self.send_back_to_qeue(sfc_list_tracker, changed_location=False)
            fallen_list.extend(sfc_list_tracker)

    def _compute_crash_metrics(
        self,
        servers_failed,
        affected_sfc_ids,
        high_risk,
        med_risk,
        low_risk,
        pre_crash_latencies,
        post_crash_latencies,
    ):
        # affected_sfc_ids já vem limpo do método anterior (sem backups)
        total_affected = len(affected_sfc_ids)

        # [CORREÇÃO] Usar apenas SFCs primárias como universo total
        # Antes: total_active_sfcs = len(self.substrate_network.sfc_dict)
        total_active_sfcs = self.substrate_network.get_number_active_primary_sfcs()

        avg_lat_before = (
            np.mean(list(pre_crash_latencies.values())) if pre_crash_latencies else 0.0
        )
        avg_lat_after = (
            np.mean(list(post_crash_latencies.values())) if post_crash_latencies else 0.0
        )

        lat_diffs = [
            post_crash_latencies[sfc_id] - pre_crash_latencies[sfc_id]
            for sfc_id in post_crash_latencies
            if sfc_id in pre_crash_latencies
        ]

        avg_lat_diff = np.mean(lat_diffs) if lat_diffs else 0.0

        # Agora a porcentagem reflete o impacto real no serviço
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
            avg_lat_diff,
        )

    def server_recovery_operation(self, nodes_to_recover: list[str]):
        if not nodes_to_recover:
            return

        recovered_count = 0
        for node in nodes_to_recover:
            if self.fail_manager.recover_specific_node(self.substrate_network, node):
                print(f">>> [RECOVERY] Servidor recuperado: {node}")
                recovered_count += 1

        if recovered_count > 0:
            self.sfc_manager.crashed_servers = self.fail_manager.nodes_crashed

    def link_fail_operation(self):
        link_failed = self.fail_manager.activate_link_crasher(self.substrate_network)
        if not link_failed:
            print(">>> [CRASHER] Nenhum link disponível para falhar.")
            return None, []

        u, v = link_failed
        print(f">>> [CRASHER] Link Sorteado na Roleta: {u} <-> {v}")

        affected_sfcs = []
        for sfc_id, routing_info in self.sfc_manager.sfcs_routing_info.items():
            path_broken = False
            for vnf, path in routing_info.items():
                if vnf in ["src", "dst"] or not path:
                    continue

                for i in range(len(path) - 1):
                    node_a, node_b = path[i], path[i + 1]
                    if {node_a, node_b} == {u, v}:
                        path_broken = True
                        break
                if path_broken:
                    break

            if path_broken:
                affected_sfcs.append(sfc_id)

        self.substrate_network.set_link_down(u, v)

        fallen_sfcs_objects = []
        affected_groups = set()

        for sfc_id in affected_sfcs:
            try:
                sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                affected_groups.add(sfc_obj.dst_node)
            except (KeyError, AttributeError):
                pass

        for group_id in affected_groups:
            if group_id in self.sfc_manager.sfcs_tracker:
                self.mobility_manager.mark_vehicle_as_redeploying(group_id)
                sfc_list = self.sfc_manager.sfcs_tracker[group_id]["sfc_list"]
                self.send_back_to_qeue(sfc_list, changed_location=False)
                fallen_sfcs_objects.extend(sfc_list)

        return link_failed, fallen_sfcs_objects

    def link_recovery_operation(self, link_tuple):
        u, v = link_tuple
        self.substrate_network.restore_link(u, v)
        print(f">>> [RECOVERY] Link Restaurado: {u} <-> {v}")

    ###########################################################################
    #                      MAINTENANCE & REPORTING                            #
    ###########################################################################

    def check_simulation_end(self, sfc_list):
        """Verifica se a ÚLTIMA SFC da ÚLTIMA SESSÃO foi processada."""
        last_sf_mono = f"sfc_unique_p{self.players}_{self.flows}"
        last_sf_dec = f"sfc_mono_p{self.players}_{self.flows}"

        for sfc_id in sfc_list:
            if sfc_id == last_sf_mono or sfc_id == last_sf_dec:
                print(f"[INFO] Last SFC released detected: {sfc_id}")
                print("Max queue size:", self.max_queue_size)
                time.sleep(1)
                return True
        return False

    def get_effective_system_reliability(self) -> float:
        """Calcula a confiabilidade média REAL do sistema."""
        total_reliability = 0.0
        active_count = 0

        backups_dict = {}
        if self.sfc_manager and self.sfc_manager.backup_manager:
            backups_dict = self.sfc_manager.backup_manager.sfcs_backups_instatiated

        for sfc_id, sfc in self.substrate_network.sfc_dict.items():
            if sfc_id not in self.substrate_network.sfc_route_info:
                continue
            if "backup" in sfc_id:
                continue

            route_info = self.substrate_network.sfc_route_info[sfc_id]
            vnf_placement = {}
            for vnf_id, path in route_info.items():
                if vnf_id not in ["src", "dst"] and path:
                    vnf_placement[vnf_id] = path[0]

            if not vnf_placement:
                continue

            sfc_effective_rel = 1.0

            for vnf_id, main_node in vnf_placement.items():
                r_main = self.substrate_network.get_node_reliability(main_node)
                r_backup = 0.0

                if sfc_id in backups_dict:
                    for bk_info in backups_dict[sfc_id]:
                        if bk_info.get("vnf_id") == vnf_id:
                            bk_route = bk_info.get("route_info", {})
                            for bk_k, bk_path in bk_route.items():
                                if bk_k not in ["src", "dst"] and bk_path:
                                    bk_node = bk_path[0]
                                    r_backup = self.substrate_network.get_node_reliability(bk_node)
                                    break

                stage_rel = 1.0 - ((1.0 - r_main) * (1.0 - r_backup))
                sfc_effective_rel *= stage_rel

            if sfc_effective_rel > 0.0001:
                total_reliability += sfc_effective_rel
                active_count += 1

        return total_reliability / active_count if active_count > 0 else 0.0

    def handle_resources_cleanup(self) -> None:
        """
        Orquestrador Central de Limpeza de Recursos.
        """
        current_time = time.time()

        # 1. Limpeza de Sessões Expiradas (Regra de Negócio)
        expired_sessions = self.sfc_manager.get_expired_sessions(current_time)
        for session_id in expired_sessions:
            sfc_ids = self.sfc_manager.cleanup_session_state(session_id)
            for sfc_id in sfc_ids:
                self._force_remove_sfc_and_backups(sfc_id)
            self.remove_mobile_user(session_id)

        # 2. Limpeza de Backups Obsoletos (Regra de Negócio)
        if self.sfc_manager.backup_manager:
            backups_to_kill = self.sfc_manager.backup_manager.identify_obsolete_backups()
            for backup_id in backups_to_kill:
                self._safe_undeploy_backup(backup_id)

        # 3. Coleta de Lixo de Consistência (Safety Net)
        # Executa periodicamente para limpar "Zumbis"
        if self.iteration_counter % 10 == 0:
            self._run_garbage_collector()

    def _run_garbage_collector(self):
        """
        Remove inconsistências entre a camada lógica (Managers) e física (Net2).
        Aplica Set Theory para identificar 'Zumbis' de forma agnóstica ao nome.
        """
        # 1. Snapshot da Realidade Física (Apenas leitura)
        infra_sfc_ids = set(self.substrate_network.sfc_dict.keys())

        # 2. Construção da Verdade Lógica (Agregação de Fontes Confiáveis)
        valid_logical_ids = set()

        # Fonte A: Sessões Ativas (SFCs Primárias)
        # Desacoplamento: Acessamos os valores, sem saber a estrutura interna da chave
        for session_info in self.sfc_manager.sfcs_tracker.values():
            # List Comprehension para extração rápida
            valid_logical_ids.update(sfc.id for sfc in session_info["sfc_list"])

        # Fonte B: Backups Ativos
        if self.sfc_manager.backup_manager:
            # O Manager deve ser a fonte da verdade sobre seus próprios IDs
            valid_logical_ids.update(
                self.sfc_manager.backup_manager.backups_sfc_instantiated.keys()
            )

        # 3. Identificação de Zumbis (Lógica Pura: O que está na Infra mas não na Lógica)
        zombie_ids = infra_sfc_ids - valid_logical_ids

        # 4. Execução da Limpeza
        if zombie_ids:
            if self.verbose:
                print(
                    f"[GC] Inconsistência detectada. Removendo {len(zombie_ids)} SFCs órfãs: {zombie_ids}"
                )

            for z_id in zombie_ids:
                try:
                    # Delegação: O Controller manda a Rede limpar, sem saber como a Rede faz isso.
                    self.substrate_network.undeploy_sfc(z_id)
                except Exception as e:
                    print(f"[GC] Erro crítico ao limpar zumbi {z_id}: {e}")

    def _force_remove_sfc_and_backups(self, sfc_id: str) -> None:
        """Remove SFC e backups associados de forma atômica."""
        bm = self.sfc_manager.backup_manager
        if bm and sfc_id in bm.sfcs_backups_instatiated:
            backups_list = list(bm.sfcs_backups_instatiated[sfc_id])
            for backup_entry in backups_list:
                b_id = backup_entry["sfc_backup_id"]
                self._safe_undeploy_backup(b_id)

        try:
            self.substrate_network.undeploy_sfc(sfc_id)
        except Exception:
            pass

    def _safe_undeploy_backup(self, backup_id: str) -> None:
        try:
            self.substrate_network.undeploy_sfc(backup_id)
        except Exception:
            pass

        if self.sfc_manager.backup_manager:
            self.sfc_manager.backup_manager.cleanup_internal_state(backup_id)

    def output_results(
        self, results_dict, sfc_id, is_success, res_output=False, wait_time=None
    ) -> None:
        current_time = time.time()

        def output_network_resources(current_time):
            self.output_writter.output_cpu_utilization(
                self.substrate_network, current_time, self.fail_manager.nodes_crashed
            )
            self.output_writter.output_gpu_utilization(
                self.substrate_network, current_time, self.fail_manager.nodes_crashed
            )
            self.output_writter.output_cache_utilization(
                self.substrate_network, current_time, self.fail_manager.nodes_crashed
            )
            self.output_writter.output_bandwidth_utilization(self.substrate_network, current_time)
            self.output_writter.output_nodes_sf_utilization(self.substrate_network, current_time)

        def output_flows(
            current_time,
            sfc_id,
            latency,
            comp_latency,
            comm_latency,
            run_duration,
            is_success,
            alg_name=self.alg,
            fail_reason=None,
            latency_diff=None,
            wait_time=None,
        ):
            bw_transcode = 0
            server_energy_consumption = self.energy_calculator.calculate_total_server_power(
                self.substrate_network
            )
            mobile_energy_consumption = self.energy_calculator.calculate_total_mobile_device_power(
                self.substrate_network
            )
            total_energy_consumption = server_energy_consumption + mobile_energy_consumption

            if isinstance(self.sfc_manager.backup_manager, int):
                aux = 0
            else:
                aux = self.sfc_manager.backup_manager.sfcs_backups_instatiated
            self.substrate_network.calculate_average_system_reliability(aux)

            backups_dict_ref = {}
            if self.sfc_manager.backup_manager and not isinstance(
                self.sfc_manager.backup_manager, int
            ):
                backups_dict_ref = self.sfc_manager.backup_manager.sfcs_backups_instatiated
            # ---------------------------------------------------------------

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
                server_energy_consumption,
                mobile_energy_consumption,
                total_energy_consumption,
                latency_diff,
                len(self.fail_manager.nodes_crashed) != 0,
                # avg_sfc_reliability_override removido
                backups_dict=backups_dict_ref,  # <--- PASSANDO O DICIONÁRIO AQUI
            )

        def resilient_output(sfc_id, info):
            trial_id = info.get("crash_trial", self.crashs_trials)
            self.output_writter.resilient_output(sfc_id, info, trial_id)

        if not res_output:
            self.success.append(is_success)
            output_network_resources(current_time=current_time)
            output_flows(
                current_time,
                sfc_id,
                results_dict["latency"],
                results_dict.get("comp_latency"),
                results_dict.get("comm_latency"),
                results_dict["run_duration"],
                is_success,
                wait_time=wait_time,
            )

        sfcs_crash_aff = copy.deepcopy(list(self.sfcs_crash_affected.keys()))
        if sfc_id in sfcs_crash_aff:
            stored_data = self.sfcs_crash_affected[sfc_id]

            # Se chegamos aqui, é porque a SFC foi redeployada via Fila.

            if results_dict:  # Se o redeploy na fila funcionou
                stored_data["recover_success"] = is_success
                if is_success:
                    # RECUPERAÇÃO LENTA (Sucesso após Fila)
                    time_to_recover = time.time() - stored_data["fall_time"]

                    # Novas métricas corrigidas
                    latency_before = stored_data["old_latency"]
                    latency_after = results_dict["latency"]
                    latency_diff = latency_after - latency_before

                    stored_data["latency_before"] = latency_before  # <--- NOVO
                    stored_data["latency_after"] = latency_after  # <--- NOVO
                    stored_data["latency_diff"] = latency_diff

                    resource_factor = stored_data["resource_info"] - results_dict["resource_info"]
                    stored_data["latency_degrad"] = latency_diff
                    stored_data["resource_degrad"] = resource_factor
                    stored_data["time_to_recover"] = time_to_recover
                    stored_data["final_status"] = "Slow Recover"
                else:
                    # FALHA (Não conseguiu realocar)
                    stored_data["final_status"] = "Failed"  # <--- DEFINIDO
            else:
                # Falha por timeout ou erro
                stored_data["recover_success"] = False
                stored_data["final_status"] = "Failed"  # <--- DEFINIDO

            # Garantir que risco exista (caso venha de legado)
            if "risk_level" not in stored_data:
                stored_data["risk_level"] = "Medium"

            # Salva no arquivo
            resilient_output(sfc_id, stored_data)
            del self.sfcs_crash_affected[sfc_id]

        if self.verbose:
            print("__________________________________________")
            self.output_writter.print_output_info(self.substrate_network, self.success)
            print("")
