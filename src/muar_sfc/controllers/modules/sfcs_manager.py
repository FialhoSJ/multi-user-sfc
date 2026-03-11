import copy
import time
import logging
import re
from typing import Any

from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC
from muar_sfc.controllers.sfc_generator import SFCGenerator # <- Novo import necessário

logger = logging.getLogger(__name__)


class SFCManager:
    """
    Gerencia o ciclo de vida lógico das Service Function Chains (SFCs).

    Responsabilidades:
    1. Rastrear sessões ativas (SFCs agrupadas por destino/usuário).
    2. Armazenar informações de roteamento e duração.
    3. Facilitar a reconstrução de rotas em caso de falhas (Recovery).

    Nota: A remoção física de recursos (undeploy) é orquestrada pelo Controller,
    baseada nas informações fornecidas por esta classe.
    """

    def __init__(self, args, backup_manager: BackupManager, alg):
        # Configuration & Managers
        self.backup_manager: BackupManager | None = backup_manager
        self.alg = alg
        self.alg_name = getattr(alg, "name", str(alg))
        self.args = args
        self.verbose = True

        # State Trackers
        # Mapeia: Destino (Group ID) -> Info da Sessão
        self.sfcs_tracker: dict[str, dict[str, Any]] = {}

        # Mapeia: SFC ID -> Informações de Rota (Cópia profunda da implantação)
        self.sfcs_routing_info: dict[str, dict] = {}

        # Mapeia: SFC ID -> Dados de duração/chegada
        self.sfc_id_duration: dict[str, dict] = {}

        # Listas auxiliares de estado
        self.crashed_servers: list[str] = []
        self.risk_servers: list[str] = []

        # Counters
        self.counter = 0

    # ==========================================
    # Core Lifecycle Methods (Deploy)
    # ==========================================

    def submit_solution(
        self, sfc_list: list[SFC], solution: dict, substrate_network: Net2, is_backup: bool = False
    ) -> dict[str, Any]:
        """
        Aplica a solução de roteamento na rede física e registra a sessão no tracker.

        Args:
            sfc_list: Lista de objetos SFC a serem implantados.
            solution: Dicionário com a solução do algoritmo (rotas).
            substrate_network: A instância da rede física.
            is_backup: Flag indicando se é uma implantação de backup (não rastreada como sessão).

        Returns:
            Dict com status de sucesso e informações da rota exportada.
        """
        deployment_success = True
        route_info_export = None

        for sfc in sfc_list:
            if sfc.id not in solution:
                deployment_success = False
                continue

            rf = solution[sfc.id].get("route_info")
            if not rf:
                deployment_success = False
                continue

            try:
                # Tenta realizar o deploy físico na rede
                substrate_network.deploy_sfc(sfc, rf)

                # Mantém cópia profunda da rota para referência futura (Recovery/Migration)
                self.sfcs_routing_info[sfc.id] = copy.deepcopy(rf)
                route_info_export = rf

            except Exception as e:
                if self.verbose:
                    print(f"[SFCManager] Deploy failed for {sfc.id}: {e}")
                deployment_success = False
                # Em caso de falha atômica, idealmente deveríamos fazer rollback,
                # mas aqui seguimos a lógica original de tentar o próximo.
                continue

        # Registro no Tracker de Sessões (Apenas para SFCs Primárias)
        if not is_backup and deployment_success:
            group_id = sfc_list[0].dst_node
            duration = getattr(sfc_list[0], "duration", 100)

            if group_id in self.sfcs_tracker:
                # Log de aviso em vez de erro fatal pode ser mais seguro em simulação
                print(
                    f"[SFCManager] Aviso: SFC Session {group_id} já instanciada. Sobrescrevendo."
                )

            self.sfcs_tracker[group_id] = {
                "sfc_list": sfc_list,
                "solution": solution,
                "duration": duration,
                "timer": time.time(),
            }
            self.counter += 1

        return {"is_success": deployment_success, "route_info": route_info_export}

    # ==========================================
    # Lifecycle Management (Expiration/Cleanup)
    # ==========================================

    def get_expired_sessions(self, current_time: float) -> list[str]:
        """
        Identifica sessões (grupos de SFCs) cujo tempo de vida expirou.

        Returns:
            List[str]: Lista de IDs de sessão (group_id/dst_node) para remoção.
        """
        expired_sessions = []
        for sfc_list_id, info in self.sfcs_tracker.items():
            start_time = info.get("timer", 0)
            duration = info.get("duration", 0)

            elapsed_time = current_time - start_time
            if elapsed_time >= duration:
                expired_sessions.append(sfc_list_id)

        return expired_sessions

    def cleanup_session_state(self, sfc_list_id: str) -> list[str]:
        """
        Remove o rastreamento lógico de uma sessão e retorna os IDs das SFCs
        individuais para que o Controller possa remover os recursos físicos.

        Args:
            sfc_list_id: ID da sessão (geralmente o nó de destino).

        Returns:
            List[str]: Lista de IDs das SFCs que compunham a sessão.
        """
        if sfc_list_id not in self.sfcs_tracker:
            return []

        sfc_group = self.sfcs_tracker[sfc_list_id]
        sfc_list = sfc_group.get("sfc_list", [])

        # Coleta IDs para retorno
        individual_sfc_ids = [sfc.id for sfc in sfc_list]

        # Limpeza do estado interno
        del self.sfcs_tracker[sfc_list_id]

        # Limpeza opcional de routing info (pode ser mantido para logs ou limpo aqui)
        for sfc_id in individual_sfc_ids:
            if sfc_id in self.sfcs_routing_info:
                del self.sfcs_routing_info[sfc_id]

        return individual_sfc_ids

    # ==========================================
    # Backup Delegation
    # ==========================================

    def create_backups(self, network: Net2, agent_ref: Any = None) -> tuple[list[Any], str]:
        """Delega a criação de backups para o BackupManager.

        Args:
            network: A rede de substrato.
            agent_ref: O agente de aprendizado (opcional).

        Returns:
            Tupla com (lista de backups, nome da estratégia). Retorna vazio se falhar.
        """
        if self.backup_manager:
            return self.backup_manager.create_backups(network, agent_ref)
        return [], "none"

    def reconstruct_and_redeploy(
        self, sfc_obj: SFC, crashed_node_id: str, old_route_info: dict, substrate_network: Net2
    ) -> bool:
        """
        Executa a recuperação de falha (Stitching) ativando um backup existente.
        """
        # 1. Verifica Backups Disponíveis
        if (
            not self.backup_manager
            or sfc_obj.id not in self.backup_manager.sfcs_backups_instatiated
        ):
            if self.verbose:
                print(f"❌ [STITCH-FAIL] {sfc_obj.id}: Nenhum backup instanciado.")
            return False

        # 2. Identifica VNF afetada pelo nó caído
        affected_vnf_id = None
        for vnf_id, path in old_route_info.items():
            if vnf_id in ["src", "dst"] or not path:
                continue
            if path[0] == crashed_node_id:
                affected_vnf_id = vnf_id
                break

        if not affected_vnf_id:
            return False

        # 3. Busca a Réplica Específica
        backups_list = self.backup_manager.sfcs_backups_instatiated[sfc_obj.id]
        target_backup = None
        for backup_entry in backups_list:
            b_vnf_clean = backup_entry["vnf_id"].replace("_b", "")
            if b_vnf_clean == affected_vnf_id:
                target_backup = backup_entry
                break

        if not target_backup:
            return False

        # 4. Prepara a nova rota (Stitching)
        new_route_info = copy.deepcopy(old_route_info)
        backup_route = target_backup["route_info"]
        backup_sfc_id = target_backup["sfc_backup_id"]

        key_ingress = "src_virt"
        key_egress = None
        for k in backup_route:
            if k.endswith("_b") and k != "src_virt" and k != "dst_virt":
                key_egress = k
                break

        if not key_egress:
            return False

        path_ingress = backup_route.get(key_ingress)
        path_egress = backup_route.get(key_egress)

        if not path_ingress or not path_egress:
            return False

        # Verifica saúde do nó de backup
        backup_node = path_egress[0]
        backup_node_obj = substrate_network.graph.nodes[backup_node]
        if not backup_node_obj.get("is_active", True):
            if self.verbose:
                print(f"❌ [STITCH-FAIL] Nó de backup {backup_node} também falhou.")
            return False

        # =================================================================
        # 5. EXECUÇÃO CRÍTICA: ATOMIC SWAP
        # =================================================================

        affected_vnf_obj = sfc_obj.get_vnf_by_id(affected_vnf_id)
        bw_in = affected_vnf_obj.get_income_interface_bandwidth()
        bw_out = affected_vnf_obj.get_outcome_interface_bandwidth()

        # Ativa banda nos caminhos de backup
        if not substrate_network.activate_backup_path_bandwidth(path_ingress, bw_in, sfc_obj.id):
            return False
        if not substrate_network.activate_backup_path_bandwidth(path_egress, bw_out, sfc_obj.id):
            return False

        # Swap de CPU/RAM (Desaloca Backup -> Aloca Original)
        try:
            backup_sfc_obj = substrate_network.get_sfc_by_id(backup_sfc_id)
            backup_vnf_obj = backup_sfc_obj.get_vnf_by_id(key_egress)

            # A) Desaloca fisicamente o backup
            substrate_network.deallocate_microservice(backup_node, backup_sfc_id, backup_vnf_obj)
            substrate_network.detach_vnf_from_route_record(backup_sfc_id, key_egress)

            # B) Aloca a VNF original no lugar
            substrate_network.allocate_microservice(sfc_obj, affected_vnf_obj, backup_node)

        except ValueError as e:
            if self.verbose:
                print(f"❌ [STITCH-FAIL] Falha na troca de recursos: {e}")
            return False

        # C) Atualiza Rotas
        prev_vnf = sfc_obj.get_previous_vnf(affected_vnf_obj)
        if prev_vnf.id == "src":
            new_route_info["src"] = path_ingress
        else:
            new_route_info[prev_vnf.id] = path_ingress
        new_route_info[affected_vnf_id] = path_egress

        self.sfcs_routing_info[sfc_obj.id] = new_route_info
        if hasattr(substrate_network, "sfc_route_info"):
            substrate_network.sfc_route_info[sfc_obj.id] = copy.deepcopy(new_route_info)

        # D) Ressuscita SFC se necessário
        if sfc_obj.id not in substrate_network.sfc_dict:
            substrate_network.sfc_dict[sfc_obj.id] = sfc_obj

        # E) Limpeza Pós-Recuperação
        # Aqui usamos a nova interface do BackupManager e removemos fisicamente o resto do backup
        try:
            # 1. Remove SFC de backup da rede física (já removemos a VNF crítica, falta o resto)
            substrate_network.undeploy_sfc(backup_sfc_id)

            # 2. Atualiza estado do BackupManager
            self.backup_manager.cleanup_internal_state(backup_sfc_id)

            # Registra que este backup foi "ativado" (consumido)
            if backup_sfc_id not in self.backup_manager.backups_activated:
                self.backup_manager.backups_activated.append(backup_sfc_id)

        except Exception as e:
            print(f"Aviso na limpeza do backup {backup_sfc_id}: {e}")

        if self.verbose:
            print(
                f"[STITCH-SUCCESS] {sfc_obj.id}: VNF {affected_vnf_id} recuperada em {backup_node}"
            )

        return True
    
    
    
    # ==========================================
    # Lógica Extraída do Controller (SRP)
    # ==========================================

    def rebuild_sfcs_for_requeue(
        self, sfc_list: list[SFC], changed_location: bool, new_location: Any
    ) -> tuple[list[SFC], float]:
        """
        Reconstrói uma lista de SFCs para re-enfileiramento.
        Abstrai do Controller a manipulação profunda de dicionários e RegEx.
        Retorna as novas SFCs e o tempo restante (remaining_duration).
        """
        if not sfc_list:
            return [], 0.0

        session_id = sfc_list[0].dst_node
        sfcs_tracker_info = self.sfcs_tracker.get(session_id)

        if not sfcs_tracker_info:
            return [], 0.0

        session_start_time = sfcs_tracker_info["timer"]
        original_duration = sfcs_tracker_info["duration"]

        time_elapsed_absolute = time.time() - session_start_time
        remaining_duration = original_duration - time_elapsed_absolute

        if remaining_duration <= 1.0:
            return [], remaining_duration # Sinaliza que expirou

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

                if sfc_id in self.sfcs_routing_info:
                    current_sfc_routes = self.sfcs_routing_info[sfc_id]
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
                "duration": remaining_duration,
                "closer_router": location,
                "latency": sfc.latency_request,
            }
            new_sfc_list_dicts.append(new_sfc_dict)

        new_sfcs_objects = [SFCGenerator(d).generate() for d in new_sfc_list_dicts]

        current_enqueue_time = time.time()
        for sfc in new_sfcs_objects:
            sfc.enqueue_time = current_enqueue_time

        return new_sfcs_objects, remaining_duration

    def run_garbage_collection(self, substrate_network: Net2) -> None:
        """
        Remove inconsistências entre a camada lógica e física (Zumbis).
        """
        infra_sfc_ids = set(substrate_network.sfc_dict.keys())
        valid_logical_ids = set()

        for session_info in self.sfcs_tracker.values():
            valid_logical_ids.update(sfc.id for sfc in session_info["sfc_list"])

        if self.backup_manager:
            valid_logical_ids.update(
                self.backup_manager.backups_sfc_instantiated.keys()
            )

        zombie_ids = infra_sfc_ids - valid_logical_ids

        if zombie_ids:
            if self.verbose:
                logger.warning(
                    f"[GC] Inconsistência detectada. Removendo {len(zombie_ids)} "
                    f"SFCs órfãs: {zombie_ids}"
                )

            for z_id in zombie_ids:
                try:
                    substrate_network.undeploy_sfc(z_id)
                except Exception:
                    logger.exception(f"[GC] Erro crítico ao limpar zumbi {z_id}.")

    # ==========================================
    # Helpers
    # ==========================================

    def get_sfc_list(self, sfc_id: str, sb_net: Net2) -> list[SFC]:
        """Recupera a lista de SFCs de uma sessão a partir de um ID de SFC."""
        sfc = sb_net.get_sfc_by_id(sfc_id)
        if not sfc:
            return []
        group_id = sfc.dst_node
        if group_id in self.sfcs_tracker:
            return self.sfcs_tracker[group_id]["sfc_list"]
        return []

    def get_running_players_sessions(self) -> tuple[int, int, int]:
        """Estatísticas sobre sessões ativas (usado para logs)."""
        running_sfcs = self.sfcs_tracker
        players = set()
        sessions = set()

        # Lógica aproximada baseada na nomenclatura (sfc_pX_Y)
        # Ajuste conforme seu padrão de nomes
        for group_id, _info in running_sfcs.items():
            sessions.add(group_id)
            # Extração heurística do player ID
            # Ex: sfc_list_p1_10 -> p1
            parts = group_id.split("_")
            for p in parts:
                if p.startswith("p") and p[1:].isdigit():
                    players.add(p)
                    break

        return len(running_sfcs), len(players), len(sessions)

    def deploy_success(self, sfc: SFC) -> None:
        if self.verbose:
            print(f"Deploy succeed: {sfc.id}")

    def deploy_failed(self, sfc: SFC) -> None:
        if self.verbose:
            print(f"Deploy FAILED: {sfc.id}")
