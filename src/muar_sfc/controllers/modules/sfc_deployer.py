import copy
import time
from typing import Any

from loguru import logger

from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.sfc_state_tracker import SFCStateTracker
from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC


class SFCDeployer:
    """
    Orquestrador responsável exclusivamente pelo processo de alocação (Deploy)
    e reversão atômica (Rollback) de Service Function Chains na infraestrutura.
    
    Aplica o princípio SRP isolando a transação na rede física do rastreamento
    de tempo e estado lógico.
    """

    def __init__(self, tracker: SFCStateTracker, verbose: bool = True):
        # Injeção de Dependência: Recebe o rastreador para atualizar o estado global
        self.tracker = tracker
        self.verbose = verbose

    def submit_solution(
        self, sfc_list: list[SFC], solution: dict, substrate_network: Net2, is_backup: bool = False
    ) -> dict[str, Any]:
        """
        Tenta aplicar uma lista de SFCs na rede de substrato.
        Possui mecanismo transacional atômico: ou todas as SFCs do lote 
        são alocadas com sucesso, ou o sistema desfaz a operação inteira.
        """
        deployment_success = True
        route_info_export = {}
        deployed_in_this_batch = []

        # ==========================================
        # 1. VALIDAÇÃO PRÉVIA (Prevenção de falha)
        # ==========================================
        for sfc in sfc_list:
            if sfc.id not in solution or solution[sfc.id].get("route_info") is None:
                if self.verbose:
                    logger.warning(f"[Deployer] Solução inválida ou sem rotas para a SFC {sfc.id}.")
                return {"is_success": False, "route_info": None}

        # ==========================================
        # 2. TENTATIVA DE DEPLOY
        # ==========================================
        for sfc in sfc_list:
            rf = solution[sfc.id]["route_info"]

            try:
                # Deploy físico na topologia
                substrate_network.deploy_sfc(sfc, rf)

                # Registro de sucesso parcial no estado da Fachada
                self.tracker.sfcs_routing_info[sfc.id] = copy.deepcopy(rf)
                route_info_export[sfc.id] = rf
                deployed_in_this_batch.append(sfc.id)

            except Exception as e:
                logger.error(f"[Deployer] Deploy falhou na infraestrutura para {sfc.id}: {e}")
                deployment_success = False
                break  # Aborta imediatamente para iniciar o Rollback

        # ==========================================
        # 3. ROLLBACK ATÔMICO (Transação Compensatória)
        # ==========================================
        if not deployment_success:
            for sfc_id in deployed_in_this_batch:
                logger.warning(f"[Rollback] Desfazendo alocação parcial da SFC {sfc_id}")
                try:
                    # Remove o rastro físico
                    substrate_network.undeploy_sfc(sfc_id)

                    # Remove o rastro lógico via abordagem EAFP
                    self.tracker.sfcs_routing_info.pop(sfc_id, None)

                except Exception as rollback_e:
                    logger.exception(f"Falha CRÍTICA ao aplicar rollback na SFC {sfc_id}: {rollback_e}")

            return {"is_success": False, "route_info": None}

        # ==========================================
        # 4. EFETIVAÇÃO DO ESTADO LÓGICO
        # ==========================================
        # Se tudo ocorreu perfeitamente e não é um provisionamento oculto de backup:
        if not is_backup and deployment_success:
            group_id = sfc_list[0].dst_node
            duration = getattr(sfc_list[0], "duration", 100)

            if group_id in self.tracker.sfcs_tracker:
                logger.warning(f"[Deployer] Aviso: SFC Session {group_id} já instanciada. Sobrescrevendo rastreamento.")

            # Gravando no guardião do estado respeitando o TypedDict (SessionInfo)
            self.tracker.sfcs_tracker[group_id] = {
                "sfc_list": sfc_list,
                "solution": solution,
                "duration": duration,
                "timer": time.time()
            }
            self.tracker.counter += 1

        return {"is_success": deployment_success, "route_info": route_info_export}

    def run_garbage_collection(self, substrate_network: Net2, backup_manager: BackupManager | None) -> None:
        """
        Coletor de Lixo: Varre a rede de substrato física procurando instâncias
        que não possuem mais vínculos lógicos (Sessões ou Backups válidos),
        desalocando zumbis que drenam recursos.
        """
        # IDs que existem de fato fisicamente na rede
        infra_sfc_ids = set(substrate_network.sfc_dict.keys())
        valid_logical_ids = set()

        # Coleta os IDs lógicos rastreados atualmente no sistema
        for session_info in self.tracker.sfcs_tracker.values():
            valid_logical_ids.update(sfc.id for sfc in session_info["sfc_list"])

        # Coleta os IDs retidos na malha do gerenciador de backups
        if backup_manager:
            valid_logical_ids.update(backup_manager.backups_sfc_instantiated.keys())

        # Diferença de conjuntos: o que está na rede mas não está na lógica
        zombie_ids = infra_sfc_ids - valid_logical_ids

        if zombie_ids:
            if self.verbose:
                logger.warning(
                    f"[GC] Inconsistência detectada. Removendo {len(zombie_ids)} "
                    f"SFCs órfãs da rede: {zombie_ids}"
                )

            # Executa a purga
            for z_id in zombie_ids:
                try:
                    substrate_network.undeploy_sfc(z_id)
                except Exception:
                    logger.exception(f"[GC] Erro crítico ao limpar zumbi {z_id} da infraestrutura.")

    def safe_network_removal(self, sfc_id: str, substrate_network: Net2) -> None:
        """
        Remove com segurança uma SFC da infraestrutura física (SRP).
        """
        try:
            substrate_network.undeploy_sfc(sfc_id)
            if self.verbose:
                logger.debug(f"[RECURSOS LIBERADOS] Instância {sfc_id} purgada da rede física.")
        except Exception:
            logger.exception(f"[FALHA DE DESALOCAÇÃO] Erro CRÍTICO ao tentar remover {sfc_id} da infraestrutura.")
