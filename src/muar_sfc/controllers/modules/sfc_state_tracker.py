import copy
import time
from typing import Any, TypedDict

from loguru import logger

from muar_sfc.core.net_v2 import Net2
from muar_sfc.core.sfc import SFC
from muar_sfc.controllers.sfc_generator import SFCGenerator


# Contrato estrito para o rastreamento das sessões, garantindo integridade das chaves
class SessionInfo(TypedDict):
    sfc_list: list[SFC]
    solution: dict
    duration: float
    timer: float


class SFCStateTracker:
    """
    Guardião do estado lógico das Service Function Chains (SFCs).
    
    Responsabilidade Única (SRP): Rastrear as sessões, contabilizar tempo de 
    expiração e manter os dicionários de roteamento e metadados sincronizados,
    isolando o estado da lógica de alocação física ou recuperação de desastres.
    """

    def __init__(self):
        # State Trackers Centrais
        self.sfcs_tracker: dict[str, SessionInfo] = {}
        self.sfcs_routing_info: dict[str, dict] = {}
        self.sfc_id_duration: dict[str, dict] = {}

        # Listas auxiliares de estado
        self.crashed_servers: list[str] = []
        self.risk_servers: list[str] = []

        # Contador global
        self.counter = 0

    def get_expired_sessions(self, current_time: float) -> list[str]:
        """
        Identifica sessões cujo tempo de vida expirou comparado ao limite instanciado.
        """
        expired_sessions = []
        for sfc_list_id, info in self.sfcs_tracker.items():
            start_time = info.get("timer", 0)
            duration = info.get("duration", 0)
            elapsed_time = current_time - start_time
            
            if elapsed_time >= duration:
                expired_sessions.append(sfc_list_id)
                logger.debug(
                    f"[EXPIRAÇÃO] Sessão {sfc_list_id} estourou o tempo! "
                    f"(Durou: {elapsed_time:.2f}s / Limite: {duration}s)"
                )

        return expired_sessions

    def cleanup_session_state(self, sfc_list_id: str) -> list[str]:
        """
        Remove o rastreamento lógico de uma sessão via EAFP, eliminando checagens
        condicionais duplas e retornando as SFCs individuais.
        """
        # Abordagem EAFP: tenta remover e captura de forma segura se não existir
        sfc_group = self.sfcs_tracker.pop(sfc_list_id, None)
        if not sfc_group:
            return []

        individual_sfc_ids = [sfc.id for sfc in sfc_group["sfc_list"]]

        # Limpeza do dicionário de rotas associadas a cada SFC limpa
        for sfc_id in individual_sfc_ids:
            self.sfcs_routing_info.pop(sfc_id, None)

        return individual_sfc_ids

    def rebuild_sfcs_for_requeue(
        self, sfc_list: list[SFC], changed_location: bool, new_location: Any
    ) -> tuple[list[SFC], float]:
        """
        Reconstrói uma lista de SFCs para re-enfileiramento baseando-se no
        tempo restante de vida rastreado na sessão original.
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
            return [], remaining_duration

        new_sfc_list_dicts = []

        for sfc in sfc_list:
            sfc_id = sfc.id
            location = new_location if changed_location else sfc.closer_router
            new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)

            if changed_location and "cache" in sfc_id:
                old_loc = str(sfc.closer_router)
                new_loc = str(new_location)

                ma_old_key = f"MA_region_{old_loc}"
                re_old_key = f"RE_region_{old_loc}"

                if sfc_id in self.sfcs_routing_info:
                    current_sfc_routes = self.sfcs_routing_info[sfc_id]
                    
                    # Substituição otimizada nativa via String (evitando o gargalo da engine Regex)
                    if ma_old_key in current_sfc_routes:
                        ma_new_key = ma_old_key.replace(old_loc, new_loc)
                        new_vnfs_list_dict[1]["name"] = ma_new_key

                    if re_old_key in current_sfc_routes:
                        re_new_key = re_old_key.replace(old_loc, new_loc)
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

        # Regeração usando o construtor da camada de domínio
        new_sfcs_objects = [SFCGenerator(d).generate() for d in new_sfc_list_dicts]

        current_enqueue_time = time.time()
        for sfc in new_sfcs_objects:
            sfc.enqueue_time = current_enqueue_time

        return new_sfcs_objects, remaining_duration

    def get_sfc_list(self, sfc_id: str, sb_net: Net2) -> list[SFC]:
        """Recupera a lista lógica de SFCs de uma sessão a partir de um ID isolado."""
        sfc = sb_net.get_sfc_by_id(sfc_id)
        if not sfc:
            return []
        
        group_id = sfc.dst_node
        if group_id in self.sfcs_tracker:
            return self.sfcs_tracker[group_id]["sfc_list"]
        return []

    def get_running_players_sessions(self) -> tuple[int, int, int]:
        """
        Calcula as estatísticas sobre sessões ativas com base nas chaves registradas.
        Usado extensamente para logs operacionais.
        """
        running_sfcs = self.sfcs_tracker
        players = set()
        sessions = set(running_sfcs.keys())

        # Extração heurística do player ID (Ex: sfc_list_p1_10 -> p1)
        for group_id in running_sfcs.keys():
            parts = group_id.split("_")
            for p in parts:
                if p.startswith("p") and p[1:].isdigit():
                    players.add(p)
                    break

        return len(running_sfcs), len(players), len(sessions)