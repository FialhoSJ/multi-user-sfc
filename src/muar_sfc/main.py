import argparse
import signal
import sys
import time
from typing import Any, TypedDict

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<cyan>[{file.name}]</cyan> <level>{message}</level>", level="INFO")

from muar_sfc.algorithms.instantiator import AlgorithmInstantiator
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.crasher import Crasher
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.modules.sfcs_instatiator import SFCInstatiator
from muar_sfc.controllers.modules.sfcs_manager import SFCManager
from muar_sfc.controllers.sfc_queue import SFCQueue
from muar_sfc.controllers.substrate_network_controller import SubstrateNetworkController
from muar_sfc.core.poisson_emitter import PoissonEmitter
from muar_sfc.core.scenarios.muar import MuarScenario
from muar_sfc.topology.instantiator import TopologyInstantiator
from muar_sfc.utils.failure_generator import calcular_janelas_falha
from muar_sfc.utils.manager_results import OutputWritter, create_output_dir


class FailureEvent(TypedDict):
    type: str
    start: float
    duration: float


def build_cli_parser() -> argparse.ArgumentParser:
    """F5: flags de CLI que sobrescrevem o SimulationSettings (usadas pelos runners)."""
    parser = argparse.ArgumentParser(description="MUAR-SFC Simulator")
    parser.add_argument("--n_sessions", type=int, help="número de sessões")
    parser.add_argument("--n_players", type=int, help="número de players por sessão")
    parser.add_argument("--alg", type=str, help="algoritmo de orquestração")
    parser.add_argument("--topology", type=str, help="nome da topologia")
    parser.add_argument("--sfc", type=str, help="on/off (gera SFCs)")
    parser.add_argument("--share", type=str, help="on/off (compartilhamento de SFs)")
    parser.add_argument("--ava", type=float, help="confiabilidade alvo")
    parser.add_argument("--number_of_fails", type=int, help="número de falhas")
    parser.add_argument("--crash_at", type=float, nargs="*", default=None,
                        help="tempos fixos de falha")
    parser.add_argument("--sfc_lifetime", type=int, help="duração da SFC (sessão)")
    parser.add_argument("--time", type=float, help="tempo total de simulação")
    parser.add_argument("--eco_effi_ratio", type=float, help="razão de eficiência ecológica")
    parser.add_argument("--verbose", type=str, help="on/off (log verboso)")
    parser.add_argument("--fail_target", type=str, help="alvo de falha")
    parser.add_argument("--service_mix", type=str, help="lista de serviços (ex.: muar,streaming)")
    parser.add_argument("--service_weights", type=str, help="pesos dos serviços (ex.: 0.7,0.3)")
    return parser


def _parse_bool(value: str | None) -> bool | None:
    """F5: converte strings legadas ('on'/'y'/'1') para bool."""
    if value is None:
        return None
    return value.strip().lower() in ("on", "y", "yes", "true", "1")


def build_settings_overrides(cli_args: argparse.Namespace) -> dict[str, Any]:
    """F5: mapeia as flags da CLI para campos do SimulationSettings."""
    overrides: dict[str, Any] = {}
    for field in (
        "n_sessions", "n_players", "alg", "topology", "ava", "number_of_fails",
        "crash_at", "sfc_lifetime", "time", "eco_effi_ratio", "fail_target",
        "service_mix", "service_weights",
    ):
        value = getattr(cli_args, field, None)
        if value is not None:
            overrides[field] = value
    for field in ("sfc", "share", "verbose"):
        value = _parse_bool(getattr(cli_args, field, None))
        if value is not None:
            overrides[field] = value
    return overrides


def generate_failure_schedule(settings: SimulationSettings, simulation_duration: float) -> list[FailureEvent]:
    """Gera o cronograma de falhas para nós e links de forma unificada."""
    fixed_time = settings.crash_at
    if fixed_time and fixed_time[0] == -1.0:
        fixed_time = None

    raw_node_schedule = calcular_janelas_falha(
        duracao_simulacao=simulation_duration,
        num_falhas=settings.number_of_fails,
        duracao_minima_falha=settings.min_fail_duration,
        confiabilidade=settings.ava,
        start_times=fixed_time,
    )

    raw_link_schedule = calcular_janelas_falha(
        duracao_simulacao=simulation_duration,
        num_falhas=settings.number_of_link_fails,
        duracao_minima_falha=settings.min_link_fail_duration,
        confiabilidade=settings.link_ava,
    )

    full_failure_schedule: list[FailureEvent] = []

    for start, duration in raw_node_schedule:
        full_failure_schedule.append({
            "type": "node",
            "start": float(start),
            "duration": float(duration),
        })

    for start, duration in raw_link_schedule:
        full_failure_schedule.append({
            "type": "link",
            "start": float(start),
            "duration": float(duration),
        })

    full_failure_schedule.sort(key=lambda x: x["start"])
    return full_failure_schedule


def setup_controller(
    settings: SimulationSettings,
    topology: Any,
    sfc_queue: SFCQueue,
    alg: Any,
    full_failure_schedule: list[FailureEvent],
) -> SubstrateNetworkController:
    """Instancia e configura o controlador da rede substrata usando Injeção de Dependência."""
    network = topology.generate_substrate_network()
    network.set_reliability_params(settings)

    share_str = "y" if settings.share else "n"
    network.set_sharing_params(share_str)
    network.verbose = "y" if settings.verbose else "n"

    backup_manager = BackupManager(args=settings)
    sfc_manager = SFCManager(settings, backup_manager=backup_manager, alg=alg)
    sfc_manager.alg_name = settings.alg

    sfc_instantiator = SFCInstatiator(alg, args=settings)
    fail_manager = Crasher(topology=topology, args=settings)
    mobility_manager = MobilityManager(settings)
    
    # Pathlib já deve estar sendo embutido implicitamente no create_output_dir
    output_writter = OutputWritter(topology, *create_output_dir(settings, topology))

    sbn_controller = SubstrateNetworkController(
        substrate_network=network,
        sfc_queue=sfc_queue,
        sfc_manager=sfc_manager,
        sfc_instantiator=sfc_instantiator,
        fail_manager=fail_manager,
        backup_manager=backup_manager,
        mobility_manager=mobility_manager,
        output_writter=output_writter,
        failure_schedule=full_failure_schedule,
        alg=alg,
        players=settings.n_players,
        flows=settings.n_sessions,
        sfc_name=settings.sfc,
        verbose=settings.verbose
    )

    return sbn_controller


def main() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)
    cli_args = build_cli_parser().parse_args()
    settings = SimulationSettings(**build_settings_overrides(cli_args))

    topology = TopologyInstantiator().instantiate_topology(
        settings.topology,
        settings.eco_effi_ratio,
    )

    alg = AlgorithmInstantiator().instantiate_algorithm(settings.alg)
    sfc_queue = SFCQueue()
    official_rate = 20
    sfc_poisson_emitter = PoissonEmitter(official_rate)
    muar_scenario = MuarScenario(settings, sfc_queue, topology, sfc_poisson_emitter)

    sfc_poisson_emitter.start(
        muar_scenario.generate_sfc_session,
        (None,)
    )

    simulation_duration = settings.n_sessions * official_rate
    full_failure_schedule = generate_failure_schedule(settings, simulation_duration)

    sbn_controller = setup_controller(
        settings, topology, sfc_queue, alg, full_failure_schedule
    )

    try:
        logger.info("Iniciando a simulação do controlador de rede...")
        sbn_controller.start()

        # Mantém a thread principal viva para poder receber o Ctrl+C de forma limpa
        while not sbn_controller.is_stopped:
            time.sleep(1)

        # F4: grava o resumo agregado por tipo de serviço ao final da simulação
        sbn_controller.output_writter.write_service_summary()

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário (Ctrl+C). Iniciando encerramento suave...")
        sbn_controller.stop()        # 1. Para o Controlador
        sfc_poisson_emitter.stop()   # 2. Para o Gerador de Sessões (A CORREÇÃO)
        sys.exit(0)


if __name__ == "__main__":
    main()
