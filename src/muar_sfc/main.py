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


# 1. O Contrato de Tipagem
class FailureEvent(TypedDict):
    type: str
    start: float
    duration: float


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

    # Coleções nativas 'list'
    full_failure_schedule: list[FailureEvent] = []

    for start, duration in raw_node_schedule:
        full_failure_schedule.append(
            {
                "type": "node",
                "start": float(start),
                "duration": float(duration),
            }
        )

    for start, duration in raw_link_schedule:
        full_failure_schedule.append(
            {
                "type": "link",
                "start": float(start),
                "duration": float(duration),
            }
        )

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

    # Proteção: Caso a biblioteca subjacente ainda exija strings "y"/"n",
    # nós fazemos a conversão aqui, mantendo a tipagem bool limpa no config.py
    share_str = "y" if settings.share else "n"
    network.set_sharing_params(share_str)
    network.verbose = "y" if settings.verbose else "n"

    # Instanciamos as dependências primeiro
    backup_manager = BackupManager(args=settings)
    sfc_manager = SFCManager(settings, backup_manager=backup_manager, alg=alg)
    sfc_manager.alg_name = settings.alg

    sfc_instantiator = SFCInstatiator(alg, args=settings)
    fail_manager = Crasher(topology=topology, args=settings)
    mobility_manager = MobilityManager(settings)
    output_writter = OutputWritter(topology, *create_output_dir(settings, topology))

    # Injeção Limpa
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

    settings = SimulationSettings()

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

    full_failure_schedule = generate_failure_schedule(
        settings,
        simulation_duration,
    )

    sbn_controller = setup_controller(
        settings,
        topology,
        sfc_queue,
        alg,
        full_failure_schedule,
    )

    try:
        logger.info("Iniciando a simulação do controlador de rede...")
        sbn_controller.start()

        # Mantém a thread principal viva para poder receber o Ctrl+C de forma limpa
        while not sbn_controller.is_stopped:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário (Ctrl+C). Iniciando encerramento suave...")
        sbn_controller.stop()  # Primeiro para a Thread do controlador
        sys.exit(0)


if __name__ == "__main__":
    main()
