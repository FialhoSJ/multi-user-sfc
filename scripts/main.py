import logging
import signal
import sys
from typing import Any, TypedDict

from muar_sfc.algorithms.instantiator import AlgorithmInstantiator
from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.crasher import Crasher
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.modules.sfcs_instatiator import SFCInstatiator
from muar_sfc.controllers.modules.sfcs_manager import SFCManager
from muar_sfc.controllers.sfc_queue import SFCQueue
from muar_sfc.controllers.substrate_network_controller import SubstrateNetworkController
from muar_sfc.controllers.substrate_network_controller_parallel import SubstrateNetworkControllerP
from muar_sfc.core.poisson_emitter import PoissonEmitter
from muar_sfc.core.scenarios.muar import MuarScenario
from muar_sfc.topology.instantiator import TopologyInstantiator
from muar_sfc.utils.failure_generator import calcular_janelas_falha
from muar_sfc.utils.manager_results import OutputWritter, create_output_dir

# Adoção correta da biblioteca logging. Em uma futura refatoração para nuvem, 
# você pode transitar para estruturação JSON via structlog.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 1. O Contrato de Tipagem (O Padrão Ouro)
# Usando TypedDict para firmar o contrato exato das chaves sem onerar a memória.
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

    # Note que utilizamos as coleções nativas 'list', obliterando as arcaicas 'typing.List'
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
    topology: Any,  # Tipagem transitória (Any) para evitar o erro Unknown do Pylance
    sfc_queue: SFCQueue,
    alg: Any,       # Tipagem transitória (Any)
    full_failure_schedule: list[FailureEvent],
) -> SubstrateNetworkController:
    """Instancia e configura o controlador da rede substrata."""

    network = topology.generate_substrate_network()
    network.verbose = "y"

    parallel_run = False
    sbn_controller = (
        SubstrateNetworkControllerP() if parallel_run else SubstrateNetworkController()
    )

    sbn_controller.substrate_network = network
    sbn_controller.sfc_queue = sfc_queue
    sbn_controller.sfc = settings.sfc
    sbn_controller.alg = settings.alg
    sbn_controller.fail_manager = Crasher(topology=topology, args=settings)
    sbn_controller.failure_schedule = full_failure_schedule

    sbn_controller.substrate_network.set_reliability_params(settings)
    sbn_controller.substrate_network.set_sharing_params(settings.share)

    sbn_controller.mobility_manager = MobilityManager(settings)

    sbn_controller.sfc_manager = SFCManager(
        settings,
        backup_manager=BackupManager(args=settings),
        alg=alg,
    )

    sbn_controller.sfc_instantiator = SFCInstatiator(alg, args=settings)

    sbn_controller.sfc_manager.alg_name = settings.alg
    sbn_controller.verbose = settings.verbose == "y"

    sbn_controller.output_writter = OutputWritter(
        topology,
        *create_output_dir(settings, topology),
    )

    sbn_controller.flows = settings.n_sessions
    sbn_controller.players = settings.n_players

    return sbn_controller


def main() -> None:
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # Instancia as configurações tipadas via Pydantic
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
        (None),
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

    # Bloco EAFP excelente. O tratamento de anomalias com captura passiva (except) é a diretiva idiomática
    try:
        logger.info("Iniciando a simulação do controlador de rede...")
        sbn_controller.start()

    except KeyboardInterrupt:
        # KeyboardInterrupt e SystemExit fazem parte do invólucro sistêmico global atômico intrínseco (BaseException)
        logger.warning("Execução interrompida pelo usuário (Ctrl+C).")
        sys.exit(0)

    except Exception:
        # A invocação de logger.exception() garante a inclusão do Stack Trace de forma nativa e integral
        logger.exception("Ocorreu um erro crítico durante a execução.")
        sys.exit(1)


if __name__ == "__main__":
    main()