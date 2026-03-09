import argparse
import logging
import signal
import sys

from muar_sfc.algorithms.instantiator import AlgorithmInstantiator
from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.crasher import Crasher
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.modules.sfcs_instatiator import SFCInstatiator

# Imports dos Controladores e Módulos
from muar_sfc.controllers.modules.sfcs_manager import SFCManager
from muar_sfc.controllers.sfc_queue import SFCQueue
from muar_sfc.controllers.substrate_network_controller import SubstrateNetworkController
from muar_sfc.controllers.substrate_network_controller_parallel import SubstrateNetworkControllerP

# Imports do Core e Algoritmos
from muar_sfc.core.poisson_emitter import PoissonEmitter
from muar_sfc.core.scenarios.muar import MuarScenario
from muar_sfc.topology.instantiator import TopologyInstantiator

# Imports de Utilitários
from muar_sfc.utils.failure_generator import calcular_janelas_falha
from muar_sfc.utils.manager_results import OutputWritter, create_output_dir

# Configuração de Observabilidade Básica (Substituindo o uso de print)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Realiza o parse dos argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Select Immersive Service arguments")

    # --- Parâmetros Gerais e de Log ---
    parser.add_argument("--application", type=str, default="muar", help="Type of application")
    parser.add_argument("--alg", type=str, default="vegeta", help="Algorithm name")
    parser.add_argument(
        "--sfc_lifetime",
        dest="time",
        type=int,
        default=120,
        help="Duration/Lifetime of a single SFC session in seconds",
    )
    parser.add_argument("--verbose", type=str, default="y", help="Verbose log (y/n)")

    # --- Parâmetros de Topologia e Rede ---
    parser.add_argument(
        "--topology",
        type=str,
        default="luxembourgv2",
        help="Topology name (ex: luxembourgv2, paloalto)",
    )
    parser.add_argument(
        "--eco_effi_ratio",
        type=float,
        default=0.7,
        help="Proporção slots econômicos (ex: 0.7 = 70% eco)",
    )
    parser.add_argument("--mobility", type=str, default="n", help="Mobility enabled (y/n)")

    # --- Parâmetros de Tráfego (Sessões/Jogadores) ---
    parser.add_argument("--n_sessions", type=int, default=50, help="Number of sessions")
    parser.add_argument("--n_players", type=int, default=6, help="Number of players")

    # --- Parâmetros SFC ---
    parser.add_argument(
        "--allow_md_host",
        type=str,
        default="y",
        help="Permite que o dispositivo móvel (usuário) hospede VNFs (y/n)",
    )
    parser.add_argument("--sfc", type=str, default="on", help="SFC enabled (on/off)")
    parser.add_argument("--share", type=str, default="y", help="Share SFs (y/n)")
    parser.add_argument("--shareband", type=str, default="n", help="Share bandwidth (y/n)")
    parser.add_argument("--allow_delay", type=str, default="n", help="Allow delay (y/n)")

    # --- Parâmetros de Confiabilidade e Falhas ---
    parser.add_argument(
        "--backup",
        type=str,
        default="n",
        help="Backup enabled (y/n). Active only for specific algorithms.",
    )
    parser.add_argument("--ava", type=str, default="0.99", help="Meta Global de Disponibilidade")
    parser.add_argument("--number_of_fails", type=str, default="3", help="Number of node failures")
    parser.add_argument(
        "--min_fail_duration", type=float, default=20, help="Min duration of node failure"
    )
    parser.add_argument(
        "--crash_at",
        nargs="+",
        type=float,
        default=[180.0, 520.0, 640.0],
        help="Forcar falha em X segundos (Ex: 300 400 500). Use -1 para aleatorio.",
    )

    # 2. Configuração MICRO (Define a Confiabilidade Base por Nível)
    parser.add_argument(
        "--rel_high",
        type=float,
        default=0.999,
        help="Confiabilidade base para Nível Alto (Tier C)",
    )
    parser.add_argument(
        "--rel_normal",
        type=float,
        default=0.98,
        help="Confiabilidade base para Nível Normal (Tier B)",
    )
    parser.add_argument(
        "--rel_low", type=float, default=0.95, help="Confiabilidade base para Nível Baixo (Tier A)"
    )

    # Fator de Estresse
    parser.add_argument(
        "--stress_high",
        type=float,
        default=0.02,
        help="Penalidade por estresse para Tier C (High)",
    )
    parser.add_argument(
        "--stress_normal",
        type=float,
        default=0.08,
        help="Penalidade por estresse para Tier B (Normal)",
    )
    parser.add_argument(
        "--stress_low", type=float, default=0.15, help="Penalidade por estresse para Tier A (Low)"
    )

    parser.add_argument(
        "--fail_target",
        type=str,
        default="all",
        help="Alvo das falhas: all, high_risk (Nivel A), med_risk (Nivel B), low_risk (Nivel C)",
    )
    parser.add_argument(
        "--link_ava", type=str, default="0.95", help="Link availability (0.0 to 1.0)"
    )
    parser.add_argument(
        "--number_of_link_fails", type=str, default="0", help="Number of link failures"
    )
    parser.add_argument(
        "--min_link_fail_duration", type=float, default=20, help="Min duration of link failure"
    )

    return parser.parse_args()


def generate_failure_schedule(args: argparse.Namespace, simulation_duration: float) -> list[dict]:
    """Gera o cronograma de falhas para nós e links de forma unificada."""

    # Corrige a lógica do argparse para tempo aleatório (-1 ou [-1.0])
    fixed_time = args.crash_at
    if fixed_time and fixed_time[0] == -1.0:
        fixed_time = None
    elif isinstance(fixed_time, (float, int)):
        fixed_time = [float(fixed_time)]

    # Gera cronograma para NÓS
    raw_node_schedule = calcular_janelas_falha(
        duracao_simulacao=simulation_duration,
        num_falhas=int(args.number_of_fails),
        duracao_minima_falha=args.min_fail_duration,
        confiabilidade=float(args.ava),
        start_times=fixed_time,
    )

    # Gera cronograma para LINKS
    raw_link_schedule = calcular_janelas_falha(
        duracao_simulacao=simulation_duration,
        num_falhas=int(args.number_of_link_fails),
        duracao_minima_falha=args.min_link_fail_duration,
        confiabilidade=float(args.link_ava),
    )

    full_failure_schedule = []

    for start, duration in raw_node_schedule:
        full_failure_schedule.append(
            {
                "type": "node",
                "start": start,
                "duration": duration,
            }
        )

    for start, duration in raw_link_schedule:
        full_failure_schedule.append({"type": "link", "start": start, "duration": duration})

    # Ordenação Cronológica
    full_failure_schedule.sort(key=lambda x: x["start"])
    return full_failure_schedule


def setup_controller(
    args: argparse.Namespace, topology, sfc_queue: SFCQueue, alg, full_failure_schedule: list[dict]
) -> SubstrateNetworkController:
    """Instancia e configura o controlador da rede substrata com Injeção de Dependências."""
    network = topology.generate_substrate_network()
    network.verbose = "y"

    parallel_run = False
    sbn_controller = (
        SubstrateNetworkControllerP() if parallel_run else SubstrateNetworkController()
    )

    # Injeção de dependências no controlador
    sbn_controller.substrate_network = network
    sbn_controller.sfc_queue = sfc_queue
    sbn_controller.sfc = args.sfc
    sbn_controller.alg = alg.name
    sbn_controller.fail_manager = Crasher(topology=topology, args=args)
    sbn_controller.failure_schedule = full_failure_schedule
    sbn_controller.substrate_network.set_reliability_params(args)
    sbn_controller.substrate_network.set_sharing_params(args.share)

    # Gerenciadores Auxiliares
    sbn_controller.mobility_manager = MobilityManager(args)
    sbn_controller.sfc_manager = SFCManager(args, backup_manager=BackupManager(args=args), alg=alg)
    sbn_controller.sfc_instantiator = SFCInstatiator(alg, args=args)

    # Configurações Finais
    sbn_controller.sfc_manager.alg_name = args.alg
    sbn_controller.verbose = args.verbose == "y"
    sbn_controller.output_writter = OutputWritter(topology, *create_output_dir(args, topology))
    sbn_controller.flows = int(args.n_sessions)
    sbn_controller.players = int(args.n_players)

    return sbn_controller


def main() -> None:
    # Restaura o handler padrão de Ctrl+C
    signal.signal(signal.SIGINT, signal.default_int_handler)

    args = parse_arguments()

    # Inicializa Topologia e Algoritmo
    topology = TopologyInstantiator().instantiate_topology(args.topology, args.eco_effi_ratio)
    alg = AlgorithmInstantiator().instantiate_algorithm(args.alg)

    # Filas e Emissor
    sfc_queue = SFCQueue()
    official_rate = 20
    sfc_poisson_emitter = PoissonEmitter(official_rate)

    # Cenário
    muar_scenario = MuarScenario(args, sfc_queue, topology, sfc_poisson_emitter)

    # Inicia a geração de sessões no emissor
    sfc_poisson_emitter.start(muar_scenario.generate_sfc_session, (None))

    # Geração de Cronograma
    simulation_duration = float(args.n_sessions) * official_rate
    full_failure_schedule = generate_failure_schedule(args, simulation_duration)

    # Configuração do Controlador
    sbn_controller = setup_controller(args, topology, sfc_queue, alg, full_failure_schedule)

    # Execução baseada no paradigma EAFP (Easier to Ask Forgiveness than Permission)
    try:
        logger.info("Iniciando a simulação do controlador de rede...")
        sbn_controller.start()
    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário (Ctrl+C).")
        sys.exit(0)
    except Exception:
        logger.exception("Ocorreu um erro crítico durante a execução.")
        sys.exit(1)


if __name__ == "__main__":
    main()
