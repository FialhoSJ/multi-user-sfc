import sys
import signal
import argparse

# Imports dos Controladores e Módulos
from controllers.modules.sfcs_manager import SFCManager
from controllers.modules.sfcs_instatiator import SFCInstatiator
from controllers.modules.mobility_manager import MobilityManager
from controllers.modules.backup_manager import BackupManager
from controllers.modules.crasher import Crasher
from controllers.substrate_network_controller import SubstrateNetworkController
from controllers.substrate_network_controller_parallel import SubstrateNetworkControllerP
from controllers.sfc_queue import SFCQueue

# Imports do Core e Algoritmos
from core.poisson_emitter import PoissonEmitter
from core.scenarios.muar import MuarScenario
from algorithms.instantiator import AlgorithmInstantiator
from topology.instantiator import TopologyInstantiator

# Imports de Utilitários
from utils.failure_generator import calcular_janelas_falha
from utils.manager_results import create_output_dir, OutputWritter

def main():
    # Restaura o handler padrão de Ctrl+C
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # ==============================================================================
    # 1. PARSE DE ARGUMENTOS
    # ==============================================================================
    parser = argparse.ArgumentParser(description='Select Immersive Service arguments')
    
    # --- Parâmetros Gerais e de Log ---
    parser.add_argument('--application', type=str, default='muar', help='Type of application')
    parser.add_argument('--alg', type=str, default='SBRCMASKABLEPPO', help='Algorithm name')
    parser.add_argument('--time', type=int, default=120, help='Total simulation time in seconds')
    parser.add_argument('--verbose', type=str, default='y', help='Verbose log (y/n)')
    
    # --- Parâmetros de Topologia e Rede ---
    parser.add_argument('--topology', type=str, default='luxembourgv2', help='Topology name (ex: luxembourgv2, paloalto)')
    parser.add_argument('--eco_effi_ratio', type=float, default=0.7, help='Proporção slots econômicos (ex: 0.7 = 70% eco)')
    parser.add_argument('--mobility', type=str, default='y', help='Mobility enabled (y/n)')
    
    # --- Parâmetros de Tráfego (Sessões/Jogadores) ---
    parser.add_argument('--n_sessions', type=int, default=50, help='Number of sessions')
    parser.add_argument('--n_players', type=int, default=4, help='Number of players')
    
    # --- Parâmetros SFC ---
    parser.add_argument('--sfc', type=str, default='on', help='SFC enabled (on/off)')
    parser.add_argument('--share', type=str, default='y', help='Share SFs (y/n)')
    parser.add_argument('--shareband', type=str, default='n', help='Share bandwidth (y/n)')
    parser.add_argument('--allow_delay', type=str, default='n', help='Allow delay (y/n)')
    
   # --- Parâmetros de Confiabilidade e Falhas ---
    parser.add_argument('--backup', type=str, default='y', help='Backup enabled (y/n). Active only for specific algorithms.')
    
    # 1. Configuração MACRO (Gera o Cronograma)
    parser.add_argument('--ava', type=str, default='0.97', help='Meta Global de Disponibilidade')
    parser.add_argument('--number_of_fails', type=str, default='25', help='Number of node failures')
    parser.add_argument('--min_fail_duration', type=float, default=20, help='Min duration of node failure')
    parser.add_argument('--crash_at', type=float, default=-1, help='Forcar falha em X segundos (Ex: [300, 400, 500]). Use -1 para aleatorio. Respeite o numero de falhas.')

    # 2. Configuração MICRO (Define a Confiabilidade Base por Nível)
    # Valores entre 0.0 e 1.0 (Ex: 0.99 = 99% confiável)
    parser.add_argument('--rel_high', type=float, default=0.99, help='Confiabilidade base para Nível Alto (Tier C)')
    parser.add_argument('--rel_normal', type=float, default=0.98, help='Confiabilidade base para Nível Normal (Tier B)')
    parser.add_argument('--rel_low', type=float, default=0.95, help='Confiabilidade base para Nível Baixo (Tier A)')
    
    # Fator de Estresse (NOVO)
    # Define o quanto a carga da CPU penaliza a confiabilidade.
    # Ex: 0.05 significa que 100% de uso reduz a confiabilidade em 5%.
# Fator de Estresse por Tier (Deltas de Penalidade)
    # Tier C (High): 0.9999 -> 0.9990 (Delta 0.0009)
    parser.add_argument('--stress_high', type=float, default=0.1, help='Penalidade por estresse para Tier C (High)')
    
    # Tier B (Normal): 0.99 -> 0.95 (Delta 0.04)
    parser.add_argument('--stress_normal', type=float, default=0.1, help='Penalidade por estresse para Tier B (Normal)')
    
    # Tier A (Low): 0.95 -> 0.80 (Delta 0.15)
    parser.add_argument('--stress_low', type=float, default=0.1, help='Penalidade por estresse para Tier A (Low)')
    
    # Opções: 'all' (qualquer um), 'high_risk' (Tier A), 'med_risk' (Tier B), 'low_risk' (Tier C)
    parser.add_argument('--fail_target', type=str, default='all', help='Alvo das falhas: all, high_risk (Nivel A), med_risk (Nivel B), low_risk (Nivel C)')
    
    # Falhas de LINKS
    parser.add_argument('--link_ava', type=str, default='0.95', help='Link availability (0.0 to 1.0)')
    parser.add_argument('--number_of_link_fails', type=str, default='0', help='Number of link failures')
    parser.add_argument('--min_link_fail_duration', type=float, default=20, help='Min duration of link failure')

    args = parser.parse_args()

    # ==============================================================================
    # 2. INICIALIZAÇÃO DO AMBIENTE (TOPOLOGIA, CENÁRIO, ALGORITMO)
    # ==============================================================================
    
    # Inicializa Topologia
    topology = TopologyInstantiator().instantiate_topology(args.topology, args.eco_effi_ratio)

    # Filas e Emissor
    sfc_queue = SFCQueue()
    official = 20 # Taxa base para o Poisson/Simulação
    sfc_poisson_emitter = PoissonEmitter(official)

    # Cenário e Algoritmo
    muar_scenario = MuarScenario(args, sfc_queue, topology, sfc_poisson_emitter)
    ALG = AlgorithmInstantiator().instantiate_algorithm(args.alg)

    # Inicia a geração de sessões no emissor
    sfc_poisson_emitter.start(muar_scenario.generate_sfc_session, (None))

    # ==============================================================================
    # 3. GERAÇÃO DE CRONOGRAMA DE FALHAS (NÓS E LINKS)
    # ==============================================================================
    
    simulation_duration = float(args.n_sessions) * official

    # Define se usa tempo fixo ou aleatório
    fixed_time = args.crash_at
    
    if fixed_time == -1:
        fixed_time = None
    # Se veio um número único do argparse, envelopa numa lista para ter len()
    elif isinstance(fixed_time, (float, int)):
        fixed_time = [fixed_time]

    # Gera cronograma para NÓS
    raw_node_schedule = calcular_janelas_falha(
        duracao_simulacao=simulation_duration,
        num_falhas=int(args.number_of_fails),
        duracao_minima_falha=args.min_fail_duration,
        confiabilidade=float(args.ava),
        start_times=fixed_time # <--- Agora isso será uma lista [300.0] ou None
    )

    # Gera cronograma para NÓS
    raw_node_schedule = calcular_janelas_falha(
        duracao_simulacao=simulation_duration,
        num_falhas=int(args.number_of_fails),
        duracao_minima_falha=args.min_fail_duration,
        confiabilidade=float(args.ava),
        start_times=fixed_time # <--- Passando o novo parametro
    )

    # Gera cronograma para LINKS
    raw_link_schedule = calcular_janelas_falha(
        duracao_simulacao=simulation_duration,
        num_falhas=int(args.number_of_link_fails),
        duracao_minima_falha=args.min_link_fail_duration,
        confiabilidade=float(args.link_ava)
    )

    # Unificação e "Etiquetagem" dos eventos
    full_failure_schedule = []

    for start, duration in raw_node_schedule:
        full_failure_schedule.append({
            'type': 'node', 
            'start': start, 
            'duration': duration,
        })

    for start, duration in raw_link_schedule:
        full_failure_schedule.append({
            'type': 'link',
            'start': start,
            'duration': duration
        })

    # Ordenação Cronológica (Crucial)
    full_failure_schedule.sort(key=lambda x: x['start'])

    # =============================================================================
    # 4. CONFIGURAÇÃO DO CONTROLADOR (NETWORK CONTROLLER)
    # =============================================================================
    
    network = topology.generate_substrate_network()
    network.verbose = 'y'
    
    parallel_run = False
    if parallel_run:
        sbn_controller = SubstrateNetworkControllerP()
    else: 
        sbn_controller = SubstrateNetworkController()

    # Injeção de dependências no controlador
    sbn_controller.substrate_network = network 
    sbn_controller.sfc_queue = sfc_queue
    sbn_controller.sfc = args.sfc
    sbn_controller.alg = ALG.name
    sbn_controller.fail_manager = Crasher(topology=topology, args=args)
    sbn_controller.failure_schedule = full_failure_schedule
    sbn_controller.substrate_network.set_reliability_params(args)
    
    # Gerenciadores Auxiliares
    sbn_controller.mobility_manager = MobilityManager(args)
    sbn_controller.sfc_manager = SFCManager(args, backup_manager=BackupManager(args=args), alg=ALG)
    sbn_controller.sfc_instantiator = SFCInstatiator(ALG)
    
    # Configurações Finais
    sbn_controller.sfc_manager.alg_name = args.alg
    sbn_controller.verbose = (args.verbose == 'y')
    sbn_controller.output_writter = OutputWritter(topology, *create_output_dir(args, topology))
    sbn_controller.flows = int(args.n_sessions)
    sbn_controller.players = int(args.n_players)

    # ==============================================================================
    # 5. EXECUÇÃO
    # ==============================================================================
    try:
        sbn_controller.start()
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário (Ctrl+C).")
        sys.exit(0)

if __name__ == "__main__":
    main()