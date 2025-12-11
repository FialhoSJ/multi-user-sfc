
def main():
    import argparse
    from controllers.modules.sfcs_manager import SFCManager
    from controllers.modules.sfcs_instatiator import SFCInstatiator
    from utils.failure_generator import calcular_janelas_falha

    from controllers.substrate_network_controller import SubstrateNetworkController
    from datetime import datetime as dt
    from controllers.substrate_network_controller_parallel import SubstrateNetworkControllerP
    from core.poisson_emitter import PoissonEmitter
    from controllers.sfc_queue import SFCQueue
    from algorithms.instantiator import AlgorithmInstantiator
    from controllers.modules.mobility_manager import MobilityManager
    from controllers.modules.backup_manager import BackupManager

    from controllers.modules.crasher import Crasher
    from core.scenarios.muar import MuarScenario
    from topology.instantiator import TopologyInstantiator
    from utils.manager_results import create_output_dir, OutputWritter
    import signal
    import sys
    # Restaura o handler padrão de Ctrl+C
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # command line arguments
    parser = argparse.ArgumentParser(description='Select Immersive Service arguments') 
    parser.add_argument('--application', type=str, help='type of application', default='muar')
    parser.add_argument('--alg',   type=str, help='(str) algorithm name', default='greedyb')
    parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
    parser.add_argument('--n_players', type=int, help='(int) number of players', default=6)
    #on:  quebrar mais em funçoes
    #off: monolítico
    parser.add_argument('--sfc',   type=str, help='(str) on or off', default='on')
    parser.add_argument('--topology', type=str, help='(str) wich topology ex: luxembourg,small luxembourg ,paloalto', default='luxembourgv2') #luxembourgv2
    parser.add_argument(
        '--eco_effi_ratio',
        type=float,
        default=0.7,
        help='Define a proporção para slots econômicos (ex: 0.7 significa 70%% eco e 30%% eficiência).'
    )
    parser.add_argument('--share',  type=str, help='(str) whether to share sfs or not', default='y')
    parser.add_argument('--shareband',  type=str, help='(str) whether to share sfs or not', default='n')
    parser.add_argument('--time',  type=int, help='(int) the total time for the simulation in seconds', default=120)
    parser.add_argument('--mobility',  type=str, help='(str) mobility', default='y')

    parser.add_argument('--allow_delay', type=str, help='(str) whether to allow delay or not', default='n')

    parser.add_argument('--backup', type=str, help='(str) whether to allow delay or not', default='s')
    parser.add_argument('--ava', type=str, help='(str) whether to allow delay or not', default='0.95')
    parser.add_argument('--number_of_fails', type=str, help='(str) whether to allow delay or not', default='5')
    parser.add_argument('--min_fail_duration', type=float, help='(float) minimum duration of a failure in seconds', default=10)
    parser.add_argument('--verbose',   type=str, help='verbose log', default='y')

    #Coleta dos parâmetros da simulação
    args = parser.parse_args()

    # -------------- Inicializa Topologia ----------------
    topology = TopologyInstantiator().instantiate_topology(args.topology, args.eco_effi_ratio)

    # Criando a fila de SFCs 
    sfc_queue = SFCQueue()

    # Criando emissor Poisson
    test = 10
    official = 20
    sfc_poisson_emitter = PoissonEmitter(official)

    # Criando instância da classe MuarScenario
    muar_scenario = MuarScenario(args, sfc_queue,topology, sfc_poisson_emitter)

    # Iniciando a simulação 
    sfc_poisson_emitter.start(muar_scenario.generate_sfc_session,(None))

    ALG = AlgorithmInstantiator().instantiate_algorithm(args.alg)

    # 1. Gera cronograma para NÓS (Servidores)
    raw_node_schedule = calcular_janelas_falha(
        duracao_simulacao=float(args.n_sessions)*official,
        num_falhas=int(args.number_of_fails),
        duracao_minima_falha=args.min_fail_duration,
        confiabilidade=float(args.ava)
    )

    raw_link_schedule = calcular_janelas_falha(
        duracao_simulacao=float(args.n_sessions)*official,
        num_falhas=int(args.number_of_link_fails),     # Novo parametro
        duracao_minima_falha=args.min_link_fail_duration, # Novo parametro
        confiabilidade=float(args.link_ava)            # Novo parametro
    )
    # print(f"--- Cronograma de Falhas Agendado (Início, Duração) ---\n{failure_schedule}\n----------------------------------------------------")

    # 3. Unifica e "Etiqueta" os eventos
    full_failure_schedule = []

    # Adiciona nós com a etiqueta 'node'
    for start, duration in raw_node_schedule:
        full_failure_schedule.append({
            'type': 'node', 
            'start': start, 
            'duration': duration
        })

    # Adiciona links com a etiqueta 'link'
    for start, duration in raw_link_schedule:
        full_failure_schedule.append({
            'type': 'link', 
            'start': start, 
            'duration': duration
        })

    # 4. Ordena tudo cronologicamente (importante para o popleft funcionar certo)
    full_failure_schedule.sort(key=lambda x: x['start'])
    
    network = topology.generate_substrate_network()
    network.verbose = 'y'
    
    parallel_run = False
    if parallel_run:
        sbn_controller = SubstrateNetworkControllerP() # run parallel
    else: 
        sbn_controller = SubstrateNetworkController() # runs sequential    

    sbn_controller.substrate_network = network 
    sbn_controller.sfc_queue = sfc_queue
    sbn_controller.sfc = args.sfc
    sbn_controller.alg = ALG.name
    sbn_controller.fail_manager = Crasher(topology=topology, args=args)
    sbn_controller.failure_schedule = failure_schedule  # Passa o cronograma para o controlador

    # sbn_controller.backup_manager = BackupManager(args=args)
    sbn_controller.mobility_manager = MobilityManager(args)
    sbn_controller.sfc_manager = SFCManager(args,backup_manager=BackupManager(args=args),alg=ALG)
    sbn_controller.sfc_instantiator = SFCInstatiator(ALG)

    sbn_controller.sfc_manager.alg_name = args.alg
    sbn_controller.verbose = (args.verbose == 'y')
    sbn_controller.output_writter = OutputWritter(topology, *create_output_dir(args, topology))
    sbn_controller.flows = int(args.n_sessions)
    sbn_controller.players = int(args.n_players)


    try:
        sbn_controller.start()
    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário (Ctrl+C).")
        sys.exit(0)
        # Aqui você pode fazer algum cleanup, se necessário
    # sbn_controller.allow_high_latency = allow_delay

main()