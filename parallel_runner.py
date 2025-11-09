import os
import argparse
from multiprocessing import Pool
from datetime import datetime as dt
import time
import sys  # 1. Importe o módulo sys (já estava no seu arquivo)

def run_process(process):
    """Executa um comando em um subprocesso."""
    # 2. Use sys.executable para chamar o python explicitamente
    command_to_run = f'{sys.executable} {process}'
    os.system(command_to_run)
    # A impressão foi movida para o 'apply_async' para melhor feedback
    # print(process)

def process_callback(process_name):
    """Função chamada quando um processo termina."""
    print(f"Processo finalizado: {process_name}")

if __name__ == '__main__':
    # Análise de argumentos de linha de comando
    # (TODA A SEÇÃO ARGPARSE ESTÁ IDÊNTICA AO ORIGINAL)
    parser = argparse.ArgumentParser(description='Select MUAR arguments')
    parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
    parser.add_argument('--n_players', type=int, help='(int) number of players', default=6)
    parser.add_argument('--threads', type=int, help='(int) number of cores to use', default=1)
    parser.add_argument('--repetition', type=int, help='(int) repetitions', default=1)
    parser.add_argument('--sfc', type=str, help='(str) on or off', default='on')
    parser.add_argument('--alg', type=str, help='(str  algorithm name', default='hephaestus') # osfem, msf, musfico, ga;...
    parser.add_argument('--share', type=str, help='(str) whether to share sfs or not', default='y')
    parser.add_argument('--shareband', type=str, help='(str) whether to share sfs or not', default='y')
#     parser.add_argument('--servers_to_crash', type=str, nargs='+', help='(list) list of reliability values',default=[3]) #0.95, 0.975, 0.99
    parser.add_argument('--verbose',   type=str, help='verbose log', default='n')
    parser.add_argument('--time', type=int, help='(int) the total time for the simulation in seconds', default=120)
    parser.add_argument(
    '--eco_effi_ratio',
    type=float,
    default=0.7,
    help='Define a proporção para slots econômicos (ex: 0.7 significa 70%% eco e 30%% eficiência).'
    )
    args = parser.parse_args()

    # =========================================================================
    # ### CONTROLE DE EXECUÇÃO (NOVO) ###
    # =========================================================================
    #
    # Altere a variável RUN_MODE para selecionar a lógica de execução:
    #
    # 'single': Modo original. 
    #           Usa os argumentos: --alg, --repetition, --threads
    #
    # 'batch':  Modo em lote (X, Y). 
    #           Usa as variáveis BATCH_* definidas abaixo.
    #           (Ignora --alg, --repetition, --threads)
    #
    RUN_MODE = 'batch'
    
    # --- Configurações do MODO BATCH (ignoradas se RUN_MODE = 'single') ---
    BATCH_ALGS = ['hephaestusMaskablePPO', 'darsppoMaskablePPO', 'ga', "kuririnMaskablePPO"] # (Lista) Lista de algoritmos
    BATCH_TOTAL_RUNS = 15                    # (X) Execuções totais POR algoritmo
    BATCH_PARALLEL_RUNS = 15                # (Y) Execuções simultâneas (pool)
    # =========================================================================

    
    begin = dt.now()
    cmd = []
    num_parallel_processes = 1 # Valor padrão

    # Parâmetros fixos (comuns a ambos os modos)
    avas = ['1.0']
    number_of_fails = ['3']

    if RUN_MODE == 'single':
        # --- MODO SINGLE: Lógica original preservada ---
        print(f"[INFO] Executando em MODO SINGLE (lógica original, via argparse)")
        print(f"[INFO] Algoritmo: {args.alg}, Repetições: {args.repetition}, Threads: {args.threads}")
        num_parallel_processes = args.threads # <-- Usa --threads

        for a in avas: 
            for n in number_of_fails: 
                for _ in range(args.repetition): # <-- Usa --repetition
                    command = './main.py' + \
                        ' --n_sessions ' + str(args.n_sessions) + \
                        ' --alg ' + args.alg + \
                        ' --sfc ' + args.sfc + \
                        ' --ava ' + str(a) + \
                        ' --number_of_fails ' + str(n) + \
                        ' --n_players ' + str(args.n_players) + \
                        ' --time ' + str(args.time) + \
                        ' --eco_effi_ratio ' + str(args.eco_effi_ratio) 
                    cmd.append(command)

    elif RUN_MODE == 'batch':
        # --- MODO BATCH: Nova lógica (X, Y, Lista de Algs) ---
        print(f"[INFO] Executando em MODO BATCH (lógica customizada)")
        print(f"[INFO] Algoritmos: {', '.join(BATCH_ALGS)}")
        print(f"[INFO] Execuções por Alg (X): {BATCH_TOTAL_RUNS}")
        print(f"[INFO] Paralelismo (Y): {BATCH_PARALLEL_RUNS}")
        
        num_parallel_processes = BATCH_PARALLEL_RUNS # <-- Usa Y (variável)
        
        # 1. Loop sobre a lista de algoritmos
        for alg_name in BATCH_ALGS:
            # 2. Loop para 'X' (total_runs) execuções
            for i in range(BATCH_TOTAL_RUNS):
                for a in avas: 
                    for n in number_of_fails:
                        # Nota: Os args originais (n_sessions, n_players, etc.) ainda são usados
                        command = './main.py' + \
                            ' --n_sessions ' + str(args.n_sessions) + \
                            ' --alg ' + alg_name + \
                            ' --sfc ' + str(args.sfc) + \
                            ' --ava ' + str(a) + \
                            ' --number_of_fails ' + str(n) + \
                            ' --n_players ' + str(args.n_players) + \
                            ' --time ' + str(args.time) + \
                            ' --eco_effi_ratio ' + str(args.eco_effi_ratio) 
                        cmd.append(command)
    else:
        print(f"ERRO: RUN_MODE '{RUN_MODE}' desconhecido. Use 'single' ou 'batch'.", file=sys.stderr)
        sys.exit(1)


    # =========================================================================
    # ### EXECUÇÃO DO POOL (Comum aos dois modos) ###
    # =========================================================================

    print(f"[INFO] Total de comandos a serem executados: {len(cmd)}")
    print(f"[INFO] Iniciando pool com {num_parallel_processes} processos paralelos...")
    
    pool = Pool(processes=num_parallel_processes)

    # Executar cada comando com um atraso de 3 segundos entre as submissões
    for command in cmd:
        # Usando 'callback' para imprimir quando o processo REALMENTE terminar
        pool.apply_async(run_process, (command,), callback=lambda c=command: process_callback(c))
        time.sleep(3.0)  # Atraso antes de submeter o próximo comando

    pool.close()  # Nenhum outro trabalho será adicionado
    pool.join()  # Esperar por todos os processos terminarem

    duration = dt.now() - begin
    print('Processing time:', duration)
    print('Finished')