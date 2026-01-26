import os
import argparse
from multiprocessing import Pool
from datetime import datetime as dt
import time
import sys

def run_process(process):
    """Executa um comando em um subprocesso."""
    # Chama o python explicitamente usando o executável atual
    command_to_run = f'{sys.executable} {process}'
    os.system(command_to_run)

def process_callback(process_name):
    """Função chamada quando um processo termina."""
    print(f"Processo finalizado: {process_name}")

if __name__ == '__main__':
    # =========================================================================
    # ARGUMENTOS (Mantidos para compatibilidade, mas ignorados no modo BATCH)
    # =========================================================================
    parser = argparse.ArgumentParser(description='Select MUAR arguments')
    parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
    parser.add_argument('--n_players', type=int, help='(int) number of players', default=6)
    parser.add_argument('--threads', type=int, help='(int) number of cores to use', default=17)
    parser.add_argument('--repetition', type=int, help='(int) repetitions', default=5)
    parser.add_argument('--sfc', type=str, help='(str) on or off', default='on')
    parser.add_argument('--alg', type=str, help='(str algorithm name', default='ga')
    parser.add_argument('--share', type=str, help='(str) whether to share sfs or not', default='y')
    parser.add_argument('--shareband', type=str, help='(str) whether to share sfs or not', default='y')
    parser.add_argument('--verbose', type=str, help='verbose log', default='n')
    parser.add_argument('--time', type=int, help='(int) the total time for the simulation in seconds', default=120)
    parser.add_argument(
        '--eco_effi_ratio',
        type=float,
        default=0.7,
        help='Define a proporção para slots econômicos.'
    )
    args = parser.parse_args()

    # =========================================================================
    # CONFIGURAÇÃO DE EXECUÇÃO (CUSTOMIZADA)
    # =========================================================================
    
    # Define o modo de execução para 'batch' para ignorar argumentos de linha de comando
    # e usar as listas definidas abaixo.
    RUN_MODE = 'batch'
    
    # 1. Algoritmos solicitados
    BATCH_ALGS = ['kuririnMaskablePPO', 'ga', 'greedyb']
    # BATCH_ALGS = ['greedyb']

    
    # 2. Número de vezes que CADA algoritmo vai rodar
    BATCH_TOTAL_RUNS = 5  
    
    # 3. Quantos processos rodam ao mesmo tempo (Threads)
    BATCH_PARALLEL_RUNS = 5 
    
    # Parâmetros alinhados com o padrão do main.py
    # ava='0.99', number_of_fails='50'
    avas = ['0.99']
    number_of_fails = ['40']

    # =========================================================================
    # LÓGICA DE GERAÇÃO DE COMANDOS
    # =========================================================================
    
    begin = dt.now()
    cmd = []
    num_parallel_processes = BATCH_PARALLEL_RUNS

    print(f"[INFO] Modo: {RUN_MODE}")
    print(f"[INFO] Algoritmos: {BATCH_ALGS}")
    print(f"[INFO] Repetições por Algoritmo: {BATCH_TOTAL_RUNS}")
    print(f"[INFO] Processos em Paralelo: {BATCH_PARALLEL_RUNS}")

    # Loop principal para gerar os comandos
    for alg_name in BATCH_ALGS:
        for i in range(BATCH_TOTAL_RUNS):
            for a in avas: 
                for n in number_of_fails:
                    # Constrói o comando garantindo verbose='n' e argumentos do main
                    command = 'main.py' + \
                        ' --n_sessions ' + str(args.n_sessions) + \
                        ' --alg ' + alg_name + \
                        ' --sfc ' + str(args.sfc) + \
                        ' --ava ' + str(a) + \
                        ' --number_of_fails ' + str(n) + \
                        ' --n_players ' + str(args.n_players) + \
                        ' --time ' + str(args.time) + \
                        ' --eco_effi_ratio ' + str(args.eco_effi_ratio) + \
                        ' --verbose n'  # Forçando não verboso
                    
                    cmd.append(command)

    # =========================================================================
    # EXECUÇÃO DO POOL
    # =========================================================================

    print(f"[INFO] Total de comandos gerados: {len(cmd)}")
    print(f"[INFO] Iniciando pool com {num_parallel_processes} processos...")
    
    pool = Pool(processes=num_parallel_processes)

    for command in cmd:
        pool.apply_async(run_process, (command,), callback=lambda c=command: process_callback(c))
        # Pequeno delay para evitar conflito na criação de pastas de log iniciais
        time.sleep(2.0)

    pool.close()
    pool.join()

    duration = dt.now() - begin
    print('Tempo total de processamento:', duration)
    print('Execução finalizada.')