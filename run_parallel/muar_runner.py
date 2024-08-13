import os
import argparse
from multiprocessing import Pool
from datetime import datetime as dt
import time

def run_process(process):
    """Executa um comando em um subprocesso."""
    os.system('python {}'.format(process))
    print(process)

if __name__ == '__main__':
    # Análise de argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Select MUAR arguments')
    parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
    parser.add_argument('--n_players', type=int, help='(int) number of players', default=8)
    parser.add_argument('--threads', type=int, help='(int) number of cores to use', default=30)
    parser.add_argument('--repetition', type=int, help='(int) repetitions', default=30)
    parser.add_argument('--sfc', type=str, help='(str) on or off', default='on')
    parser.add_argument('--alg', type=str, help='(str) algorithm name', default='goku') # osfem, msf, musfico, ga;...
    parser.add_argument('--share', type=str, help='(str) whether to share sfs or not', default='y')
    parser.add_argument('--shareband', type=str, help='(str) whether to share sfs or not', default='y')
#     parser.add_argument('--servers_to_crash', type=str, nargs='+', help='(list) list of reliability values',default=[3]) #0.95, 0.975, 0.99
    parser.add_argument('--verbose',   type=str, help='verbose log', default='n')
    parser.add_argument('--time', type=int, help='(int) the total time for the simulation in seconds', default=120)
    args = parser.parse_args()

    pool = Pool(processes=args.threads)
    begin = dt.now()

    # Construção dos comandos a serem executados
    cmd = []
    #for servers_to_crash in args.servers_to_crash:
    for _ in range(args.repetition):
        command = './muar.py' + \
                  ' --n_sessions ' + str(args.n_sessions) + \
                  ' --alg ' + args.alg + \
                  ' --sfc ' + args.sfc + \
                  ' --n_players ' + str(args.n_players) + \
                  ' --time ' + str(args.time)
        cmd.append(command)

    # Executar cada comando com um atraso de 1 segundo entre eles
    for command in cmd:
        pool.apply_async(run_process, (command,))
        time.sleep(0.5)  # Atraso antes de iniciar o próximo comando

    pool.close()  # Nenhum outro trabalho será adicionado
    pool.join()  # Esperar por todos os processos terminarem

    duration = dt.now() - begin
    print('Processing time:', duration)
    print('Finished')
