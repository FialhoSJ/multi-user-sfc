import argparse
import os
import sys
import time
from datetime import datetime as dt
from multiprocessing import Pool


def run_process(process):
    """Executa um comando em um subprocesso."""
    # Chama o python explicitamente usando o executável atual
    command_to_run = f"{sys.executable} {process}"
    os.system(command_to_run)


def process_callback(process_name):
    """Função chamada quando um processo termina."""
    pass


if __name__ == "__main__":
    # =========================================================================
    # ARGUMENTOS (Ignorados no modo BATCH, mas mantidos para não quebrar)
    # =========================================================================
    parser = argparse.ArgumentParser(description="Select MUAR arguments")
    parser.add_argument("--n_sessions", type=int, default=50)
    parser.add_argument("--n_players", type=int, default=6)
    parser.add_argument("--time", type=int, default=120)
    parser.add_argument("--eco_effi_ratio", type=float, default=0.7)
    parser.add_argument("--sfc", type=str, default="on")
    args, unknown = parser.parse_known_args()

    # ========================================================================
    # CONFIGURAÇÃO DE EXECUÇÃO (ATUALIZADA)
    # =========================================================================

    RUN_MODE = "batch"

    # 1. Lista de Algoritmos (Adicionados GA e GreedyB)
    BATCH_ALGS = ["msf", "musfico", "vegeta", "greedyb"]

    # 2. Cenários de Risco
    BATCH_FAIL_TARGETS = ["low_risk", "med_risk", "high_risk", "all"]

    # 3. Número de repetições por cenário
    BATCH_TOTAL_RUNS = 20 // len(BATCH_FAIL_TARGETS)

    # ATENÇÃO: Isso vai disparar 27 processos Python pesados simultaneamente.
    # Certifique-se de que sua máquina aguenta (CPU/RAM).
    BATCH_PARALLEL_RUNS = 20

    # --- Configurações de Falha ---
    number_of_fails = ["3"]
    CRASH_AT_TIME = [400, 520, 640]
    avas = ["0.99"]

    # =========================================================================
    # LÓGICA DE GERAÇÃO DE COMANDOS
    # =========================================================================

    begin = dt.now()
    cmd = []

    print(f"[INFO] Modo: {RUN_MODE}")
    print(f"[INFO] Algoritmos: {BATCH_ALGS}")
    print(f"[INFO] Cenários: {BATCH_FAIL_TARGETS}")
    print(f"[INFO] Configuração: 3 falha em T={CRASH_AT_TIME}s")
    print(f"[INFO] Total de Jobs: {len(BATCH_ALGS) * len(BATCH_FAIL_TARGETS) * BATCH_TOTAL_RUNS}")

    # Loop para gerar os comandos
    for alg_name in BATCH_ALGS:
        for fail_target in BATCH_FAIL_TARGETS:
            for i in range(BATCH_TOTAL_RUNS):
                for a in avas:
                    for n in number_of_fails:
                        command = (
                            "main.py"
                            + " --n_sessions "
                            + str(args.n_sessions)
                            + " --alg "
                            + alg_name
                            + " --fail_target "
                            + fail_target
                            + " --sfc "
                            + str(args.sfc)
                            + " --ava "
                            + str(a)
                            + " --number_of_fails "
                            + str(n)
                            + " --crash_at "
                            + " ".join(map(str, CRASH_AT_TIME))
                            + " --n_players "
                            + str(args.n_players)
                            + " --sfc_lifetime "
                            + str(args.time)
                            + " --eco_effi_ratio "
                            + str(args.eco_effi_ratio)
                            + " --verbose n"
                        )

                        cmd.append(command)

    # =========================================================================
    # EXECUÇÃO DO POOL
    # =========================================================================

    print(f"[INFO] Iniciando pool com {BATCH_PARALLEL_RUNS} processos...")
    print("[AVISO] Rodando tudo simultaneamente. Monitore o uso de CPU/RAM.")

    pool = Pool(processes=BATCH_PARALLEL_RUNS)

    for i, command in enumerate(cmd):
        pool.apply_async(run_process, (command,), callback=lambda c=command: process_callback(c))
        time.sleep(1.0)  # Pequeno delay para evitar conflito de I/O na criação de logs

    pool.close()
    pool.join()

    duration = dt.now() - begin
    print("\n==================================================")
    print(f"Execução finalizada em: {duration}")
    print("==================================================")
