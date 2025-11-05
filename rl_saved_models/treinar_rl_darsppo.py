#=====Mecanismo para resolver importação relativa==================
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#============================

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
# REMOVIDO: a importação de make_vec_env e DummyVecEnv não são mais necessárias
from stable_baselines3.common.logger import configure

from utils.salvar_var import carregar_lista
from algorithms.environments.env_da_rsppo import SFC_AllocationEnv_DARSPPO

# --- IMPORTAÇÕES ADICIONADAS ---
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
# --------------------------------

# --- FLAG DE CONTROLE ADICIONADA ---
USE_MASKING = False
# -----------------------------------

# ==============================================================================
#      FUNÇÃO PARA CARREGAR O AMBIENTE (Seu código original, sem alterações)
# ==============================================================================
def carregar_dados_do_ambiente():
    """
    Carrega os dados e inicializa o ambiente SFC_AllocationEnv_DARSPPO.
    """
    try:
        list_graph = []
        list_sfc = []
        for i in range(1, 5):
            list_graph = list_graph + carregar_lista(f"list_graph{i}")
            list_sfc = list_sfc + carregar_lista(f"list_sfc{i}")
            
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se que os arquivos de dados existem.")
        return None

    valid_nodes = []
    if list_graph and list_graph[0]:
        for node in list_graph[0].nodes():
            if list_graph[0].nodes[node]["type"] == "server":
                valid_nodes.append(node)
    valid_nodes.append("M")
    
    env = SFC_AllocationEnv_DARSPPO(list_graph=list_graph, list_sfc=list_sfc, valid_nodes=valid_nodes)
    # A chamada reset() não é mais necessária aqui, o Monitor cuidará disso.
    return env


# ==============================================================================
#               FLUXO PRINCIPAL DE TREINAMENTO (MODIFICADO)
# ==============================================================================

if __name__ == '__main__':
    # --- 1. DEFINIÇÃO DOS DIRETÓРИОS ---
    log_dir = "logs/"
    tensorboard_log_dir = "tensorboard_logs/"
    save_dir = "rl_saved_models/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. CRIAÇÃO DOS AMBIENTES (AGORA AMBOS SÃO AMBIENTES ÚNICOS) ---
    print("Iniciando com um único processo (sem paralelismo).")
    
    # Cria o ambiente de treino como um ambiente único e o envolve com Monitor
    # para registrar estatísticas de recompensa, passos, etc.
    train_env = carregar_dados_do_ambiente()
    train_env = Monitor(train_env)
    
    # O ambiente de avaliação já era único, mantemos como está.
    eval_env = carregar_dados_do_ambiente()
    eval_env = Monitor(eval_env)
    
    # --- 3. CARREGAR MODELO EXISTENTE OU CRIAR UM NOVO (MODIFICADO) ---
    if USE_MASKING:
        model_name = "MASKABLEPPO_DARSPPO_allocation_model.zip"
        ModelClass = MaskablePPO
        model_log_name = "MaskablePPO_DARSPPO_Allocation"
        print("Configurado para usar MaskablePPO.")
    else:
        model_name = "PPO_DARSPPO_allocation_model.zip"
        ModelClass = PPO
        model_log_name = "PPO_DARSPPO_Allocation"
        print("Configurado para usar PPO Padrão.")

    final_model_path = os.path.join(save_dir, model_name)

    if os.path.exists(final_model_path):
        print(f"Modelo salvo encontrado em '{final_model_path}'. Carregando para continuar o treinamento...")
        model = ModelClass.load(final_model_path, env=train_env) # Usa ModelClass
        new_logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])
        model.set_logger(new_logger)
    else:
        print(f"Nenhum modelo salvo encontrado. Iniciando novo treinamento para {model_name}...")
        model = ModelClass( # Usa ModelClass
            "MultiInputPolicy",
            train_env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir
        )

    # --- 4. TREINAMENTO (NOVO OU CONTINUADO) (MODIFICADO) ---
    if USE_MASKING:
        print("Usando MaskableEvalCallback.")
        eval_callback = MaskableEvalCallback(
            eval_env,
            log_path=log_dir,
            eval_freq=1000,
            n_eval_episodes=30,
            deterministic=False,
            render=False
        )
    else:
        print("Usando EvalCallback padrão.")
        eval_callback = EvalCallback( # Callback Padrão
            eval_env,
            log_path=log_dir,
            eval_freq=1000,
            n_eval_episodes=30,
            deterministic=False,
            render=False
        )
    
    additional_timesteps = 150_000 # Mantido o valor original do seu script
    
    print(f"--- Iniciando/Continuando o treinamento por mais {additional_timesteps} passos ---")
    model.learn(
        total_timesteps=additional_timesteps,
        callback=eval_callback,
        tb_log_name=model_log_name, # Usa o nome do log dinâmico
        reset_num_timesteps=False
    )
    print("--- Treinamento finalizado ---")

    # --- 5. SALVAR O MODELO ATUALIZADO (MODIFICADO) ---
    model.save(final_model_path) # Salva com o nome correto
    print(f"\nModelo final salvo em: {final_model_path}")

    # --- 6. TESTE COM O MODELO FINAL (MODIFICADO) ---
    print(f"\n--- Iniciando teste com o modelo {model_name} em 1000 episódios ---")
    
    num_episodes = 1000
    all_rewards = []
    successful_runs = 0
    total_latency_on_success = 0.0

    for i in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

        while not done:
            # --- MODIFICADO: Chamada condicional do predict ---
            if USE_MASKING:
                # MaskablePPO precisa das máscaras
                action_masks = eval_env.env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
            else:
                # PPO Padrão não usa máscaras
                action, _ = model.predict(obs, deterministic=False)
            
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            total_reward += reward
            done = terminated or truncated

        all_rewards.append(total_reward)
        if eval_env.env.success: # Acesso direto ao atributo 'success'
            successful_runs += 1
            total_latency_on_success += eval_env.env.latency_used

        if (i + 1) % 100 == 0:
            success_status = eval_env.env.success
            print(f"Episódio {i + 1}/{num_episodes} concluído. Recompensa: {total_reward:.2f}, Sucesso: {success_status}")

    # --- 7. CÁLCULO E EXIBIÇÃO DAS MÉTRICAS DE DESEMPENHO ---
    print(f"\n--- Métricas de Desempenho ({num_episodes} execuções) ---")

    success_rate = (successful_runs / num_episodes) * 100 if num_episodes > 0 else 0
    mean_reward = np.mean(all_rewards) if all_rewards else 0
    std_reward = np.std(all_rewards) if all_rewards else 0
    variance = np.var(all_rewards) if all_rewards else 0
    average_latency = total_latency_on_success / successful_runs if successful_runs > 0 else 0

    print(f"Taxa de Sucesso: {success_rate:.2f}% ({successful_runs}/{num_episodes})")
    print(f"Recompensa Média: {mean_reward:.2f}")
    print(f"Desvio Padrão da Recompensa: {std_reward:.2f} (Variância: {variance:.2f})")
    print(f"Latência Média (apenas em sucessos): {average_latency:.2f}")