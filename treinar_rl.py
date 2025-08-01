import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
# NOVO: Importações para o treinamento em paralelo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Assumindo que suas funções e classes personalizadas estão disponíveis
# Se 'salvar_var' e 'env_sfc_allocation' estiverem em outros arquivos,
# certifique-se de que eles possam ser importados corretamente.
from salvar_var import carregar_lista
from env_sfc_allocation.env_sfc_allocation import SFC_AllocationEnv

# Importações do Stable Baselines3 e SB3-Contrib
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks # Importação corrigida para o teste final

# ==============================================================================
#      FUNÇÃO PARA CARREGAR O AMBIENTE (Seu código original)
# ==============================================================================
def carregar_dados_do_ambiente():
    """
    Carrega os dados e inicializa o ambiente SFC_AllocationEnv.
    (Esta é a sua função, mantida como está).
    """
    try:
        list_graph = carregar_lista("list_graph")
        list_aux_sfc = carregar_lista("list_sfc")
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se que os arquivos de dados existem.")
        return None

    list_sfc = []
    session_id = '1'
    list_aux = []
    if list_aux_sfc:
        for idx, sfc in enumerate(list_aux_sfc):
            session_sfc = sfc.id.split("_")[-1]
            if session_sfc != session_id:
                list_sfc.append(list_aux)
                list_aux = []
                session_id = sfc.id.split("_")[-1]
            list_aux.append(sfc)
        list_sfc.append(list_aux)

    valid_nodes = []
    if list_graph and list_graph[0]:
        for node in list_graph[0].nodes():
            if list_graph[0].nodes[node]["type"] == "server":
                valid_nodes.append(node)
    valid_nodes.append("M")
    
    pesos = {"cpu": 1.0, "cache": 1.0, "lat": 1.0, "band": 1.0}
    
    return SFC_AllocationEnv(list_graph_per_session=list_graph, list_sfcs_per_session=list_sfc, valid_nodes=valid_nodes, pesos_fatores=pesos)

# ==============================================================================
#               FLUXO PRINCIPAL DE TREINAMENTO
# ==============================================================================

# NOVO: É crucial usar este bloco para o treinamento em paralelo funcionar corretamente
if __name__ == '__main__':
    # --- 1. DEFINIÇÃO DOS DIRETÓRIOS (MODIFICADO) ---
    log_dir = "logs/"
    tensorboard_log_dir = "tensorboard_logs/"
    # Novo diretório para salvar o modelo final
    save_dir = "rl_saved_models/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


    # --- 2. CRIAÇÃO DOS AMBIENTES ---
    # MODIFICADO: Criação de ambientes paralelos para treinamento
    num_cpu = 4  # Defina o número de processos paralelos (geralmente o número de núcleos da CPU)
    print(f"Iniciando treinamento com {num_cpu} processos paralelos.")
    env = make_vec_env(carregar_dados_do_ambiente, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

    # O ambiente de avaliação permanece único para garantir consistência nos testes
    eval_env = carregar_dados_do_ambiente()
    eval_env = Monitor(eval_env)


    # --- 3. CONFIGURAÇÃO DO CALLBACK DE AVALIAÇÃO (MODIFICADO) ---
    eval_callback = MaskableEvalCallback(
        eval_env,
        log_path=log_dir,
        eval_freq=1000,
        n_eval_episodes=30,
        deterministic=True,
        render=False,
    )

    # --- 4. CRIAÇÃO E TREINAMENTO DO MODELO ---
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir
    )

    # Inicia o treinamento com o callback para gerar dados para os gráficos
    print("--- Iniciando o treinamento ---")
    # MODIFICADO: Aumento do número total de passos de treinamento
    model.learn(
        total_timesteps=300_000,
        callback=eval_callback,
        tb_log_name="MaskablePPO_SFC_Allocation_Parallel"
    )
    print("--- Treinamento finalizado ---")


    # --- 5. SALVAR O MODELO FINAL (MODIFICADO) ---
    final_model_path = os.path.join(save_dir, "ppo_allocation_model")
    model.save(final_model_path)

    print(f"\nModelo final salvo em: {final_model_path}.zip")

    # --- 6. TESTE COM O MODELO FINAL SALVO (OPCIONAL) ---
    print("\nCarregando o modelo final para um teste...")
    loaded_model = MaskablePPO.load(final_model_path)

    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    while not done:
        action_masks = get_action_masks(eval_env)
        action, _ = loaded_model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        print(eval_env.env.servers_used)
        total_reward += reward
        done = terminated or truncated

    print(f"Recompensa total do episódio de teste com o modelo final: {total_reward}")
    print(eval_env.env.success)