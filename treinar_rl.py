import os
from pathlib import Path

import gymnasium as gym
import numpy as np

# A importação do ActionMasker foi REMOVIDA
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback

os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    # Certifique-se que o new_environment.py modificado seja importado
    from new_environment import NetworkEnv
    from salvar_var import carregar_lista
except ImportError:
    print("Erro: Verifique se 'new_environment.py' e 'salvar_var.py' estão na mesma pasta.")
    exit()

def load_user_data():
    """
    Função para carregar todos os dados e parâmetros do seu problema.
    """
    try:
        lista_grafos = carregar_lista("lista_grafo")
        lista_sfcs = carregar_lista("lista_sfc")
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se que os arquivos 'lista_grafo' e 'lista_sfc' existem.")
        return None, None, None, None

    pesos = {"cpu": 5, "cache": 5, "latency": 2, "band": 2}
    valid_nodes = []
    if lista_grafos and lista_grafos[0]:
        for node in lista_grafos[0].nodes():
            if lista_grafos[0].nodes[node]["type"] == "server":
                valid_nodes.append(node)
    
    valid_nodes.append("M")
    print(f"Dados carregados: {len(lista_grafos)} grafos e {len(lista_sfcs)} SFCs.")
    return lista_grafos, lista_sfcs, valid_nodes, pesos

def train_rl_model():
    """
    Função principal que executa o pipeline de treinamento.
    """
    print("Passo 1: Carregando dados do usuário...")
    list_graph, list_sfc, valid_nodes, pesos = load_user_data()

    if not all([list_graph, list_sfc, valid_nodes, pesos]):
        print("Erro: Falha ao carregar os dados.")
        return

    TOTAL_TIMESTEPS = 200_000
    LOG_DIR = "./rl_logs/"
    SAVE_DIR = "./saved_rl_models/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\nPasso 2: Configurando o ambiente e o agente MaskablePPO...")
    
    # Criamos a instância do seu ambiente diretamente. NENHUM WRAPPER É NECESSÁRIO.
    train_env = NetworkEnv(list_graph=list_graph, valid_nodes=valid_nodes, list_sfc=list_sfc, pesos=pesos)

    model_params = {
        'n_steps': 2048, 'ent_coef': 0.01, 'learning_rate': 0.0003, 'gamma': 0.99,
        'gae_lambda': 0.95, 'clip_range': 0.2, 'verbose': 1,
        'tensorboard_log': LOG_DIR, 'device': 'cpu'
    }

    model_path = Path(SAVE_DIR) / "ppo_sfc_allocation.zip"
    
    if model_path.exists():
        print(f"\nModelo salvo encontrado em '{model_path}'. Carregando para continuar o treinamento...")
        model = MaskablePPO.load(model_path, env=train_env, **model_params)
    else:
        print("\nNenhum modelo salvo encontrado. Inicializando novo agente MaskablePPO...")
        model = MaskablePPO("MultiInputPolicy", train_env, **model_params)

    print("\nPasso 3: Configurando o callback de avaliação...")
    
    # O ambiente de avaliação também não precisa de wrapper.
    eval_env = NetworkEnv(list_graph=list_graph, valid_nodes=valid_nodes, list_sfc=list_sfc, pesos=pesos)
    
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=SAVE_DIR, log_path=LOG_DIR,
        eval_freq=5000, n_eval_episodes=50, deterministic=True, render=False
    )

    print(f"\nPasso 4: Iniciando o treinamento por {TOTAL_TIMESTEPS} timesteps...")
    print("=" * 50)
    print(f"Para monitorar o treinamento, execute: tensorboard --logdir {LOG_DIR}")
    print("=" * 50)
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback,
        progress_bar=True, reset_num_timesteps=False
    )
    
    print("\nTreinamento concluído!")
    print("\nPasso 5: Salvando o último estado do modelo para continuidade...")
    model.save(model_path)
    print(f"Último modelo salvo em: {model_path}")

    print("\n--- Demonstração do MELHOR modelo encontrado ---")
    
    best_model_path = Path(SAVE_DIR) / "best_model.zip"
    if best_model_path.exists():
        print(f"Carregando o melhor modelo de: {best_model_path}")
        # Carregue o modelo sem especificar o env, ele usará a política salva.
        loaded_model = MaskablePPO.load(best_model_path)
    else:
        print("AVISO: 'best_model.zip' não encontrado. Usando o último modelo treinado.")
        loaded_model = model

    N_DEMO_EPISODES = 500
    successful_allocations = 0
    demo_env = NetworkEnv(list_graph=list_graph, valid_nodes=valid_nodes, list_sfc=list_sfc, pesos=pesos)
    
    for _ in range(N_DEMO_EPISODES):
        obs, _ = demo_env.reset()
        done = False
        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True, action_masks=demo_env.action_masks())
            obs, reward, terminated, truncated, info = demo_env.step(action)
            done = terminated or truncated
        
        print(demo_env.servers_used)
        if demo_env.success:
            successful_allocations += 1
        
    success_rate = successful_allocations / N_DEMO_EPISODES
    print(f"\nTaxa de alocações bem-sucedidas em {N_DEMO_EPISODES} episódios: {success_rate:.2%}")

if __name__ == '__main__':
    train_rl_model()