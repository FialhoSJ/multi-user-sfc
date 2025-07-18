import os
import gymnasium as gym
import networkx as nx
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from salvar_var import carregar_lista
# Importa a classe do seu ambiente do arquivo new_environment.py
# Certifique-se que este script está na mesma pasta que new_environment.py
try:
    from new_environment import NetworkEnv
except ImportError:
    print("Erro: Verifique se 'new_environment.py' está na mesma pasta que este script.")
    exit()

# ==============================================================================
# --- CONFIGURAÇÃO DO USUÁRIO (TODO) ---
# Você só precisa modificar esta seção.
# ==============================================================================

def load_user_data():
    """
    Função para carregar todos os dados e parâmetros do seu problema.
    Modifique o corpo desta função para carregar seus dados reais.
    """
    
    lista_grafos = carregar_lista("lista_grafo")
    lista_sfcs = carregar_lista("lista_sfc")
    pesos = {"cpu": 3, "cache": 3, "latency": 3, "band": 3}  # Exemplo de pesos

    valid_nodes = []
    for node in lista_grafos[0].nodes():
        if lista_grafos[0].nodes[node]["type"] == "server":
            valid_nodes.append(node)
    
    valid_nodes.append("M")
    
    return lista_grafos, lista_sfcs, valid_nodes, pesos

# ==============================================================================
# --- SCRIPT DE TREINAMENTO AUTOMATIZADO ---
# Nenhuma modificação necessária abaixo desta linha.
# ==============================================================================

def train_rl_model():
    """
    Função principal que executa o pipeline de treinamento do agente de RL.
    """
    # --- Carregamento dos Dados ---
    print("Passo 1: Carregando dados do usuário...")
    list_graph, list_sfc, valid_nodes, pesos = load_user_data()

    if not all([list_graph, list_sfc, valid_nodes, pesos]):
        print("Erro: Falha ao carregar os dados. Verifique a função 'load_user_data'.")
        return

    # --- Definição de Parâmetros de Treinamento ---
    TOTAL_TIMESTEPS = 50_000
    N_ENVS = 4
    
    # --- CORREÇÃO AQUI ---
    LOG_DIR = "./rl_logs/"
    SAVE_DIR = "./saved_rl_models/"  # Pasta de destino para o modelo salvo
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True) # Garante que a pasta de salvamento exista
    
    # --- Passo 2: Configuração Inteligente do Agente PPO ---
    print("\nPasso 2: Configurando o ambiente e o agente PPO...")
    
    def env_creator():
        env = NetworkEnv(list_graph=list_graph, valid_nodes=valid_nodes, list_sfc=list_sfc, pesos=pesos)
        env.action_space = gym.spaces.Discrete(len(valid_nodes))
        env.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(valid_nodes) * 5,), dtype=np.float32
        )
        return env

    train_env = make_vec_env(env_creator, n_envs=N_ENVS)
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        ent_coef=0.01,
        gamma=0.99,
        n_steps=2048,
        tensorboard_log=LOG_DIR
    )

    # --- Passo 3: Treinamento com Avaliação Contínua ---
    print("\nPasso 3: Configurando o callback de avaliação...")
    
    eval_env = env_creator()
    
    # --- CORREÇÃO AQUI ---
    # O callback agora salva o melhor modelo (best_model.zip) na pasta SAVE_DIR
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=max(model.n_steps // N_ENVS, 1),
        n_eval_episodes=50,
        deterministic=True,
        render=False
    )

    # --- Passo 4: Executar o Treinamento e Analisar ---
    print(f"\nPasso 4: Iniciando o treinamento por {TOTAL_TIMESTEPS} timesteps...")
    print("="*50)
    print("Para monitorar o treinamento, abra um novo terminal e execute:")
    print(f"tensorboard --logdir {LOG_DIR}")
    print("="*50)
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True
    )
    
    print("\nTreinamento concluído!")

    # --- Passo 5: Carregar e Usar o Melhor Modelo ---
    print("\nPasso 5: Carregando e demonstrando o melhor modelo salvo...")
    
    # --- CORREÇÃO AQUI ---
    # Caminho correto para o melhor modelo salvo pelo callback
    best_model_path = os.path.join(SAVE_DIR, 'best_model.zip')
    
    # Caminho final com o nome que você deseja
    final_model_path = os.path.join(SAVE_DIR, 'ppo_saved_model.zip')

    if os.path.exists(best_model_path):
        # Carrega o melhor modelo
        best_model = PPO.load(best_model_path)
        
        # Salva o melhor modelo com o nome final desejado
        best_model.save(final_model_path)
        print(f"Melhor modelo salvo com o nome final em: {final_model_path}")
        
        print("\nDemonstração do modelo final no ambiente de avaliação:")
        
        # Testando o modelo por alguns episódios
        for episode in range(3):
            obs, info = eval_env.reset()
            done = False
            total_reward = 0
            step = 0
            print(f"\n--- Episódio de Demonstração {episode + 1} ---")
            
            while not done:
                action, _states = best_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                
                # Mantendo sua lógica original de print
                chosen_node = eval_env.valid_nodes[action]
                print(f"Passo {step}: Ação={action} (Nó: {chosen_node}), Recompensa={reward:.2f}")
                
                total_reward += reward
                step += 1
            
            print(f"Resultado do Episódio: Recompensa Total = {total_reward:.2f}")
            print(f"Alocação bem-sucedida: {eval_env.success}")
            if eval_env.success:
                print(f"Servidores Usados: {eval_env.servers_used}")
    else:
        # Mensagem de erro corrigida para apontar para o local certo
        print(f"ERRO: Nenhum modelo foi salvo pelo callback em '{best_model_path}'. Verifique os logs de treinamento.")
if __name__ == '__main__':
    train_rl_model()