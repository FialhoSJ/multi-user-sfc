import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from pathlib import Path

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# --- Importações do seu ambiente e dados ---
# Certifique-se que estes arquivos estão na mesma pasta que este script.
try:
    from new_environment import NetworkEnv
    from salvar_var import carregar_lista
except ImportError:
    print("Erro: Verifique se 'new_environment.py' e 'salvar_var.py' estão na mesma pasta.")
    exit()

# ==============================================================================
# --- CONFIGURAÇÃO DO USUÁRIO (TODO) ---
# Você só precisa modificar esta seção para carregar seus dados.
# ==============================================================================

def load_user_data():
    """
    Função para carregar todos os dados e parâmetros do seu problema.
    Modifique o corpo desta função para carregar seus dados reais.
    """
    try:
        lista_grafos = carregar_lista("lista_grafo")
        lista_sfcs = carregar_lista("lista_sfc")
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se que os arquivos 'lista_grafo' e 'lista_sfc' existem.")
        return None, None, None, None

    pesos = {"cpu": 4, "cache": 4, "latency": 1.5, "band": 1.5}  # Exemplo de pesos

    valid_nodes = []
    if lista_grafos and lista_grafos[0]:
        for node in lista_grafos[0].nodes():
            if lista_grafos[0].nodes[node]["type"] == "server":
                valid_nodes.append(node)
    
    valid_nodes.append("M")  # Nó de migração
    print(len(lista_grafos), " || ",len(lista_sfcs))
    return lista_grafos, lista_sfcs, valid_nodes, pesos

# ==============================================================================
# --- SCRIPT DE TREINAMENTO AUTOMATIZADO ---
# Nenhuma modificação necessária abaixo desta linha.
# ==============================================================================

def train_rl_model():
    """
    Função principal que executa o pipeline de treinamento do agente de RL.
    """
    # --- Passo 1: Carregamento dos Dados ---
    print("Passo 1: Carregando dados do usuário...")
    list_graph, list_sfc, valid_nodes, pesos = load_user_data()

    if not all([list_graph, list_sfc, valid_nodes, pesos]):
        print("Erro: Falha ao carregar os dados. Verifique a função 'load_user_data'.")
        return

    # --- Definição de Parâmetros de Treinamento ---
    
    ### ALTERAÇÃO E JUSTIFICATIVA ###
    # O valor original (len(list_graph) * 75) era extremamente baixo.
    # RL precisa de centenas de milhares de passos para aprender tarefas complexas.
    # Monitore a recompensa no TensorBoard para ver se este valor é suficiente
    TOTAL_TIMESTEPS = 6*4_500_000

    N_ENVS = 8  # Número de ambientes em paralelo.
    LOG_DIR = "./rl_logs/"
    SAVE_DIR = "./saved_rl_models/"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Passo 2: Configuração Inteligente do Agente PPO ---
    print("\nPasso 2: Configurando o ambiente e o agente PPO...")
    
    def env_creator():
        """Função wrapper para criar o ambiente."""
        env = NetworkEnv(list_graph=list_graph, valid_nodes=valid_nodes, list_sfc=list_sfc, pesos=pesos)
        # As definições de space devem ser feitas dentro do ambiente, mas podemos garantir aqui.
        env.action_space = gym.spaces.Discrete(len(valid_nodes))
        env.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(valid_nodes) * 6,), dtype=np.float32
        )
        return env

    # Cria ambientes vetorizados para treinamento em paralelo
    train_env = make_vec_env(env_creator, n_envs=N_ENVS)

    ### ALTERAÇÃO E JUSTIFICATIVA ###
    # Hiperparâmetros foram unificados em um dicionário para garantir consistência
    # ao criar um novo modelo ou ao continuar o treinamento de um existente.
    # Os valores foram ajustados para serem mais robustos e alinhados com os padrões do PPO.
    model_params = {
        'n_steps': 2048,           # Tamanho do buffer de rollout. Mais estável que valores pequenos.
        'ent_coef': 0.1,          # Coeficiente de entropia para incentivar a exploração.
        'learning_rate': 0.0003,   # Taxa de aprendizado.
        'gamma': 0.99,             # Fator de desconto para recompensas futuras.
        'gae_lambda': 0.95,        # Fator do General Advantage Estimation.
        'clip_range': 0.2,         # Range de clipping do PPO.
        'verbose': 1,
        'tensorboard_log': LOG_DIR,
        'device': 'cpu'
    }

    model_path = Path(SAVE_DIR) / "ppo_sfc_allocation.zip"
    if model_path.exists():
        print(f"\nModelo salvo encontrado em '{model_path}'. Carregando para continuar o treinamento...")
        # Ao carregar, passamos os parâmetros para garantir que o buffer e outras configurações sejam consistentes.
        model = PPO.load(model_path, env=train_env, **model_params)
    else:
        print("\nNenhum modelo salvo encontrado. Inicializando novo agente PPO...")
        model = PPO("MlpPolicy", train_env, **model_params)

    # --- Passo 3: Treinamento com Avaliação Contínua ---
    print("\nPasso 3: Configurando o callback de avaliação...")
    
    eval_env = env_creator() # Ambiente único para avaliação

    ### ALTERAÇÃO E JUSTIFICATIVA ###
    # A frequência de avaliação foi ajustada. Avaliar com muita frequência (a cada poucos passos)
    # torna o treinamento muito lento. Uma avaliação a cada 5000-10000 passos é mais razoável.
    EVAL_FREQ_PER_ENV = max(5000 // N_ENVS, 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ_PER_ENV,
        n_eval_episodes=50, # 50 episódios é um bom número para ter uma média confiável.
        deterministic=True,
        render=False
    )

    # --- Passo 4: Executar o Treinamento ---
    print(f"\nPasso 4: Iniciando o treinamento por {TOTAL_TIMESTEPS} timesteps...")
    print("=" * 50)
    print("Para monitorar o treinamento, abra um novo terminal e execute:")
    print(f"tensorboard --logdir {LOG_DIR}")
    print("=" * 50)
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
        # Salva o modelo no final, para que possamos continuar o treinamento depois.
        reset_num_timesteps=False 
    )
    
    print("\nTreinamento concluído!")

    # --- Passo 5: Salvar o Último Modelo e Usar o MELHOR para Demonstração ---
    print("\nPasso 5: Salvando o último estado do modelo para continuidade...")
    # Salva o estado final do modelo para que o treinamento possa ser retomado.
    model.save(model_path)
    print(f"Último modelo salvo em: {model_path}")

    ### ALTERAÇÃO E JUSTIFICATIVA ###
    # A lógica foi corrigida para usar o MELHOR modelo encontrado durante o treinamento
    # (salvo pelo EvalCallback como 'best_model.zip') para a demonstração final.
    # Este modelo tem o melhor desempenho médio generalizado, não sendo apenas o último.
    print("\nDemonstração do MELHOR modelo no ambiente de avaliação:")
    
    best_model_path = Path(SAVE_DIR) / "best_model.zip"
    if best_model_path.exists():
        print(f"Carregando o melhor modelo de: {best_model_path}")
        loaded_model = PPO.load(best_model_path)
    else:
        print(f"AVISO: 'best_model.zip' não encontrado. Usando o último modelo treinado para a demonstração.")
        loaded_model = model

    ### ALTERAÇÃO E JUSTIFICATIVA ###
    # O número de episódios de demonstração foi fixado em um valor maior para
    # obter uma métrica de sucesso estatisticamente mais relevante.
    N_DEMO_EPISODES = 100
    successful_allocations = 0

    for episode in range(N_DEMO_EPISODES):
        obs, info = eval_env.reset()
        done = False
        while not done:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
        
        # Após o episódio terminar (done=True), verificamos se foi um sucesso.
        if eval_env.success:
            successful_allocations += 1

    success_rate = successful_allocations / N_DEMO_EPISODES
    print(f"\nTaxa de alocações bem-sucedidas em {N_DEMO_EPISODES} episódios: {success_rate:.2%}")

if __name__ == '__main__':
    # O loop foi removido, pois a lógica de treinamento contínuo já está
    # implementada através do carregamento do modelo salvo.
    # Basta executar o script novamente para continuar o treinamento.
    train_rl_model()