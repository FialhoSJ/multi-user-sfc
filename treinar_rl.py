import os
from salvar_var import carregar_lista
from stable_baselines3 import PPO, A2C, DQN
import random # <-- Adicione a importação de random para o passo de teste/avaliação

from new_environment import NetworkEnv

for i in range(3):
    def preparar_listas_de_lista_sfc():
        cont = 1
        listas = []
        while True:
            lista = carregar_lista(f"lista_sfc{cont}")
            if lista:
                listas.append(lista)
            else:
                break
            cont+=1


    def preparar_valid_nodes(grafo):
        valid_nodes=[]
        for node in grafo.nodes:
            if grafo.nodes[node]['type'] == 'server':
                valid_nodes.append(node)
            elif grafo.nodes[node]['type'] == 'mobile_device':
                valid_nodes.append(node)
                break

        return valid_nodes

    listas_sfcs = []
    cont = 1
    while True:
        
        lista = carregar_lista(f"lista_sfc{cont}")
        if lista:
            listas_sfcs.append(lista)
        else: 
            break
        cont+=1

    grafo = carregar_lista(f"lista_grafo1")[0]
    valid_nodes = preparar_valid_nodes(grafo)


    if __name__ == '__main__':
        # --- Parâmetros de Configuração ---
        MODELOS_DISPONIVEIS = {'ppo': PPO, 'a2c': A2C, 'dqn': DQN}
        MODEL_CHOICE = 'ppo'  # <-- ESCOLHA O MODELO AQUI (PPO, A2C, ou DQN)
        
        TIMESTEPS = 100000     # <-- Defina o total de passos de treinamento
        
        # Define os diretórios para salvar logs e modelos
        LOG_DIR = "logs/"
        MODEL_SAVE_PATH = f"saved_models_rl/{MODEL_CHOICE}_sfc_allocation"
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs("models/", exist_ok=True)
        
        # --- 1. Preparação dos Dados e Ambiente ---
        # print("1. Preparando dados e ambiente...")

        pesos = {"cpu": 3, "cache": 3, "band": 1.0, "latency": 3}

        # --- 2. Criação do Ambiente Gym ---
        # print("2. Criando o ambiente Gym...")

        env = NetworkEnv(graph=grafo, 
                        valid_nodes=valid_nodes, 
                        pesos=pesos,
                        all_sfc_lists=listas_sfcs)
        aux = env.observation_space.shape
        env.reset()  # Reseta o ambiente para o estado inicial
        
        
        # --- 3. Criação ou Carregamento do Modelo Stable Baselines3 ---
        # print("3. Verificando a existência de um modelo salvo...")
        model_class = MODELOS_DISPONIVEIS.get(MODEL_CHOICE)
        if not model_class:
            raise ValueError(f"Modelo '{MODEL_CHOICE}' não reconhecido. Escolha entre {list(MODELOS_DISPONIVEIS.keys())}")
        
        model_path_zip = f"{MODEL_SAVE_PATH}.zip"

        # VERIFICA SE O ARQUIVO DO MODELO EXISTE
        if os.path.exists(model_path_zip):
            # print(f"   -> Modelo encontrado! Carregando de '{model_path_zip}'...")
            # Carrega o modelo e o associa ao ambiente atual para continuar o treinamento
            model = model_class.load(MODEL_SAVE_PATH, env=env, verbose = 1)
        else:
            # print(f"   -> Nenhum modelo encontrado. Criando um novo modelo {MODEL_CHOICE}...")
            # "MlpPolicy" é uma política padrão que usa uma rede neural (Multi-Layer Perceptron)
            if MODEL_CHOICE == "ppo":
                PPO_KWARGS = {
                "policy": "MlpPolicy",
                "env": env,
                "n_steps": 2048,  
                "batch_size": int(2048/64),
                "learning_rate": 0.0003,
                "verbose": 1, 
                "device": 'cpu'
                }   
                model = PPO(**PPO_KWARGS)
        
        # --- 4. Treinamento do Modelo ---
        # print(f"\n4. Iniciando o treinamento por {TIMESTEPS} timesteps...")
        # O treinamento irá resetar o ambiente (se for o início) ou continuar de onde parou.
        # reset_num_timesteps=False garante que o contador de passos não seja zerado se o modelo foi carregado.
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        # print("\nTreinamento concluído!")
        
        # --- 5. Salvando o Modelo Treinado ---
        # print(f"5. Salvando o modelo em '{MODEL_SAVE_PATH}.zip'...")
        model.save(MODEL_SAVE_PATH)
        # print("Modelo salvo com sucesso!")

        # --- (Bônus) Como carregar e usar o modelo ---
        # print("\n--- Exemplo de Uso do Modelo Treinado ---")
        del model # remove o modelo antigo da memória
        
        loaded_model = model_class.load(MODEL_SAVE_PATH)
        
        obs, info = env.reset()
        done = False
        total_episode_reward = 0
        

        env.verbose = 0
        while not done:
            # Pede ao modelo treinado para escolher a melhor ação (deterministic=True)
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # print(f"Ação: {action}, Recompensa: {reward:.2f}, Done: {done}")
            # print(f"SFC {env.sfc.id}: {env.servers_used}")
            total_episode_reward += reward
            
        print(f"Episódio de teste finalizado! Recompensa total: {total_episode_reward:.2f}")
        if env.success:
            print(f"Sucesso na alocação: {env.success}")
        else:
            print(f"Falha na alocação: {env.fail_reason}")
            # print(f"Servers used {env.servers_used}")
            print(f"ultimo server {env.valid_nodes[action]}")
            # print(f"Rede")
            # for node in env.G.nodes:
            #     print(f"{node} - {env.G.nodes[node]}")