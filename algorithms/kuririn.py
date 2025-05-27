import copy
import logging
import re
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from algorithms.environment import NetworkEnv
import os

from config import ROOT_PATH

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(ROOT_PATH + './logs/Kuririn.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')
N_STEPS=256
IS_TRAINING = True

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Isso desabilita o uso da GPU


## device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def colect_session(texto):
    match = re.search(r'_(\d+)$',texto)
    if match:
        sessao = match.group(1)
    else:
        return None
    return sessao

def remove_end_numbers(texto): 
    texto = re.sub(r'\d+$','',texto)
    return texto

class Kuririn:
    def __init__(self, model_name):
        
        self.model_name = model_name
        self.model_path = f'saved_models_rl/{self.model_name}_sfc_allocation'
        self.name = "kuririn"
        self.env = None
        self.graph = None 
        self.sfc = None
        self.route_info = self.node_info = {}
        self.latency = self.latency_request = None
        self.num_nodes = None
        self.servers_used = []
        self.single_source_minimum_latency_path = None
        self.fail_reason = None
        self.can_host_multiple_sfs = True
        self.is_backup = False
        self.valid_nodes = None
    
        # Cost weights
        self.cpu_factor = 3
        self.cache_factor = 3
        self.band_factor = 1
        self.latency_factor = 5

        self.boot_factor = 0        
        self.env = None


    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        self.src_substrate_node = None
        self.dst_substrate_node = None
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self,graph, shareable_sfs=[]):
        self.graph = graph
        self.num_nodes = len(self.graph.nodes())
        
        self.valid_nodes = [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'router' and node != 0]

        self.env = NetworkEnv(valid_nodes = self.valid_nodes,pesos={"cpu":self.cpu_factor,
                                                                "cache": self.cache_factor,
                                                                "band": self.band_factor,
                                                                "latency":self.latency_factor})
        self.model = self._load_or_create_model(self.env)
        return self.graph

    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        self.dst_substrate_node = sfc.dst_node
        return sfc

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def get_fail_reason(self):
        return self.fail_reason

    def handle_failure(self):
        self.route_info = False
        self.latency = None

    def check_solution(self):
        if not isinstance(self.latency, (int, float)) or self.latency < 0 or self.latency > self.sfc.get_latency_request() or not self.route_info:
            return False
        if len(list(self.route_info.keys()))!=6:
            return False
        
        prev_path_end = None
        for sf, path in self.route_info.items():
            if sf == 'dst':
                continue
            if prev_path_end is not None:
                if path[-1] != prev_path_end:
                    print(f"Inconsistência entre {prev_sf} e {sf}: {prev_path_end} != {path[0]}")
                    return False  # ou raise Exception se quiser abortar
            prev_path_end = path[0]
            prev_sf = sf
        return True

    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters

    # def start_algorithm(self, shareable_sfs={}, is_backup=False):
    #     self.is_backup = False
    #     if self.algorithm(shareable_sfs):
    #         # self._save_model()
    #         return True
    #     # self._save_model()
    #     return False

    def start_algorithm(self):#,is_backup):
        self.algorithm()
        is_success = self.check_solution()
        if is_success:
            try:
                logger.info("Finished algorithm, success")
                if IS_TRAINING:
                    self._save_model()
                return True  
            except:
                self.handle_failure() 
                return False
        else:
            self.handle_failure() 
            logger.info("End algorithm, failed")
            if IS_TRAINING:
                    self._save_model()
            return False

    def algorithm(self):
        self.servers_used = []
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        nodes_resource = self.set_nodes_resources()
        network_links = copy.deepcopy(self.graph._adj)
        G = self.create_network_graph(network_links)
        services, service_requirements = self.prepare_service_requirements(self.sfc.vnfs_dict)

        route_info, latency = self.find_best_allocation_for_sfc(
            G,service_requirements, nodes_resource,services, dst
        )
        # print(f"\nRazão Falha: {self.fail_reason}")
        return self.evaluate_result(latency, route_info)

    def create_network_graph(self, network_topology):
        G = nx.Graph()
        for node, edges in network_topology.items():
            for target, attr in edges.items():
                bw_free = attr['bandwidth_capacity'] - attr['bandwidth_used']
                G.add_edge(node, target, bandwidth=bw_free, weight=1)
        return G

    def set_nodes_resources(self):
        net_info = self.graph
        resources = {}
        for server in net_info.nodes:
            if self.graph._node[server]['type'] == "server":
                reuse = self.graph._node[server]['reuse']
            else:
                reuse = []
            resources[server] = {
                'cpu_capacity': self.graph.nodes[server]['cpu_capacity'],
                'cache_capacity': self.graph.nodes[server]['cache_capacity'],
                'cpu_used': self.graph.nodes[server]['cpu_used'],
                'cache_used': self.graph.nodes[server]['cache_used'],
                'cpu_free': self.graph.nodes[server]['cpu_capacity']-self.graph.nodes[server]['cpu_used'],
                'cache_free': self.graph.nodes[server]['cpu_capacity']-self.graph.nodes[server]['cache_used'],
                # 'position': net_info.nodes[server]['position'],
                'reuse': [(remove_end_numbers(service.id), colect_session(self.sfc.id)) for service in reuse] #TODO deve considerar a sessão também e não somente o nome
            }
        return resources

    def prepare_service_requirements(self, sfs_dict):
        service_requirements = {}
        services = [item['name'] for item in sfs_dict]
        for item in sfs_dict:
            name = item['name']
            service_requirements[name] = {
                'CPU': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw'],
                'latency': item['latency']
            }
        service_requirements['dst'] = {'CPU': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0, 'latency': 0}
        return list(reversed(services)), service_requirements

    def find_best_allocation_for_sfc(self, G,service_requirements, server_resources, services, dst):

        self.env.G, self.env.substrate_network, self.env.valid_nodes  = G, self.graph, self.valid_nodes
        self.env.set_server_resources(server_resources)
        self.env.services ,self.env.service_requirements =services, service_requirements
        self.env.latency_request,self.env.dst_node = self.latency_request, dst
        self.env.session_number=colect_session(self.sfc.id)   

        if not self.model:
            self._load_or_create_model(self.env)
        self.model.learn(total_timesteps=2048)
        state, _ = self.env.reset()
        self.env.is_training = False
        self.env.allocation_results['dst'] = {'allocated_server': dst, 'path': [], 'cost': 0}

        done = False
        self.fail_reason = None

        while not done:
            action, _ = self.model.predict(state, deterministic = True)
            state, _, done, _, _ = self.env.step(action)

            
        if not self.env.success:
            if IS_TRAINING:
                state, _ = self.env.reset()
                self.model.learn(total_timesteps=2048)
                state, _ = self.env.reset()
                self.env.is_training = False
                self.env.allocation_results['dst'] = {'allocated_server': dst, 'path': [], 'cost': 0}
                done = False
                while not done:
                    action, _ = self.model.predict(state, deterministic = True)
                    state, _, done, _, _ = self.env.step(action)             
            if not self.env.success:
                self.fail_reason = self.env.fail_reason
            return [], None

        print(f"Latencia usada: {self.env.latency_used}")
        route_info = {
            key: list(reversed(value['path']))
            for key, value in self.env.allocation_results.items()
        }

        path_to_src = nx.dijkstra_path(G, int(self.env.current_location), 0, weight='weight')
        self.env.close()
        total_latency = sum(len(p) - 1 for p in route_info.values() if p)
        route_info['src'] = list(reversed(path_to_src))

        return route_info, total_latency

    def evaluate_result(self, latency, route_info):
        if self.fail_reason == 'resource':
            self.route_info = False
            self.latency = None
            return False
        if self.fail_reason == 'latency':
            self.route_info = False
            self.latency = None
            return False
        self.latency = latency
        self.route_info = route_info
        return True

    def _load_or_create_model(self,env):
        if self.model_name == "ppo":
            if os.path.exists(self.model_path + ".zip"):
                model = PPO.load(self.model_path)
                self.model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, n_steps=256, ent_coef=0.3,
                                device ='cpu')
                self.model.policy.load_state_dict(model.policy.state_dict())
            else:
                self.model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, n_steps=256, ent_coef=0.3, device='cpu')

        if self.model_name == "dqn":
            if os.path.exists(self.model_path + ".zip"):
                model = DQN.load(self.model_path)
                self.model = DQN(
                    "MlpPolicy",               # Tipo de política, MlpPolicy para redes MLP
                    env,                        # Ambiente para o agente interagir
                    verbose=0,                  # Nível de verbosidade (0 para sem saída)
                    learning_rate=0.00003,      # Taxa de aprendizado
                    buffer_size=1_000_000,      # Tamanho do buffer de replay
                    learning_starts=100,        # Passos antes de começar o treinamento real
                    batch_size=64,              # Tamanho do lote para cada atualização de gradiente
                    tau=1.0,                    # Coeficiente da atualização suave (Polyak)
                    gamma=0.99,                 # Fator de desconto
                    train_freq=4,               # Atualizar o modelo a cada 4 passos
                    gradient_steps=1,           # Número de passos de gradiente por atualização
                    replay_buffer_class=None,   # Classe do buffer de replay (None para o padrão)
                    replay_buffer_kwargs=None,  # Argumentos para o buffer de replay
                    optimize_memory_usage=False,# Ativar o uso otimizado de memória no buffer
                    target_update_interval=10000, # Atualizar a rede alvo a cada 10.000 passos
                    exploration_fraction=0.1,    # Fração do treinamento durante a qual a exploração ocorre
                    exploration_initial_eps=1.0, # Epsilon inicial de exploração
                    exploration_final_eps=0.05,  # Epsilon final de exploração
                    max_grad_norm=10,            # Normalização máxima do gradiente
                    stats_window_size=100,       # Tamanho da janela de estatísticas para log
                    tensorboard_log=None,       # Local do log do TensorBoard (None para desabilitar)
                    policy_kwargs=None,          # Argumentos adicionais para a política (se necessário)
                    seed=None,                   # Semente para o gerador de números aleatórios
                    device='cpu',                # Dispositivo (cpu, cuda, ...). Aqui estamos utilizando a CPU
                    _init_setup_model=True       # Se a rede deve ser construída no momento da criação
                )

                self.model.policy.load_state_dict(model.policy.state_dict())
            else:
                self.model = DQN(
                                "MlpPolicy",               # Tipo de política, MlpPolicy para redes MLP
                                env,                        # Ambiente para o agente interagir
                                verbose=0,                  # Nível de verbosidade (0 para sem saída)
                                learning_rate=0.00003,      # Taxa de aprendizado
                                buffer_size=100_000_000,      # Tamanho do buffer de replay
                                learning_starts=256,        # Passos antes de começar o treinamento real
                                batch_size=64,              # Tamanho do lote para cada atualização de gradiente
                                tau=1.0,                    # Coeficiente da atualização suave (Polyak)
                                gamma=0.99,                 # Fator de desconto
                                train_freq=4,               # Atualizar o modelo a cada 4 passos
                                gradient_steps=1,           # Número de passos de gradiente por atualizaçã
                                target_update_interval=256, # Atualizar a rede alvo a cada 10.000 passos
                                exploration_fraction=0.1,    # Fração do treinamento durante a qual a exploração ocorre
                                exploration_initial_eps=1.0, # Epsilon inicial de exploração
                                exploration_final_eps=0.05,  # Epsilon final de exploração
                                max_grad_norm=10,            # Normalização máxima do gradiente
                                stats_window_size=100,       # Tamanho da janela de estatísticas para log
                                tensorboard_log=None,       # Local do log do TensorBoard (None para desabilitar)
                                policy_kwargs=None,          # Argumentos adicionais para a política (se necessário)
                                seed=None,                   # Semente para o gerador de números aleatórios
                                device='cpu',                # Dispositivo (cpu, cuda, ...). Aqui estamos utilizando a CPU
                                _init_setup_model=True       # Se a rede deve ser construída no momento da criação
                            )


    def _save_model(self):
        self.model.save(self.model_path)
