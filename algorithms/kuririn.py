import copy
import logging
import re
import networkx as nx
from stable_baselines3 import PPO, DQN
from algorithms.environment import NetworkEnv
import os
from config import ROOT_PATH
# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(ROOT_PATH, 'logs/Kuririn.log'))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Constants
N_STEPS = 256
IS_TRAINING = 0
VERBOSE = False
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Desabilita o uso da GPU

def collect_session(text):
    match = re.search(r'_(\d+)$', text)
    return match.group(1) if match else None


class Kuririn:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_path = f'saved_rl_models/{self.model_name}_sfc_allocation'
        self.name = "kuririn"
        self.env = None
        self.graph = None
        self.sfc = None
        self.route_info = self.node_info = {}
        self.latency = self.latency_request = None
        self.single_source_minimum_latency_path = None
        self.fail_reason = None
        self.can_host_multiple_sfs = True
        self.is_backup = False
        self.valid_nodes = None
        self.last_propose= None
        self.precomputed_paths = {}
    
        # Cost weights
        self.cpu_factor = 4
        self.cache_factor = 4
        self.band_factor = 1
        self.latency_factor = 4
        self.boot_factor = 0
        self.env = None

    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        self.src_substrate_node = None
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self, graph, shareable_sfs=[]):
        self.graph = graph


    def install_SFC(self, sfc):
        self.sfc = sfc
        self.latency_request = sfc.get_latency_request()
        self.dst_vnf = self.sfc.get_dst_vnf()
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
        if not isinstance(self.latency, (int, float)) or not (0 <= self.latency <= self.sfc.get_latency_request()) or not self.route_info:
            return False
        if len(self.route_info) != 6:
            return False
        prev_path_end = None
        for sf, path in self.route_info.items():
            if sf == 'dst': continue
           
            if prev_path_end and path[-1] != prev_path_end:
                print(f"Inconsistência entre {prev_sf} e {sf}: {prev_path_end} != {path[0]}")
                return False
            prev_path_end = path[0]
            prev_sf = sf
        return True

    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters

    def start_algorithm(self):
        self.valid_nodes = [node for node in self.graph.nodes() if self.graph.nodes[node]['type'] != 'router' and node != 0]
        self.algorithm()
        if self.check_solution():
            try:
                logger.info("Finished algorithm, success")
                if IS_TRAINING:
                    self._save_model()
                return True  
            except Exception:
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            logger.info(f"End algorithm, failed: {self.fail_reason}")
            if IS_TRAINING:
                self._save_model()
            return False

    def algorithm(self):
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())

        G = self.graph
        services, service_requirements = self.prepare_service_requirements(self.sfc.vnfs_dict)

        route_info, latency = self.find_best_allocation_for_sfc(G, service_requirements,services, dst)
        return self.evaluate_result(latency, route_info)

    def create_network_graph(self, network_topology):
        G = nx.Graph()
        for node, edges in network_topology.items():
            for target, attr in edges.items():
                bw_free = attr['bandwidth_capacity'] - attr['bandwidth_used']
                G.add_edge(node, target, bandwidth=bw_free, weight=1)
        return G

    def set_nodes_resources(self):
        resources = {}
        for server in self.graph.nodes:
            if self.graph.nodes[server]['type'] == "server":
                reuse = self.graph.nodes[server].get('reuse', [])
            else:
                reuse = []
            resources[server] = {
                'cpu_capacity': self.graph.nodes[server]['cpu_capacity'],
                'cache_capacity': self.graph.nodes[server]['cache_capacity'],
                'cpu_used': self.graph.nodes[server]['cpu_used'],
                'cache_used': self.graph.nodes[server]['cache_used'],
                'cpu_free': self.graph.nodes[server]['cpu_capacity'] - self.graph.nodes[server]['cpu_used'],
                'cache_free': self.graph.nodes[server]['cache_capacity'] - self.graph.nodes[server]['cache_used'],
                'reuse': [(service.id, self.sfc.id.split("_")[-1]) for service in reuse]
            }
        return resources

    def prepare_service_requirements(self, sfs_dict):
        service_requirements = {}
        services = [item['name'] for item in sfs_dict]
        for item in sfs_dict:
            name = item['name']
            service_requirements[name] = {
                'cpu': item['CPU'],
                'cache': item['cache'],
                'out_bw': item['out_bw'],
                'in_bw': item['in_bw'],
                'latency': item['latency']
            }
        service_requirements['dst'] = {'CPU': 0, 'cache': 0, 'out_bw': 0, 'in_bw': 0, 'latency': 0}
        return list(reversed(services)), service_requirements

    def find_best_allocation_for_sfc(self, G, service_requirements,services, dst):


        self.env = NetworkEnv(list_graph=[self.graph],list_sfc=[self.sfc],valid_nodes=self.valid_nodes,pesos={
            "cpu": self.cpu_factor,
            "cache": self.cache_factor,
            "band": self.band_factor,
            "latency": self.latency_factor
        })


        self._load_or_create_model(self.env)

        # log_callback = LogTrainingProgressCallback(log_interval=N_STEPS // 10)
        if IS_TRAINING:
            self.model.learn(total_timesteps=N_STEPS)
        state, _ = self.env.reset()
        self.env.is_training = False

        self.env.allocation_results['dst'] = {'allocated_server': dst, 'path': [], 'cost': 0}

        done = False
        while not done:
            action, _ = self.model.predict(state, deterministic=True)
            state, _, done, _, _ = self.env.step(action)

        if not self.env.success and IS_TRAINING:
            state, _ = self.env.reset()
            self.model.learn(total_timesteps=N_STEPS*8)
            # self.model.learn(total_timesteps=N_STEPS*8)
            state, _ = self.env.reset()
            self.env.is_training = False
            self.env.allocation_results['dst'] = {'allocated_server': dst, 'path': [], 'cost': 0}
            done = False
            while not done:
                action, _ = self.model.predict(state, deterministic=True)
                state, _, done, _, _ = self.env.step(action)
        
        if not self.env.success:
            if not VERBOSE:
                # print(f"Alocação falha sugerida SFC :{self.env.servers_used} - ultimo nó escolhido {self.env.server}")
                print(f"Causa Falha: {self.env.fail_reason}")
            self.fail_reason = self.env.fail_reason
            if self.fail_reason == 'latency':
                print(f"Alocação: [{self.env.servers_used}] || Custo latencia: {self.env.latency_used}")
            
            return [], None
        if not VERBOSE:
            print(f"Solução sfc {self.sfc.id}: {self.env.servers_used} || Latencia: {self.env.latency_used}")
            # for server_results in self.env.allocation_results:
            #     print("Servidor: ",self.env.allocation_results[server_results]["allocated_server"],\
            #           "Custo: ",self.env.allocation_results[server_results]['cost'])
        self.last_propose = self.env.servers_used
        route_info = {
            key: list(reversed(value['path']))
            for key, value in self.env.allocation_results.items()
        }

        # Armazene caminhos em um dicionário para evitar recalcular
        if (self.env.current_location, 0) not in self.precomputed_paths:
            self.precomputed_paths[(self.env.current_location, 0)] = nx.dijkstra_path(G, self.env.current_location, 0, weight='weight')

        path_to_src = self.precomputed_paths[(self.env.current_location, 0)]
        self.env.close()
        total_latency = sum(len(p) - 1 for p in route_info.values() if p)
        route_info['src'] = list(reversed(path_to_src))
        
        return route_info, total_latency

    def evaluate_result(self, latency, route_info):
        if self.fail_reason in ['resource', 'latency','bandwidth']:
            self.route_info = False
            self.latency = None
            return False
        self.latency = latency
        self.route_info = route_info
        return True

    def _load_or_create_model(self, env):
        if self.model_name == "ppo":
            self.model = PPO.load(self.model_path)
        elif self.model_name == "dqn":
            if os.path.exists(self.model_path + ".zip"):
                model = DQN.load(self.model_path)
                self.model = DQN("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, buffer_size=100_000_000, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=256, device='cpu')
                self.model.policy.load_state_dict(model.policy.state_dict())
            else:
                self.model = DQN("MlpPolicy", env, verbose=0, learning_rate=0.00003, batch_size=64, buffer_size=100_000_000, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=256, device='cpu')

    def _save_model(self):
        self.model.save(self.model_path)

