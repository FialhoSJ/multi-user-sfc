import contextlib
import logging
import os
import sys

import networkx as nx
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO

from muar_sfc.algorithms.environments.env_replic import SFC_AllocationEnv
from muar_sfc.config import ROOT_DIR
from muar_sfc.core.sfc import SFC

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(ROOT_DIR, "logs/REPLIC.log"))
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Constants
IS_TRAINING = 0
VERBOSE = False
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Desabilita o uso da GPU


@contextlib.contextmanager
def suppress_output():
    """Silencia o stdout (prints) dentro deste bloco."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class REPLIC:
    ### MODIFICADO ###
    # O construtor agora carrega o modelo imediatamente.
    def __init__(self, model_name):
        # --- Atributos ---
        self.model_name = model_name.upper()  # Ex: "PPO", "MASKABLEPPO", ou "DQN"

        # --- MODIFICADO: O caminho agora é mais explícito ---
        # Garante que PPO carregue "PPO_allocation_model.zip"
        # e MaskablePPO carregue "MaskablePPO_allocation_model.zip"
        self.model_path = f"rl_saved_models/{self.model_name}_REPLIC_allocation_model.zip"
        self.name = f"REPLIC{model_name}"

        # --- ADICIONADO: Carregamento do modelo na inicialização ---
        self.model = self._load_model()

        # Demais atributos da sua classe (mantidos do original)
        self.graph = None
        self.sfc = None
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.latency_request = None
        self.single_source_minimum_latency_path = None
        self.fail_reason = None
        self.can_host_multiple_sfs = True
        self.is_backup = False
        self.valid_nodes = None
        self.last_propose = None
        self.precomputed_paths = {}
        # self.env foi removido pois não pertence mais à classe.

        # Pesos de custo (mantidos do original)
        self.cpu_factor = 5
        self.cache_factor = 5
        self.band_factor = 2
        self.latency_factor = 2
        self.boot_factor = 0

    ### ADICIONADO ###
    # Método privado para carregar o modelo, chamado apenas uma vez.
    def _load_model(self):
        """Carrega o modelo de RL do arquivo, sem precisar de um ambiente."""
        if not os.path.exists(self.model_path):
            logger.error(f"Arquivo do modelo não encontrado: {self.model_path}")
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")

        logger.info(f"Carregando modelo de: {self.model_path}")

        # Carrega MaskablePPO se o nome for "MASKABLEPPO"
        if self.model_name == "MASKABLEPPO":
            return MaskablePPO.load(self.model_path, device="cpu")
        # Carrega PPO padrão se o nome for "PPO"
        elif self.model_name == "PPO":
            return PPO.load(self.model_path, device="cpu")
        elif self.model_name == "DQN":
            return DQN.load(self.model_path, device="cpu")
        else:
            raise ValueError(
                f"Nome do modelo inválido: '{self.model_name}'. Use 'PPO', 'MaskablePPO' ou 'DQN'."
            )

    # O método clear_all foi mantido como no original.
    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        self.src_substrate_node = None
        self.single_source_minimum_latency_path = None

    # O método install_substrate_network foi mantido como no original.
    def install_substrate_network(self, graph, shareable_sfs=[]):
        self.graph = graph
        self.valid_nodes = [
            node for node in self.graph.nodes() if self.graph.nodes[node]["type"] != "router"
        ]

        if self.precomputed_paths is None:
            self.precomputed_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight="weight"))

    # O método install_SFC foi mantido como no original.
    def install_SFC(self, sfc: SFC):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        is_backup = True if "backup" in sfc.id else False
        self.is_backup = is_backup

        self.latency_request = sfc.get_latency_request()
        self.min_latency = 0

        service_requirements = {}
        services = []
        sfs_dict = sfc.vnfs_dict

        for item in sfs_dict:
            nome = item["name"]
            services.append(nome)
            service_requirements[nome] = {
                "CPU": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        if not is_backup:
            services.append("dst")
            service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

        self.service_requirements = service_requirements
        self.services = services

        return self.sfc

    # Todos os métodos getters e de utilidade foram mantidos como no original.
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
        # Validação de robustez padrão
        if not isinstance(self.latency, (int, float)) or not self.route_info:
            return False

        # CORREÇÃO:
        # Como corrigimos o env_REPLIC.py, agora todo backup terá rota completa.
        # Podemos exigir consistência mínima de conexões (edges) em vez de tamanho fixo arbitrário.

        # Se quiser manter a verificação de tamanho por segurança:
        min_hops = 2  # Pelo menos uma conexão (Origem -> Destino)
        if len(self.route_info) < min_hops:
            return False

        prev_path_end = None
        for sf, path in self.route_info.items():
            prev_sf = None
            if sf == "dst":
                continue

            if prev_path_end and path[-1] != prev_path_end:
                print(f"Inconsistência entre {prev_sf} e {sf}: {prev_path_end} != {path[0]}")
                return False
            prev_path_end = path[0]
            prev_sf = sf
        return True

    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters

    ### MODIFICADO ###
    # O método principal agora RECEBE a instância do ambiente.
    def start_algorithm(self, env: SFC_AllocationEnv, args=None):
        if not self.valid_nodes or not self.sfc or not self.graph:
            self.fail_reason = "Erro: Rede ou SFC não foram instalados..."
            logger.error(self.fail_reason)
            self.handle_failure()
            return False

        env.is_training = False
        self.fail_reason = None
        env.valid_nodes = self.valid_nodes
        env._set_list_graph_sfcs([self.graph], [self.sfc])

        # --- NOVA LÓGICA: Injeção de Configuração de Confiabilidade ---
        if args:
            # Monta o dicionário com base nos argumentos do main.py
            reliability_config = {
                "tiers": {
                    "default": getattr(args, "rel_normal", 0.99),
                    "a": getattr(args, "rel_low", 0.95),
                    "b": getattr(args, "rel_normal", 0.98),
                    "c": getattr(args, "rel_high", 0.999),
                },
                "stress": {
                    "default": getattr(args, "stress_normal", 0.04),
                    "a": getattr(args, "stress_low", 0.15),
                    "b": getattr(args, "stress_normal", 0.08),
                    "c": getattr(args, "stress_high", 0.02),
                },
            }
            # Atualiza o env
            env.reliability_config = reliability_config
        # -------------------------------------------------------------

        with suppress_output():
            self.model.set_env(env)

        self.algorithm(env)

        if self.check_solution():
            try:
                if "backup" not in self.sfc.id:
                    logger.info("Finished algorithm, success")
                return True
            except Exception:
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            if "backup" not in self.sfc.id:
                logger.info(f"End algorithm, failed: {self.fail_reason}")
            return False

    ### MODIFICADO ###
    # O método algorithm agora recebe e repassa o ambiente.
    def algorithm(self, env: SFC_AllocationEnv):
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())

        # Passa o ambiente para o método que executa o loop de predição.
        route_info, latency = self.find_best_allocation_for_sfc(env, dst)

        return self.evaluate_result(latency, route_info)

    ### MODIFICADO ###
    # Este método foi simplificado para apenas executar o loop de predição.
    def find_best_allocation_for_sfc(self, env: SFC_AllocationEnv, dst):
        # A criação e reset do ambiente
        obs, _ = env.reset()
        env.is_training = False

        # Inicializa o resultado do destino
        env.allocation_results["dst"] = {"allocated_server": dst, "path": [], "cost": 0}

        done = False
        while not done:
            if self.model_name == "MASKABLEPPO":
                action_masks = env.action_masks()
                action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=False)
            else:
                action, _ = self.model.predict(obs, deterministic=False)

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if not env.success:
            if not VERBOSE:
                # print(f"Causa Falha: {env.fail_reason}")
                pass
            self.fail_reason = env.fail_reason
            return {}, None  # Retorna dict vazio em vez de lista vazia

        # --- CONSTRUÇÃO DO ROUTE INFO ---
        # O env.allocation_results geralmente vem no formato:
        # {'vnf_id': {'path': [nó_atual, ..., proximo_no], ...}}

        route_info = {}

        # 1. Copia e inverte os caminhos (se o env retornar invertido)
        # Assumindo que env.allocation_results['path'] é [destino, ..., origem]
        # e queremos [origem, ..., destino]
        for key, value in env.allocation_results.items():
            if value["path"]:
                route_info[key] = list(reversed(value["path"]))
            else:
                route_info[key] = []

        total_latency = env.latency_used

        # --- TRATAMENTO DIFERENCIADO: SFC NORMAL vs BACKUP ---
        if "backup" not in self.sfc.id:
            # Lógica para SFC Normal (Conecta ao SRC Global/Cloud)
            # Pega o primeiro nó da primeira VNF processada (que é a última na ordem reversa do dict)
            if route_info:
                first_vnf_key = list(route_info.keys())[-1]
                if route_info[first_vnf_key]:
                    src_node_network = route_info[first_vnf_key][0]
                else:
                    # Fallback se path vazio
                    src_node_network = env.allocation_results[first_vnf_key]["allocated_server"]

                # Calcula caminho do Cloud (0) até a primeira VNF
                try:
                    path_to_src = list(
                        nx.dijkstra_path(self.graph, 0, src_node_network, weight="weight")
                    )
                    # Remove o último elemento para não duplicar com o início da próxima rota
                    # path_to_src = path_to_src[:-1]
                    route_info["src"] = path_to_src
                    total_latency += len(path_to_src) - 1  # Simplificação de latência por hops
                except nx.NetworkXNoPath:
                    self.fail_reason = "Sem rota para Cloud"
                    return {}, None
        else:
            # --- LÓGICA PARA BACKUP (MINI-SFC) ---
            # Não calculamos rota para o nó 0.
            # A Mini-SFC já é autocontida (src_virt -> vnf_b -> dst_virt).
            # O env_REPLIC já deve ter garantido a rota entre src_virt e vnf_b.

            # Apenas garantimos que não sobrou lixo e o formato é dict.
            # O código anterior que sobrescrevia 'route_info' com list() foi removido.
            pass

        if "backup" not in self.sfc.id:
            print(f"{self.sfc.id} - Alocação: {env.servers_used}")
        return route_info, total_latency

    # O método evaluate_result foi mantido como no original.
    def evaluate_result(self, latency, route_info):
        if self.fail_reason in ["resource", "latency", "bandwidth"]:
            self.route_info = False
            self.latency = None
            return False
        self.latency = latency
        self.route_info = route_info
        return True

    ### REMOVIDO ###
    # Os métodos abaixo foram removidos pois a classe não gerencia mais
    # a criação do ambiente ou o carregamento do modelo em tempo de execução.
    # def _load_or_create_model(self, env):
    # def load_model(self, env):
    # def reset_environment(self, list_graph, list_sfc):
    # def _initialize_environment_and_model(self):
