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

# --- REFATORAÇÃO: Logging orientado a objetos multiplataforma ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Garante ativamente que o diretório de logs exista na raiz estrutural antes de alocar
log_dir = ROOT_DIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "REPLIC.log"

file_handler = logging.FileHandler(log_file)
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
    def __init__(self, model_name):
        self.model_name = model_name.upper()

        # --- REFATORAÇÃO: O descarte do os.path ---
        # A biblioteca pathlib atua de modo multiplataforma com o operador matriz '/'
        self.model_path = ROOT_DIR / "rl_saved_models" / f"{self.model_name}_REPLIC_allocation_model.zip"
        self.name = f"REPLIC{model_name}"

        self.model = self._load_model()

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

        self.cpu_factor = 5
        self.cache_factor = 5
        self.band_factor = 2
        self.latency_factor = 2
        self.boot_factor = 0

    def _load_model(self):
        """Carrega o modelo de RL do arquivo explorando as propriedades nativas do pathlib."""

        # Validação limpa e atômica nativa do Path
        if not self.model_path.exists():
            logger.error(f"Arquivo do modelo não encontrado: {self.model_path}")
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {self.model_path}")

        logger.info(f"Carregando modelo de: {self.model_path}")

        # Ferramentas C externas (Stable Baselines) ainda requerem strings literais no loader
        path_str = str(self.model_path)

        if self.model_name == "MASKABLEPPO":
            return MaskablePPO.load(path_str, device="cpu")
        elif self.model_name == "PPO":
            return PPO.load(path_str, device="cpu")
        elif self.model_name == "DQN":
            return DQN.load(path_str, device="cpu")
        else:
            raise ValueError(
                f"Nome do modelo inválido: '{self.model_name}'. Use 'PPO', 'MaskablePPO' ou 'DQN'."
            )

    def clear_all(self):
        self.substrate_network = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.latency = None
        self.src_substrate_node = None
        self.single_source_minimum_latency_path = None

    def install_substrate_network(self, graph, shareable_sfs=None):
        if shareable_sfs is None:
            shareable_sfs = []
        self.graph = graph
        self.valid_nodes = [
            node for node in self.graph.nodes() if self.graph.nodes[node]["type"] != "router"
        ]

        if getattr(self, "precomputed_paths", None) is None or not self.precomputed_paths:
            self.precomputed_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight="weight"))

    def install_SFC(self, sfc: SFC):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        is_backup = "backup" in sfc.id
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
        if not isinstance(self.latency, (int, float)) or not self.route_info:
            return False

        min_hops = 2
        if len(self.route_info) < min_hops:
            return False

        prev_path_end = None
        for sf, path in self.route_info.items():
            if sf == "dst":
                continue

            if prev_path_end and path[-1] != prev_path_end:
                logger.warning(f"Inconsistência entre {prev_sf} e {sf}: {prev_path_end} != {path[0]}")
                return False
            prev_path_end = path[0]
            prev_sf = sf
        return True

    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters

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

        if args:
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
            env.reliability_config = reliability_config

        with suppress_output():
            self.model.set_env(env)

        self.algorithm(env)

        if self.check_solution():
            # REFATORAÇÃO: Morte ao pass silencioso
            try:
                if "backup" not in self.sfc.id:
                    logger.info("Finished algorithm, success")
                return True
            except Exception as e:
                logger.error(f"Erro ao computar sucesso da alocação de rede: {e}")
                self.handle_failure()
                return False
        else:
            self.handle_failure()
            if "backup" not in self.sfc.id:
                logger.info(f"End algorithm, failed: {self.fail_reason}")
            return False

    def algorithm(self, env: SFC_AllocationEnv):
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        route_info, latency = self.find_best_allocation_for_sfc(env, dst)
        return self.evaluate_result(latency, route_info)

    def find_best_allocation_for_sfc(self, env: SFC_AllocationEnv, dst):
        obs, _ = env.reset()
        env.is_training = False

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
            self.fail_reason = env.fail_reason
            return {}, None

        route_info = {}

        for key, value in env.allocation_results.items():
            if value["path"]:
                route_info[key] = list(reversed(value["path"]))
            else:
                route_info[key] = []

        total_latency = env.latency_used

        if "backup" not in self.sfc.id:
            if route_info:
                first_vnf_key = list(route_info.keys())[-1]
                if route_info[first_vnf_key]:
                    src_node_network = route_info[first_vnf_key][0]
                else:
                    src_node_network = env.allocation_results[first_vnf_key]["allocated_server"]

            try:
                path_to_src = list(
                    nx.dijkstra_path(self.graph, 0, src_node_network, weight="weight")
                )
                route_info["src"] = path_to_src
                total_latency += len(path_to_src) - 1
            except nx.NetworkXNoPath:
                self.fail_reason = "Sem rota para Cloud"
                return {}, None

        return route_info, total_latency

    def evaluate_result(self, latency, route_info):
        if self.fail_reason in ["resource", "latency", "bandwidth"]:
            self.route_info = False
            self.latency = None
            return False
        self.latency = latency
        self.route_info = route_info
        return True
