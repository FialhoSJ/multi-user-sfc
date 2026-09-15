"""Algoritmo de inferência do agente híbrido (Pham + Wu + Zhao).

Segue a mesma interface dos demais algoritmos RL do projeto (REPLIC/DARSPPO):
``install_substrate_network``, ``install_SFC``, ``start_algorithm(env)``,
``get_route_info``, ``get_latency``, ``handle_failure`` e ``check_solution``.

Se o modelo treinado (``rl_saved_models/HYBRID_allocation_model.pt``) não
existir, opera em modo **fallback heurístico** (α = 1, nó de menor custo
local), permitindo ``--alg hybrid`` rodar mesmo sem treino.
"""

import logging
from pathlib import Path

import networkx as nx

from muar_sfc.algorithms.environments.env_replic import build_valid_nodes
from muar_sfc.algorithms.hybrid.hybrid_env import SFC_AllocationEnv_Hybrid
from muar_sfc.algorithms.hybrid.ppo import HybridPPO
from muar_sfc.algorithms.networkUtils import get_available_shortest_path_fast
from muar_sfc.config import ROOT_DIR
from muar_sfc.core.sfc import SFC

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = Path(ROOT_DIR) / "logs" / "HYBRID.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class HybridSFC:
    """Agente de alocação de SFCs com custos híbridos e ação nó + α."""

    def __init__(self, model_name: str = "HYBRID"):
        self.model_name = str(model_name).upper()
        self.name = "HybridSFC"
        self.model_path = (
            Path(ROOT_DIR) / "rl_saved_models" / f"{self.model_name}_allocation_model.pt"
        )
        self.model = self._load_model()

        self.graph = None
        self.sfc = None
        self.route_info = {}
        self.node_info = {}
        self.alphas: dict[str, float] = {}
        self.latency = None
        self.latency_request = None
        self.fail_reason = None
        self.can_host_multiple_sfs = True
        self.is_backup = False
        self.valid_nodes = None
        self.precomputed_paths = {}
        self.prev_active_nodes: set = set()

        self.cpu_factor = 5
        self.cache_factor = 5
        self.band_factor = 2
        self.latency_factor = 2
        self.boot_factor = 0

    def _load_model(self):
        """Carrega o modelo híbrido; retorna ``None`` para o fallback heurístico."""
        if not self.model_path.exists():
            logger.warning(
                f"Modelo híbrido não encontrado em {self.model_path}. "
                "Usando fallback heurístico (α=1)."
            )
            return None
        logger.info(f"Carregando modelo híbrido de: {self.model_path}")
        return HybridPPO.load(str(self.model_path), device="cpu")

    def clear_all(self):
        self.graph = None
        self.sfc = None
        self.node_info = {}
        self.route_info = {}
        self.alphas = {}
        self.latency = None

    def install_substrate_network(self, graph, shareable_sfs=None):
        self.graph = graph
        self.valid_nodes = build_valid_nodes(self.graph)
        for node_id in self.prev_active_nodes:
            if node_id in graph:
                graph.nodes[node_id]["is_active"] = True
        if not self.precomputed_paths:
            self.precomputed_paths = dict(nx.all_pairs_dijkstra_path(self.graph, weight="weight"))

    def install_SFC(self, sfc: SFC):
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.is_backup = "backup" in sfc.id
        self.latency_request = sfc.get_latency_request()

        self.services = []
        self.service_requirements = {}
        for item in sfc.vnfs_dict:
            nome = item["name"]
            self.services.append(nome)
            self.service_requirements[nome] = {
                "CPU": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        if not self.is_backup:
            self.services.append("dst")
            self.service_requirements["dst"] = {"CPU": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

        return self.sfc

    def get_latency(self):
        return self.latency

    def get_route_info(self):
        return self.route_info

    def get_alphas(self):
        return self.alphas

    def get_fail_reason(self):
        return self.fail_reason

    def handle_failure(self):
        self.route_info = {}
        self.alphas = {}
        self.latency = None

    def check_solution(self):
        if self.latency is None or not self.route_info:
            return False
        if len(self.route_info) < 2:
            return False

        next_hop_start = None
        for sf, path in self.route_info.items():
            if sf in ("src", "dst"):
                continue
            if next_hop_start is not None and path and path[-1] != next_hop_start:
                return False
            if path:
                next_hop_start = path[0]
        return True

    def set_costs(self, costs_parameters):
        self.cpu_factor, self.cache_factor, self.band_factor = costs_parameters

    def start_algorithm(self, env: SFC_AllocationEnv_Hybrid, args=None):
        if not self.valid_nodes or not self.sfc or not self.graph:
            self.fail_reason = "Erro: Rede ou SFC não foram instalados."
            logger.error(self.fail_reason)
            self.handle_failure()
            return False

        env.is_training = False
        self.fail_reason = None
        env.valid_nodes = self.valid_nodes
        env.set_prev_active_nodes(self.prev_active_nodes)
        # Inferência: usa o shadow graph compartilhado SEM copiar, para que as
        # alocações do lote acumulem (o deploy físico replica exatamente as mesmas).
        env._set_list_graph_sfcs([self.graph], [self.sfc], copy=False)

        self.algorithm(env)

        if self.check_solution():
            if not self.is_backup:
                logger.info("Finished algorithm, success")
            return True

        self.handle_failure()
        if not self.is_backup:
            logger.info(f"End algorithm, failed: {self.fail_reason}")
        return False

    def algorithm(self, env: SFC_AllocationEnv_Hybrid):
        dst = self.sfc.get_substrate_node(self.sfc.get_dst_vnf())
        route_info, latency = self.find_best_allocation_for_sfc(env, dst)
        return self.evaluate_result(latency, route_info)

    def find_best_allocation_for_sfc(self, env: SFC_AllocationEnv_Hybrid, dst):
        obs, _ = env.reset()
        env.is_training = False
        env.allocation_results["dst"] = {"allocated_server": dst, "path": [], "cost": 0}

        done = False
        while not done:
            if self.model is not None:
                masks = env.action_masks()
                node_idx, alpha = self.model.predict(obs, masks)
            else:
                node_idx, alpha = self._heuristic_action(env)

            self.alphas[env.current_vnf.id] = float(alpha)
            obs, _, terminated, truncated, _ = env.step((node_idx, alpha))
            done = terminated or truncated

        if not env.success:
            self.fail_reason = env.fail_reason
            return {}, None

        route_info = {
            key: list(reversed(value["path"])) if value["path"] else []
            for key, value in env.allocation_results.items()
        }
        total_latency = env.latency_used

        if not self.is_backup and route_info:
            try:
                first_vnf_key = next(reversed(route_info))
                src_node_network = route_info[first_vnf_key][0]
                path_to_src = list(
                    nx.dijkstra_path(self.graph, 0, src_node_network, weight="weight")
                )
                route_info["src"] = path_to_src
                total_latency += len(path_to_src) - 1
            except (nx.NetworkXNoPath, IndexError, StopIteration):
                self.fail_reason = "Sem rota para Cloud"
                return {}, None

        self.prev_active_nodes = set(env.active_nodes)
        return route_info, total_latency

    def _heuristic_action(self, env: SFC_AllocationEnv_Hybrid):
        """Fallback determinístico: menor custo (banda + latência) com α = 1."""
        vnf = env.current_vnf
        band_req = env.service_requirements[vnf.id]["out_bw"]
        current_location = env.current_location
        num_nodes = len(env.valid_nodes)

        best_idx = num_nodes - 1
        best_cost = float("inf")
        for i, node_id in enumerate(env.valid_nodes):
            target = env.current_sfc.dst_node if i == num_nodes - 1 else node_id
            path = get_available_shortest_path_fast(env.graph, current_location, target, band_req)
            if not path:
                continue
            c_b = env.cost_evaluator.compute_bandwidth_cost(path, band_req)
            lat = env.cost_evaluator.compute_latency(env.graph, path, vnf, 1.0, target)
            cost = c_b + lat * self.latency_factor
            if cost < best_cost:
                best_cost, best_idx = cost, i
        return best_idx, 1.0

    def evaluate_result(self, latency, route_info):
        if self.fail_reason in ("resource", "latency", "bandwidth"):
            self.handle_failure()
            return False
        self.latency = latency
        self.route_info = route_info
        return True
