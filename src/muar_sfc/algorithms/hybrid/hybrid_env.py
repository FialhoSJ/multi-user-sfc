"""Ambiente Gym híbrido para alocação de SFCs.

Une as três dimensões de custo (banda/saltos de Wu, energia/desgaste de Pham
e latência tripartida de Zhao) e expõe uma ação híbrida:

- **discreta**: seleção do nó de hospedagem (servidores + sentinela ``dst``);
- **contínua**: fração α ∈ (0, 1) de CPU alocada para a VNF atual.

O desgaste (wear-and-tear) compara o conjunto de nós ativos da alocação
corrente com ``prev_active_nodes`` (estado persistido entre episódios/SFCs).
"""

from __future__ import annotations

import copy as _copy
from typing import Any

import gymnasium
import networkx as nx
import numpy as np
from gymnasium import spaces
from networkx import Graph

from muar_sfc.algorithms.hybrid.cost import HybridSFCCostEvaluator, compute_node_powers
from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
    get_available_shortest_path_fast,
)
from muar_sfc.core.infrastructure.enums import vnf_is_shareable
from muar_sfc.core.sfc import SFC, VNF
from muar_sfc.utils.network_utils import (
    calcular_percentual_banda_total,
    calcular_percentual_cache_total,
    get_graph_processing_utilization_simplified,
)

EPSILON = 1e-5
MAX_VNFS = 8
NODE_FEAT_DIM = 8
VNF_FEAT_DIM = 5


class SFC_AllocationEnv_Hybrid(gymnasium.Env):
    """Ambiente com ação híbrida (nó discreto + fração α contínua)."""

    node_feat_dim = NODE_FEAT_DIM
    vnf_feat_dim = VNF_FEAT_DIM
    max_vnfs = MAX_VNFS

    def __init__(
        self,
        valid_nodes: list[int | str],
        list_graph: list[Graph],
        list_sfc: list[SFC],
        pesos_fatores: dict[str, float] | None = None,
        reward_config: dict[str, float] | None = None,
        reliability_config: dict[str, Any] | None = None,
        cost_config: dict[str, float] | None = None,
        is_training: bool = True,
        prev_active_nodes: set | None = None,
    ):
        super().__init__()

        if len(list_graph) != len(list_sfc):
            raise ValueError("A lista de grafos deve ter o mesmo tamanho da lista de SFCs.")

        self.valid_nodes = valid_nodes
        self.is_training = is_training
        self.prev_active_nodes = set(prev_active_nodes) if prev_active_nodes else set()
        self.active_nodes: set = set()

        self.pesos_fatores = pesos_fatores or {
            "cpu": 1,
            "cache": 1,
            "band": 3,
            "rel": 6,
            "lat": 3,
            "mobile": 0.0,
        }
        self.reward_config = reward_config or {
            "success_bonus": 3.0,
            "failure_penalty": -3.0,
            "congestion_weight": 1.0,
            "reliability_weight": 3.0,
            "bw_pressure_weight": 1.0,
            "hop_penalty_weight": 0.5,
        }
        self.reliability_config = reliability_config or {
            "tiers": {"default": 0.98, "a": 0.95, "b": 0.98, "c": 0.999},
            "stress": {"default": 0.08, "a": 0.15, "b": 0.08, "c": 0.02},
        }
        self.cost_evaluator = HybridSFCCostEvaluator(**(cost_config or {}))
        self.latency_target_ms = getattr(self.cost_evaluator, "latency_target_ms", 15.0)

        self.list_graph: list[Graph] = []
        self.list_sfc: list[SFC] = []
        self._set_list_graph_sfcs(list_graph, list_sfc)

        self.initial_resource_snapshot = self._initialize_snapshots(self.list_graph)

        self.graph: Graph | None = None
        self.current_sfc: SFC | None = None
        self.current_vnf: VNF | None = None
        self.current_location: int | str | None = None
        self.features = None
        self.forbidden_nodes: set = set()

        self.ratio_cpu_used = 0
        self.ratio_cache_used = 0
        self.ratio_banda_used = 0
        self.latency_used = 0
        self.last_metrics: dict[str, float] = {}

        self.servers_used: list = []
        self.allocation_results: dict = {}
        self.success = False
        self.fail_reason = None

        self._pair_lengths_cache: dict[int, dict] = {}
        self._powers_cache_key = None
        self._powers_cache: dict = {}
        self._episode_snapshot: dict | None = None
        self._eval_index = 0

        num_nodes = len(valid_nodes)
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(num_nodes),
                spaces.Box(low=0.01, high=1.0, shape=(1,), dtype=np.float32),
            )
        )
        self.observation_space = spaces.Dict(
            {
                "node_feats": spaces.Box(
                    low=-1.0, high=10.0, shape=(num_nodes, NODE_FEAT_DIM), dtype=np.float32
                ),
                "adj_matrix": spaces.Box(
                    low=0.0, high=float(num_nodes), shape=(num_nodes, num_nodes), dtype=np.float32
                ),
                "sfc_seq": spaces.Box(
                    low=-1.0, high=2.0, shape=(MAX_VNFS, VNF_FEAT_DIM), dtype=np.float32
                ),
                "sfc_seq_len": spaces.Box(
                    low=0.0, high=float(MAX_VNFS), shape=(1,), dtype=np.float32
                ),
            }
        )

    # =====================================================================
    # Interface principal do Gymnasium
    # =====================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.latency_used = 0
        idx = 0

        if self.is_training:
            idx = int(self.np_random.integers(len(self.list_graph)))
        else:
            idx = self._eval_index % len(self.list_graph)
            self._eval_index += 1

        self.graph = self.list_graph[idx]
        self._restore_graph_resources(idx)

        self.set_current_sfc(self.list_sfc[idx])

        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        self.active_nodes = set()
        self.servers_used = []
        self._mark_active_nodes()
        self._episode_snapshot = self._snapshot_graph_state(self.graph)

        vnf = self.current_vnf
        bw_req = self.service_requirements[vnf.id]["out_bw"]
        current_node = self.current_location

        self.ratio_cpu_used = get_graph_processing_utilization_simplified(self.graph)
        self.ratio_cache_used = calcular_percentual_cache_total(self.graph)
        self.ratio_banda_used = calcular_percentual_banda_total(self.graph)

        self.features = self._get_nodes_features(vnf, bw_req, current_node)
        return self._get_obs(), {}

    def step(self, action):
        node_idx, alpha = action
        alpha = float(np.clip(alpha, 0.01, 1.0))

        if not isinstance(node_idx, (int, np.integer)) or not (
            0 <= int(node_idx) < len(self.valid_nodes)
        ):
            return self._fail_step("invalid_action")
        node_idx = int(node_idx)

        if node_idx == len(self.valid_nodes) - 1:
            chosen_server = self.current_sfc.dst_node
        else:
            chosen_server = self.valid_nodes[node_idx]

        if chosen_server not in self.graph.nodes:
            return self._fail_step("invalid_node")
        if not self.graph.nodes[chosen_server].get("is_active", True):
            return self._fail_step("inactive_node")

        vnf = self.current_vnf
        band_req = self.service_requirements[vnf.id]["out_bw"]
        current_location = self.current_location

        path = get_available_shortest_path_fast(
            self.graph, current_location, chosen_server, band_req
        )
        self.ratio_cpu_used = get_graph_processing_utilization_simplified(self.graph)
        self.ratio_cache_used = calcular_percentual_cache_total(self.graph)
        self.ratio_banda_used = calcular_percentual_banda_total(self.graph)

        if not self.allocate_resources_on_node(chosen_server, vnf, alpha):
            return self._fail_step("resource")
        if not path or not self.allocate_bandwidth_along_path(path, band_req):
            return self._fail_step("bandwidth")

        # Balanceamento de carga (estilo VEGETA): penaliza concentrar VNFs em nós
        # já saturados — a utilização pós-alocação do nó escolhido entra na recompensa.
        chosen_node_data = self.graph.nodes[chosen_server]
        cpu_cap = chosen_node_data.get("cpu_capacity", 1.0) or 1.0
        congestion = chosen_node_data.get("cpu_used", 0.0) / cpu_cap
        congestion_penalty = self.reward_config.get("congestion_weight", 1.0) * congestion

        # Confiabilidade dinâmica do nó escolhido (Pham): nós menos carregados
        # e de nível superior têm maior confiabilidade.
        rel_reward = self.reward_config.get("reliability_weight", 1.5) * (
            self.get_dynamic_reliability(chosen_node_data)
        )

        # Pressão de banda por link (Wu/VEGETA): penaliza caminhos longos e
        # enlaces escassos — Σ bw_req/(bw_req + banda_livre) por salto.
        bw_pressure = 0.0
        n_hops = 0
        for u, v in zip(path[:-1], path[1:], strict=False):
            n_hops += 1
            edge = self.graph.edges[u, v]
            avail = edge.get("bandwidth_capacity", 0) - edge.get("bandwidth_used", 0)
            if avail > 0:
                bw_pressure += band_req / max(band_req + avail, 1e-6)
        bw_penalty = self.reward_config.get("bw_pressure_weight", 1.0) * bw_pressure
        hop_penalty = self.reward_config.get("hop_penalty_weight", 0.5) * n_hops

        self.servers_used.append(chosen_server)
        self.active_nodes.add(chosen_server)

        c_b = self.cost_evaluator.compute_bandwidth_cost(path, band_req)
        c_op = self.cost_evaluator.compute_operational_cost(
            self.active_nodes, self.prev_active_nodes, self._get_node_powers()
        )
        lat = self.cost_evaluator.compute_latency(self.graph, path, vnf, alpha, chosen_server)
        sla_latency = self.current_sfc.get_latency_request() or self.cost_evaluator.sla_latency_ms
        reward, metrics = self.cost_evaluator.calculate_reward(c_b, c_op, lat, sla_latency)
        reward -= congestion_penalty
        reward += rel_reward
        reward -= bw_penalty
        reward -= hop_penalty
        metrics["congestion_penalty"] = congestion_penalty
        metrics["reliability_reward"] = rel_reward
        metrics["bw_penalty"] = bw_penalty
        metrics["hop_penalty"] = hop_penalty
        self.last_metrics = metrics

        self.current_location = chosen_server
        if not self.is_training:
            self.allocation_results[self.current_vnf.id] = {
                "allocated_server": chosen_server,
                "path": path,
                "cost": c_b + c_op + lat,
            }

        self.latency_used += lat

        done = False
        if self.current_vnf == self.reverse_vnf_list[-1]:
            done = True
            self.success = True
            reward += self.reward_config["success_bonus"]
            self.prev_active_nodes = set(self.active_nodes)
            self._mark_active_nodes()
            self.current_vnf = None
        else:
            idx = self.reverse_vnf_list.index(self.current_vnf)
            self.current_vnf = self.reverse_vnf_list[idx + 1]

        bw_required = (
            self.service_requirements[self.current_vnf.id]["out_bw"] if self.current_vnf else 0
        )
        self.features = self._get_nodes_features(
            self.current_vnf, bw_required, self.current_location
        )
        obs = self._get_obs()

        return obs, reward, done, False, {}

    def action_masks(self) -> np.ndarray:
        if self.current_vnf is None:
            return np.zeros(len(self.valid_nodes), dtype=np.int8)

        vnf = self.current_vnf
        bw_req = self.service_requirements[vnf.id]["out_bw"]
        self.features = self._get_nodes_features(vnf, bw_req, self.current_location)

        mask = []
        for i, node_id in enumerate(self.valid_nodes):
            actual_node = self.current_sfc.dst_node if i == len(self.valid_nodes) - 1 else node_id
            is_valid_resource = 1 if self.features[i, 6] == 0 else 0
            is_allowed_node = 1 if node_id not in self.forbidden_nodes else 0
            node_data = self.graph.nodes[actual_node] if actual_node in self.graph.nodes else {}
            is_active = 1 if node_data.get("is_active", True) else 0
            mask.append(is_valid_resource * is_allowed_node * is_active)
        return np.array(mask, dtype=np.int8)

    def set_prev_active_nodes(self, nodes) -> None:
        self.prev_active_nodes = set(nodes) if nodes else set()

    # =====================================================================
    # Observações e features
    # =====================================================================

    def _get_obs(self) -> dict[str, np.ndarray]:
        node_feats = self.features[:, [0, 1, 2, 3, 4, 5, 7]].astype(np.float32)
        node_feats[:, 4] /= 100.0
        node_feats[:, 5] /= 100.0

        is_active = np.zeros(len(self.valid_nodes), dtype=np.float32)
        for i, node_id in enumerate(self.valid_nodes):
            if i == len(self.valid_nodes) - 1:
                node_id = self.current_sfc.dst_node
            if node_id in self.active_nodes:
                is_active[i] = 1.0
        node_feats = np.concatenate([node_feats, is_active[:, None]], axis=1)

        adj_matrix = self._get_adjacency()
        sfc_seq, sfc_seq_len = self._get_sfc_seq()

        return {
            "node_feats": node_feats,
            "adj_matrix": adj_matrix,
            "sfc_seq": sfc_seq,
            "sfc_seq_len": np.array([sfc_seq_len], dtype=np.float32),
        }

    def _get_adjacency(self) -> np.ndarray:
        pair_lengths = self._pair_lengths()
        n = len(self.valid_nodes)
        dst = self.current_sfc.dst_node
        adj = np.zeros((n, n), dtype=np.float32)

        for i, a in enumerate(self.valid_nodes):
            for j, b in enumerate(self.valid_nodes):
                if i == j:
                    continue
                na = dst if i == n - 1 else a
                nb = dst if j == n - 1 else b
                if na == nb:
                    continue
                hops = pair_lengths.get(na, {}).get(nb)
                adj[i, j] = float(hops) if hops is not None else 10.0
        return adj

    def _pair_lengths(self) -> dict:
        key = id(self.graph)
        if key not in self._pair_lengths_cache:
            self._pair_lengths_cache[key] = dict(nx.all_pairs_shortest_path_length(self.graph))
        return self._pair_lengths_cache[key]

    def _get_sfc_seq(self) -> tuple[np.ndarray, int]:
        seq = np.zeros((MAX_VNFS, VNF_FEAT_DIM), dtype=np.float32)
        vnfs = self.reverse_vnf_list
        seq_len = min(len(vnfs), MAX_VNFS)
        for i, vnf in enumerate(vnfs[:seq_len]):
            req = self.service_requirements.get(vnf.id, {})
            seq[i] = [
                float(req.get("cpu", 0.0)) / 100.0,
                float(req.get("cache", 0.0)) / 100.0,
                float(req.get("out_bw", 0.0)) / 1000.0,
                float(req.get("in_bw", 0.0)) / 1000.0,
                1.0 if vnf_is_shareable(vnf) else 0.0,
            ]
        return seq, seq_len

    def _get_nodes_features(self, vnf: VNF, bw_required, current_location) -> np.ndarray:
        num_valid_nodes = len(self.valid_nodes)
        features = np.zeros((num_valid_nodes, 8))

        if not vnf:
            features[:, 6] = 1
            return features

        for i, node_id in enumerate(self.valid_nodes):
            if i == num_valid_nodes - 1:
                node_id = self.current_sfc.dst_node
                features[i, 7] = 1

            node_data = self.graph.nodes[node_id]
            if not node_data.get("is_active", True):
                features[i, 6] = 1
                continue
            is_reusable = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)
            features[i, 2] = float(is_reusable)

            cpu_req = 0.0 if is_reusable else (vnf.get_cpu_request() or 0.0)
            cache_req = 0.0 if is_reusable else (vnf.get_cache_request() or 0.0)

            c_cap = node_data.get("cpu_capacity", 0.0)
            ca_cap = node_data.get("cache_capacity", 0.0)
            features[i, 0] = (
                (node_data.get("cpu_used", 0) + cpu_req) / c_cap if c_cap > 0 else 0.0
            )
            features[i, 1] = (
                (node_data.get("cache_used", 0) + cache_req) / ca_cap if ca_cap > 0 else 0.0
            )

            if (node_data.get("cpu_used", 0) + cpu_req) > (c_cap + EPSILON) or (
                node_data.get("cache_used", 0) + cache_req
            ) > (ca_cap + EPSILON):
                features[i, 6] = 1

            path = get_available_shortest_path_fast(
                self.graph, current_location, node_id, bw_required
            )
            if not path:
                features[i, 4] = 1.0
                features[i, 5] = 1.0
                features[i, 6] = 1
            else:
                bd_cost, latency_cost = self.calculate_bw_lat_cost(vnf, node_id, path, bw_required)
                features[i, 4] = bd_cost
                features[i, 5] = latency_cost

            features[i, 3] = self.get_dynamic_reliability(node_data)
            if node_id in self.forbidden_nodes:
                features[i, 6] = 1

        first_vnf = self.current_sfc.get_previous_vnf(self.current_sfc.get_dst_vnf())
        second_vnf = first_vnf.get_previous_vnf() if first_vnf else None
        is_1_vnf = first_vnf == self.current_vnf
        is_2_vnf = second_vnf == self.current_vnf
        if not (is_1_vnf or is_2_vnf):
            features[-1, 6] = 1

        return features

    def _get_node_powers(self) -> dict:
        key = id(self.graph)
        if self._powers_cache_key != key:
            self._powers_cache = compute_node_powers(self.graph)
            self._powers_cache_key = key
        return self._powers_cache

    def _mark_active_nodes(self) -> None:
        for node_id, data in self.graph.nodes(data=True):
            # is_active is physical availability and must not be reused as a
            # power-state flag. A cold but healthy server is still allocatable.
            data.setdefault("is_active", True)
            data["power_active"] = node_id in self.prev_active_nodes
            data["power_consumption"] = self._get_node_powers().get(node_id, 0.0)

    # =====================================================================
    # Gerenciamento de recursos
    # =====================================================================

    def allocate_resources_on_node(self, node_id, vnf: VNF, alpha: float = 1.0) -> bool:
        node = self.graph.nodes[node_id]
        can_reuse = self.is_reusable_at_node(self.current_sfc, self.graph, node_id, vnf)

        effective_cpu_req = 0.0 if can_reuse else (vnf.get_cpu_request() or 0.0) * alpha
        effective_cache_req = (
            0.0 if can_reuse else (vnf.get_cache_request() or 0.0)
        )

        cpu_ok = (node.get("cpu_used", 0) + effective_cpu_req) <= (
            node.get("cpu_capacity", 0) + EPSILON
        )
        cache_ok = (node.get("cache_used", 0) + effective_cache_req) <= (
            node.get("cache_capacity", 0) + EPSILON
        )
        if not cpu_ok or not cache_ok:
            return False

        node["cpu_used"] = node.get("cpu_used", 0) + effective_cpu_req
        node["cache_used"] = node.get("cache_used", 0) + effective_cache_req
        return True

    def allocate_bandwidth_along_path(self, path: list, bandwidth_required: float) -> bool:
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges[u, v]
            if (
                edge.get("bandwidth_capacity", 0) - edge.get("bandwidth_used", 0)
            ) < (bandwidth_required - EPSILON):
                return False

        for u, v in zip(path[:-1], path[1:], strict=False):
            self.graph.edges[u, v]["bandwidth_used"] += bandwidth_required
        return True

    def is_reusable_at_node(self, sfc: SFC, graph: Graph, node_id, vnf: VNF) -> bool:
        if not vnf_is_shareable(vnf):
            return False

        try:
            session_id = sfc.session_id
        except AttributeError:
            session_id = str(sfc.id).split("_")[-1]

        clean_id = vnf.id.replace("_b", "")
        services_dict = graph.nodes[node_id].get("services", {})
        for ex_id, ex_sess in services_dict:
            clean_ex_id = ex_id.replace("_b", "")
            if clean_ex_id == clean_id and ex_sess == session_id:
                return True
        return False

    def calculate_bw_lat_cost(self, vnf: VNF, server_id, path: list, bw_required: float):
        latency_cost = calculate_computational_latency(self.graph, server_id, vnf)
        if not path or len(path) < 2:
            return 0, latency_cost

        bw_cost = 0
        for u, v in zip(path[:-1], path[1:], strict=False):
            edge = self.graph.edges.get((u, v), {})
            bd_capacity = edge.get("bandwidth_capacity", None)
            bd_used = edge.get("bandwidth_used", 0)

            latency_cost += calculate_latency_betwen_nodes(self.graph, u, v, vnf)

            if bd_capacity is None or bd_capacity == 0 or bw_required + bd_used > bd_capacity:
                return float(999), latency_cost

            link_cost = 1.0 / (1.0 - ((bw_required + bd_used) / bd_capacity) + 1e-6)
            bw_cost += link_cost

        return bw_cost, latency_cost

    def _fail_step(self, reason: str):
        self.fail_reason = reason
        self.success = False
        self._restore_episode_state()
        reward = self.reward_config.get("failure_penalty", -40.0)

        bw_req = self.service_requirements[self.current_vnf.id]["out_bw"]
        self.features = self._get_nodes_features(
            self.current_vnf, bw_req, self.current_location
        )
        obs = self._get_obs()

        return obs, reward, True, False, {}

    # =====================================================================
    # Helpers de configuração
    # =====================================================================

    def set_forbidden_nodes(self, nodes: list) -> None:
        self.forbidden_nodes = set(nodes)

    def set_current_sfc(self, sfc: SFC) -> None:
        if not sfc:
            raise ValueError("SFC não pode ser None.")
        self.current_sfc = sfc
        self.reverse_vnf_list = self.define_reverse_vnf_list(sfc)
        self.current_vnf = self.reverse_vnf_list[0]
        self.current_location = self.current_sfc.dst_node
        self.servers_used = []

        self.services = []
        self.service_requirements = {}
        for item in sfc.vnfs_dict:
            nome = item["name"]
            self.services.append(nome)
            self.service_requirements[nome] = {
                "cpu": item["CPU"],
                "cache": item["cache"],
                "out_bw": item["out_bw"],
                "in_bw": item["in_bw"],
            }

        self.services.append("dst")
        self.service_requirements["dst"] = {"cpu": 0, "cache": 0, "out_bw": 0, "in_bw": 0}

    def define_reverse_vnf_list(self, sfc: SFC) -> list[VNF]:
        vnf_list: list[VNF] = []
        current_vnf = sfc.get_previous_vnf(sfc.get_dst_vnf())
        while True:
            vnf_list.append(current_vnf)
            if not current_vnf.previous_vnf or current_vnf.previous_vnf.id in ("src", "src_virt"):
                break
            current_vnf = sfc.get_previous_vnf(current_vnf)
        return vnf_list

    def _set_list_graph_sfcs(
        self, list_graph: list[Graph], list_sfc: list[SFC], copy: bool = True
    ) -> None:
        """Vincula grafos/SFCs ao ambiente.

        ``copy=True`` (treinamento): cria cópias seguras para cada episódio.
        ``copy=False`` (inferência): usa o shadow graph compartilhado DIRETAMENTE,
        permitindo que as alocações do PPO acumulem entre SFCs do mesmo lote
        (essencial para o deploy físico não estourar capacidade).
        """
        if not copy:
            self.list_graph = list_graph
            self.list_sfc = list_sfc
            self.initial_resource_snapshot = {}
            try:
                _ = self.action_space
                self.reset()
            except AttributeError:
                pass
            return

        safe_graphs = []
        for g in list_graph:
            sg = nx.Graph()
            for n, d in g.nodes(data=True):
                sg.add_node(n, **_copy.deepcopy(dict(d)))
            for u, v, d in g.edges(data=True):
                sg.add_edge(u, v, **_copy.deepcopy(dict(d)))
            safe_graphs.append(sg)

        self.list_graph = safe_graphs
        self.list_sfc = list_sfc

        try:
            _ = self.action_space
            self.reset()
        except AttributeError:
            pass

    def _initialize_snapshots(self, list_graph: list[Graph] = None) -> dict:
        return {idx: self._snapshot_graph_state(g) for idx, g in enumerate(list_graph)}

    def _restore_graph_resources(self, idx) -> None:
        snap = self.initial_resource_snapshot.get(idx)
        if not snap:
            return
        for n_id, st in snap["nodes"].items():
            if n_id in self.graph.nodes:
                self.graph.nodes[n_id].clear()
                self.graph.nodes[n_id].update(_copy.deepcopy(st))
        for (u, v), st in snap["edges"].items():
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v].clear()
                self.graph.edges[u, v].update(_copy.deepcopy(st))

    @staticmethod
    def _snapshot_graph_state(graph: Graph) -> dict:
        return {
            "nodes": {n: _copy.deepcopy(dict(d)) for n, d in graph.nodes(data=True)},
            "edges": {(u, v): _copy.deepcopy(dict(d)) for u, v, d in graph.edges(data=True)},
        }

    def _restore_episode_state(self) -> None:
        if self._episode_snapshot is None or self.graph is None:
            return
        for node_id, data in self._episode_snapshot["nodes"].items():
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id].clear()
                self.graph.nodes[node_id].update(_copy.deepcopy(data))
        for (u, v), data in self._episode_snapshot["edges"].items():
            if self.graph.has_edge(u, v):
                self.graph.edges[u, v].clear()
                self.graph.edges[u, v].update(_copy.deepcopy(data))
        self.active_nodes.clear()
        self.servers_used.clear()
        self.allocation_results.clear()
        self.latency_used = 0

    def get_dynamic_reliability(self, node_data: dict) -> float:
        cpu_cap = node_data.get("cpu_capacity", 1.0) or node_data.get(
            "original_cpu_capacity", 1.0
        ) or 1.0
        server_level = str(node_data.get("level_server", "default")).lower()

        base_r = self.reliability_config["tiers"].get(
            server_level, self.reliability_config["tiers"]["default"]
        )
        alpha = self.reliability_config["stress"].get(
            server_level, self.reliability_config["stress"]["default"]
        )

        return max(0.0, base_r - (min(node_data.get("cpu_used", 0.0) / cpu_cap, 1.0) * alpha))
