import networkx as nx
import numpy as np

from muar_sfc.algorithms.environments.env_replic import build_valid_nodes
from muar_sfc.algorithms.hybrid.hybrid_env import SFC_AllocationEnv_Hybrid
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.vnf import VNFType


def _make_graph():
    g = nx.Graph()
    g.add_node(0, type="server", level_server="c", ips=1e9, cpu_capacity=1000,
               cache_capacity=1000, cpu_used=0, cache_used=0, position=(0, 0))
    g.add_node(1, type="server", level_server="b", ips=1e9, cpu_capacity=6,
               cache_capacity=500, cpu_used=0, cache_used=0, position=(1, 1))
    g.add_node("11", type="mobile_device", level_server="a", ips=1e9, cpu_capacity=25,
               cache_capacity=10, cpu_used=0, cache_used=0, position=900)
    g.add_edge(0, 1, bandwidth_capacity=5000, bandwidth_used=0, latency=1)
    g.add_edge(1, "11", bandwidth_capacity=5000, bandwidth_used=0, latency=1)
    return g


def _make_sfc():
    return SFCGenerator({
        "name": "sfc_voip_p1_1",
        "vnf_list": [
            {"type": VNFType.TYPE1, "name": "codec_1_1", "CPU": 10.0, "cache": 0,
             "in_bw": 5, "out_bw": 5, "service_type": "voip", "shareable": False},
        ],
        "bandwidth": 5, "src_node": 0, "dst_node": "11", "closer_router": 1,
        "latency": 1000, "duration": 120, "service_type": "voip",
    }).generate()


def _make_env():
    graph = _make_graph()
    sfc = _make_sfc()
    env = SFC_AllocationEnv_Hybrid(
        valid_nodes=build_valid_nodes(graph),
        list_graph=[graph],
        list_sfc=[sfc],
        is_training=False,
    )
    return env, graph, sfc


def test_reset_produces_hybrid_observation():
    env, _, _ = _make_env()
    obs, _ = env.reset()

    assert set(obs) == {"node_feats", "adj_matrix", "sfc_seq", "sfc_seq_len"}
    n = len(env.valid_nodes)
    assert obs["node_feats"].shape == (n, env.node_feat_dim)
    assert obs["adj_matrix"].shape == (n, n)
    assert obs["sfc_seq"].shape == (env.max_vnfs, env.vnf_feat_dim)
    assert obs["sfc_seq_len"][0] == 1

    space = env.observation_space
    assert space.contains(obs)


def test_action_space_is_hybrid():
    env, _, _ = _make_env()
    space = env.action_space
    alpha_ok = np.array([0.5], dtype=np.float32)
    alpha_max = np.array([1.0], dtype=np.float32)
    alpha_invalid = np.array([1.5], dtype=np.float32)

    assert space.contains((1, alpha_ok))
    assert space.contains((len(env.valid_nodes) - 1, alpha_max))
    assert not space.contains((1, alpha_invalid))


def test_step_alpha_scales_cpu_allocation():
    env, _, _ = _make_env()
    obs, _ = env.reset()

    node_idx = env.valid_nodes.index(1)  # servidor 1 (cpu_capacity=6)
    next_obs, reward, done, _, _ = env.step((node_idx, 0.5))

    assert env.graph.nodes[1]["cpu_used"] == 5.0  # 10 * 0.5
    assert done is True
    assert env.success is True
    assert env.latency_used > 0.0
    assert env.prev_active_nodes == {1}


def test_step_alpha_over_capacity_fails():
    env, _, _ = _make_env()
    obs, _ = env.reset()

    node_idx = env.valid_nodes.index(1)
    next_obs, reward, done, _, _ = env.step((node_idx, 1.0))  # 10 > 6

    assert done is True
    assert env.success is False
    assert env.fail_reason == "resource"
    assert env.graph.nodes[1]["cpu_used"] == 0.0


def test_prev_active_nodes_persist_on_graph():
    env, _, _ = _make_env()
    env.set_prev_active_nodes({1})
    obs, _ = env.reset()

    assert env.graph.nodes[1]["is_active"] is True
    assert env.graph.nodes[0]["is_active"] is False
    assert env.graph.nodes[1]["power_consumption"] > 0.0
