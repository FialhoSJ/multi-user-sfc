import networkx as nx

from muar_sfc.algorithms.hybrid.cost import HybridSFCCostEvaluator, compute_node_powers
from muar_sfc.core.vnf import VNF


def _make_graph():
    g = nx.Graph()
    g.add_node(0, type="server", level_server="c", ips=1e9, cpu_capacity=1000,
               cache_capacity=1000, cpu_used=0, cache_used=0, position=(0, 0))
    g.add_node(1, type="server", level_server="b", ips=1e9, cpu_capacity=500,
               cache_capacity=500, cpu_used=0, cache_used=0, position=(1, 1))
    g.add_node(2, type="server", level_server="a", ips=1e9, cpu_capacity=100,
               cache_capacity=100, cpu_used=0, cache_used=0, position=(2, 2))
    for u, v in [(0, 1), (1, 2)]:
        g.add_edge(u, v, bandwidth_capacity=1000, bandwidth_used=0, latency=1)
    return g


def _make_vnf(income_bw=5.0):
    vnf = VNF("codec_1_1")
    vnf.set_cpu_request(10.0)
    vnf.set_cache_request(0.0)
    vnf.set_income_interface_bandwidth(income_bw)
    vnf.set_outcome_interface_bandwidth(income_bw)
    return vnf


def test_bandwidth_cost_penalizes_extra_hops():
    evaluator = HybridSFCCostEvaluator()
    assert evaluator.compute_bandwidth_cost([0, 1], req_bw=5.0) == 0.0
    assert evaluator.compute_bandwidth_cost([0, 1, 2], req_bw=5.0) == 5.0
    assert evaluator.compute_bandwidth_cost([], req_bw=5.0) == 0.0


def test_operational_cost_includes_power_and_wear():
    evaluator = HybridSFCCostEvaluator(alpha_power=0.1, beta_wear=0.1)
    node_powers = {1: 100.0}

    cost_ligando = evaluator.compute_operational_cost({1}, set(), node_powers)
    assert abs(cost_ligando - (10.0 + 0.1)) < 1e-6

    cost_sem_transicao = evaluator.compute_operational_cost({1}, {1}, node_powers)
    assert abs(cost_sem_transicao - 10.0) < 1e-6

    cost_mais_potencia = evaluator.compute_operational_cost({1, 2}, {1}, {1: 100.0, 2: 50.0})
    assert abs(cost_mais_potencia - (0.1 * 150.0 + 0.1 * 1)) < 1e-6


def test_compute_node_powers_uses_energy_model():
    graph = _make_graph()
    powers = compute_node_powers(graph)
    assert 0 in powers and powers[0] > 0
    assert powers[0] != powers[1]  # níveis a/b/c distintos


def test_latency_alpha_increases_processing_time():
    evaluator = HybridSFCCostEvaluator()
    graph = _make_graph()
    vnf = _make_vnf()

    lat_full = evaluator.compute_latency(graph, [1, 2], vnf, alpha=1.0, node=2)
    lat_metade = evaluator.compute_latency(graph, [1, 2], vnf, alpha=0.5, node=2)
    assert lat_metade > lat_full
    assert lat_full > 0.0


def test_reward_penalizes_sla_violation():
    evaluator = HybridSFCCostEvaluator()
    reward_ok, metrics = evaluator.calculate_reward(
        c_banda=0.0, c_operacional=0.0, lat=5.0, sla_latency=100.0
    )
    assert metrics["sla_penalty"] == 0.0
    assert reward_ok > 0.0

    reward_bad, metrics_bad = evaluator.calculate_reward(
        c_banda=0.0, c_operacional=0.0, lat=25.0, sla_latency=5.0
    )
    assert metrics_bad["sla_penalty"] > 0.0
    assert reward_bad < reward_ok

    reward_higher_lat, _ = evaluator.calculate_reward(
        c_banda=0.0, c_operacional=0.0, lat=10.0, sla_latency=5.0
    )
    assert reward_higher_lat < reward_ok
