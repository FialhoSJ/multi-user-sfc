from muar_sfc.controllers.vnf_generator import VNFGenerator
from muar_sfc.core.services.catalog import build_default_catalog
from muar_sfc.core.vnf import VNFType


def test_heterogeneous_session_generates_streaming_sfc():
    from muar_sfc.config import SimulationSettings
    from muar_sfc.controllers.sfc_queue import SFCQueue
    from muar_sfc.core.scenarios.muar import MuarScenario

    class FakeEmitter:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class FakeTopology:
        def get_topology_info(self):
            return {"routers": [5, 9]}

    args = SimulationSettings(
        n_sessions=1, n_players=2,
        service_mix="muar,streaming", service_weights="0.5,0.5",
    )
    queue = SFCQueue()
    scenario = MuarScenario(args, queue, FakeTopology(), FakeEmitter())
    template = scenario.service_catalog.get("streaming")
    scenario._generate_heterogeneous_session(template, "7", 2)
    assert queue.qsize() == 2
    sfc = queue.get()[0]
    assert sfc.service_type == "streaming"
    assert sfc.id == "sfc_streaming_p1_7"
    assert abs(sfc.link_bandwidth_dict[("encoder_1_7", "transcoder_1_7")] - 58.8) < 1e-9


def test_chain_propagates_bandwidth_transforms():
    from muar_sfc.controllers.sfc_generator import SFCGenerator
    catalog = build_default_catalog()
    sfc_dict = catalog.get("streaming").build_sfc_dict(
        sfc_name="sfc_stream", src_node=0, dst_node="17",
        closer_router=5, bandwidth=42.0, duration=120,
        player=1, counter=1,
    )
    sfc = SFCGenerator(sfc_dict).generate()
    assert sfc.service_type == "streaming"
    links = sfc.link_bandwidth_dict
    assert abs(links[("packetizer_1_1", "encoder_1_1")] - 42.0) < 1e-9
    assert abs(links[("encoder_1_1", "transcoder_1_1")] - 58.8) < 1e-9
    assert abs(links[("transcoder_1_1", "dst")] - 41.16) < 1e-9


def test_type1_chain_preserves_configured_egress():
    from muar_sfc.controllers.sfc_generator import SFCGenerator
    sfc_dict = {
        "name": "sfc_t1",
        "vnf_list": [
            {"type": VNFType.TYPE1, "name": "a", "CPU": 5, "cache": 0, "in_bw": 10, "out_bw": 12},
            {"type": VNFType.TYPE1, "name": "b", "CPU": 5, "cache": 0, "in_bw": 12, "out_bw": 8},
        ],
        "bandwidth": 10, "src_node": 0, "dst_node": 9, "closer_router": 1,
        "latency": 100, "duration": 120,
    }
    sfc = SFCGenerator(sfc_dict).generate()
    links = sfc.link_bandwidth_dict
    assert links[("a", "b")] == 12
    assert links[("b", "dst")] == 8


def test_vnf_type1_behavior_unchanged():
    vnf = VNFGenerator.generate(
        {"type": VNFType.TYPE1, "name": "v1", "CPU": 10, "cache": 0,
         "in_bw": 5, "out_bw": 5, "service_type": "muar"}
    )
    assert vnf.type == VNFType.TYPE1
    assert vnf.get_cpu_request() == 10
    assert vnf.vnf_bw(5) == 5
    assert vnf.service_type == "muar"


def test_vnf_type2_amplifies_bandwidth():
    vnf = VNFGenerator.generate(
        {"type": VNFType.TYPE2, "name": "enc", "CPU": 45, "cache": 0,
         "in_bw": 10, "out_bw": 10, "params": {"factor": 1.4},
         "service_type": "streaming"}
    )
    assert vnf.type == VNFType.TYPE2
    assert vnf.vnf_bw(10) == 14.0


def test_vnf_type3_compresses_bandwidth():
    vnf = VNFGenerator.generate(
        {"type": VNFType.TYPE3, "name": "transc", "CPU": 35, "cache": 0,
         "in_bw": 10, "out_bw": 10, "params": {"factor": 0.7}}
    )
    assert vnf.type == VNFType.TYPE3
    assert vnf.vnf_bw(10) == 7.0


def test_vnf_type4_aggregates():
    vnf = VNFGenerator.generate(
        {"type": VNFType.TYPE4, "name": "agg", "CPU": 6, "cache": 0,
         "in_bw": 10, "out_bw": 1, "params": {"out_bw": 3.0}}
    )
    assert vnf.type == VNFType.TYPE4
    assert vnf.vnf_bw(999) == 3.0


def test_unregistered_type_raises():
    import pytest
    with pytest.raises(ValueError):
        VNFGenerator.generate(
            {"type": 999, "name": "x", "CPU": 1, "cache": 0,
             "in_bw": 1, "out_bw": 1}
        )


def test_catalog_has_services():
    catalog = build_default_catalog()
    assert set(catalog.list()) >= {"muar", "streaming", "voip", "iot"}


def test_weighted_choice_subset_mix():
    catalog = build_default_catalog()
    # FIX: mix parcial (subconjunto do catálogo) não pode dar KeyError
    assert catalog.weighted_choice(weights={"muar": 1.0}).name == "muar"
    t = catalog.weighted_choice(weights={"muar": 0.5, "streaming": 0.5})
    assert t.name in {"muar", "streaming"}


def test_catalog_builds_chain_with_placeholders():
    catalog = build_default_catalog()
    template = catalog.get("streaming")
    chain = template.build_chain(player=3, counter=7)
    assert len(chain) == 3
    assert all(v["service_type"] == "streaming" for v in chain)
    assert chain[0]["name"] == "packetizer_3_7"
    assert chain[1]["type"] == VNFType.TYPE2
    assert chain[2]["type"] == VNFType.TYPE3


def test_catalog_builds_sfc_dict():
    catalog = build_default_catalog()
    template = catalog.get("iot")
    sfc_dict = template.build_sfc_dict(
        sfc_name="sfc_iot",
        src_node=0,
        dst_node="17",
        closer_router=5,
        bandwidth=1.0,
        duration=120,
        player=1,
        counter=2,
    )
    assert sfc_dict["name"] == "sfc_iot"
    assert sfc_dict["latency"] == 200.0
    assert sfc_dict["service_type"] == "iot"
    assert len(sfc_dict["vnf_list"]) == 2


def test_catalog_sets_shareable_flag():
    catalog = build_default_catalog()
    muar_chain = catalog.get("muar").build_chain(player=1, counter=1)
    streaming_chain = catalog.get("streaming").build_chain(player=1, counter=1)
    assert all(v["shareable"] for v in muar_chain)
    assert all(not v["shareable"] for v in streaming_chain)


def test_allocator_no_double_charge_on_redeploy():
    """FIX: mesma (nome, sessão, nó) já instanciada não pode cobrar CPU de novo."""
    import time

    from muar_sfc.controllers.modules.sfc_deployer import SFCDeployer
    from muar_sfc.controllers.modules.sfc_state_tracker import SFCStateTracker
    from muar_sfc.controllers.sfc_generator import SFCGenerator
    from muar_sfc.core.net_v2 import Net2

    net = Net2()
    net.set_sharing_params("y")
    net.add_node(0, "server", cpu_capacity=1000, cache_capacity=1000, ips=100, position=(0, 0))
    net.add_node(1, "server", cpu_capacity=500, cache_capacity=500, ips=100, position=(1, 1))
    net.add_node("11", "mobile_device", cpu_capacity=25, cache_capacity=10,
                 ips=0.1, position=(9, 9))
    net.add_edge(0, 1, bandwidth_capacity=5000, latency=1)
    net.add_edge(1, "11", bandwidth_capacity=2000, latency=1)

    tracker = SFCStateTracker()
    deployer = SFCDeployer(tracker)

    def make_sfc(name, dst):
        # VNF não-compartilhável com nome fixo (simula re-deploy da mesma SFC)
        return SFCGenerator({
            "name": name,
            "vnf_list": [
                {"type": VNFType.TYPE1, "name": "UNI_1", "CPU": 20, "cache": 0,
                 "in_bw": 5, "out_bw": 5},
            ],
            "bandwidth": 5, "src_node": 0, "dst_node": dst, "closer_router": 1,
            "latency": 1e9, "duration": 120, "service_type": "muar",
        }).generate()

    s1 = make_sfc("sfc_unique_p1_1", "11")
    route = {"UNI_1": [1], "src": [0, 1], "dst": [1, "11"]}

    # 1º deploy
    deployer.submit_solution([s1], {s1.id: {"route_info": route}}, net)
    assert net.graph.nodes[1]["cpu_used"] == 20

    # requeue realista: libera o antigo antes de re-deployar (como na recuperação de falha)
    deployer.safe_network_removal("sfc_unique_p1_1", net)
    assert net.graph.nodes[1]["cpu_used"] == 0

    # re-deploy (mesma SFC) não pode cobrar de novo além da capacidade
    deployer.submit_solution([s1], {s1.id: {"route_info": route}}, net)
    assert net.graph.nodes[1]["cpu_used"] == 20, net.graph.nodes[1]["cpu_used"]

    # cleanup libera tudo
    tracker.sfcs_tracker["11"]["timer"] = time.time() - 200
    for sfc_id in tracker.cleanup_session_state("11"):
        deployer.safe_network_removal(sfc_id, net)
    assert net.graph.nodes[1]["cpu_used"] == 0
    assert net.metrics.total_cpu_used == 0


def test_vnf_carries_shareable_flag():
    vnf = VNFGenerator.generate(
        {"type": VNFType.TYPE1, "name": "enc_1_1", "CPU": 5, "cache": 0,
         "in_bw": 5, "out_bw": 5, "shareable": True, "service_type": "streaming"}
    )
    assert vnf.shareable is True
    assert vnf.get_shareable() is True


def test_vnf_is_shareable_helper():
    from muar_sfc.core.infrastructure.enums import vnf_is_shareable
    catalog = build_default_catalog()
    muar_vnf = VNFGenerator.generate(
        catalog.get("muar").build_chain(player=1, counter=1)[0]
    )
    streaming_vnf = VNFGenerator.generate(
        {"type": VNFType.TYPE1, "name": "packetizer_1_1", "CPU": 5, "cache": 0,
         "in_bw": 40, "out_bw": 42, "service_type": "streaming"}
    )
    legacy_muar_vnf = VNFGenerator.generate(
        {"type": VNFType.TYPE1, "name": "IA_DET_FT_3", "CPU": 5, "cache": 0,
         "in_bw": 150, "out_bw": 151}
    )
    assert vnf_is_shareable(muar_vnf) is True        # flag do catálogo
    assert vnf_is_shareable(streaming_vnf) is False
    assert vnf_is_shareable(legacy_muar_vnf) is True  # prefixo MUAR legado


def _make_dry_run_sfc(latency_request: float):
    from muar_sfc.controllers.sfc_generator import SFCGenerator
    return SFCGenerator({
        "name": "s1",
        "vnf_list": [
            {"type": VNFType.TYPE1, "name": "myvnf", "CPU": 1, "cache": 0,
             "in_bw": 5, "out_bw": 5, "service_type": "voip"},
        ],
        "bandwidth": 5, "src_node": 0, "dst_node": 9, "closer_router": 1,
        "latency": latency_request, "duration": 120,
    }).generate()


def _dry_run_graph():
    import networkx as nx

    g = nx.Graph()
    g.add_node(0, type="server", ips=100, cpu_capacity=1000, cache_capacity=1000,
               cpu_used=0, cache_used=0)
    g.add_node(1, type="server", ips=100, cpu_capacity=1000, cache_capacity=1000,
               cpu_used=0, cache_used=0)
    return g


def _dry_run_instantiator():
    from muar_sfc.config import SimulationSettings
    from muar_sfc.controllers.modules.sfcs_instatiator import SFCInstatiator

    return SFCInstatiator(alg=None, args=SimulationSettings())


def test_dry_run_rejects_latency_sla_violation():
    import pytest

    sfc = _make_dry_run_sfc(latency_request=0.001)  # SLA absurdamente apertado
    with pytest.raises(ValueError):
        _dry_run_instantiator()._simulate_dry_run(_dry_run_graph(), sfc, {"myvnf": [1]})


def test_dry_run_accepts_within_sla():
    sfc = _make_dry_run_sfc(latency_request=1e9)  # SLA folgado
    result = _dry_run_instantiator()._simulate_dry_run(_dry_run_graph(), sfc, {"myvnf": [1]})
    assert len(result) == 4


def test_shadow_graph_isolates_mutable_state():
    """FIX: _create_shadow_graph não pode compartilhar services/sfcs_list/reuse com a rede real."""
    from muar_sfc.core.net_v2 import Net2

    net = Net2()
    net.add_node(1, "server", cpu_capacity=100, cache_capacity=100, ips=100, position=(1, 1))
    net.add_node(2, "server", cpu_capacity=100, cache_capacity=100, ips=100, position=(2, 2))
    net.add_edge(1, 2, bandwidth_capacity=5000, latency=1)

    net.graph.nodes[1]["services"][("IA_DET_FT_1", "1")] = {"cpu": 22, "cache": 0, "copys": 1}

    inst = _dry_run_instantiator()
    shadow = inst._create_shadow_graph(net)

    # muta o shadow (como o dry-run/algoritmo faz)
    shadow.nodes[1]["services"][("X", "9")] = {"cpu": 1, "cache": 0, "copys": 1}
    shadow.nodes[1]["services"][("IA_DET_FT_1", "1")]["cpu"] = 999

    # a rede real NÃO pode ser poluída (antes compartilhava o dict raso)
    assert ("X", "9") not in net.graph.nodes[1]["services"]
    assert net.graph.nodes[1]["services"][("IA_DET_FT_1", "1")]["cpu"] == 22


def test_dry_run_shared_vnf_no_double_charge():
    from muar_sfc.controllers.sfc_generator import SFCGenerator

    g = _dry_run_graph()  # nós 0 e 1 (server)
    g.nodes[1]["cpu_capacity"] = 1000
    g.nodes[1]["cache_capacity"] = 1000

    def make_sfc(name, vnfs):
        return SFCGenerator({
            "name": name,
            "vnf_list": vnfs,
            "bandwidth": 5, "src_node": 0, "dst_node": "17", "closer_router": 1,
            "latency": 1e9, "duration": 120, "service_type": "muar",
        }).generate()

    cache = make_sfc("sfc_cache_p1_7", [
        {"type": VNFType.TYPE1, "name": "IA_DET_FT_7", "CPU": 22, "cache": 0,
         "in_bw": 5, "out_bw": 5},
        {"type": VNFType.TYPE1, "name": "MA_region_1", "CPU": 10, "cache": 0,
         "in_bw": 5, "out_bw": 5},
    ])
    unique = make_sfc("sfc_unique_p1_7", [
        {"type": VNFType.TYPE1, "name": "IA_DET_FT_7", "CPU": 22, "cache": 0,
         "in_bw": 5, "out_bw": 5},
        {"type": VNFType.TYPE1, "name": "UNI_1", "CPU": 20, "cache": 0,
         "in_bw": 5, "out_bw": 5},
    ])

    inst = _dry_run_instantiator()
    inst._simulate_dry_run(g, cache, {"IA_DET_FT_7": [1], "MA_region_1": [1]})
    assert g.nodes[1]["cpu_used"] == 32
    inst._simulate_dry_run(g, unique, {"IA_DET_FT_7": [1], "UNI_1": [1]})
    # IA_DET_FT_7 já instanciado (cache+unique compartilham): reuso sem custo.
    # Antes do fix, o dry-run cobrava 22 de novo => 74. Agora soma só UNI (20) => 52.
    assert g.nodes[1]["cpu_used"] == 52


def test_extract_service_type_from_sfc_object():
    from muar_sfc.controllers.sfc_generator import SFCGenerator
    from muar_sfc.utils.manager_results import extract_service_type

    sfc = SFCGenerator({
        "name": "sfc_streaming_p1_3",
        "vnf_list": [{"type": VNFType.TYPE1, "name": "v", "CPU": 1, "cache": 0,
                      "in_bw": 5, "out_bw": 5}],
        "bandwidth": 5, "src_node": 0, "dst_node": "13", "closer_router": 1,
        "latency": 150, "duration": 120, "service_type": "streaming",
    }).generate()

    class FakeNet:
        sfc_dict = {"sfc_streaming_p1_3": sfc}

    assert extract_service_type("sfc_streaming_p1_3", FakeNet()) == "streaming"


def test_extract_service_type_fallback_by_id():
    from muar_sfc.utils.manager_results import extract_service_type

    class FakeNet:
        sfc_dict = {}

    assert extract_service_type("sfc_streaming_p1_3", FakeNet()) == "streaming"
    assert extract_service_type("sfc_cache_p1_3", FakeNet()) == "muar"
    assert extract_service_type("sfc_unique_p1_3", FakeNet()) == "muar"


def test_write_service_summary(tmp_path):
    from muar_sfc.utils.manager_results import OutputWritter

    class FakeTopo:
        def get_topology_info(self):
            return {"ec_servers": [1, 2], "edges": [(1, 2)]}

    services_file = tmp_path / "service_summary.csv"
    services_file.write_text(
        "service_type,requests,accepted,rejected,acceptance_rate,"
        "avg_latency_ms,avg_comp_latency_ms,avg_comm_latency_ms,cpu_saved,cache_saved\n"
    )
    ow = OutputWritter(
        FakeTopo(),
        {k: tmp_path / f"{k}.csv" for k in ("cpu", "gpu", "cache", "bandwidth", "sf")},
        tmp_path / "flows.csv", tmp_path / "crash.csv",
        tmp_path / "res.csv", services_file,
    )
    ow.service_stats["streaming"]["requests"] = 2
    ow.service_stats["streaming"]["accepted"] = 1
    ow.service_stats["streaming"]["latency"] = 100.0
    ow.write_service_summary()

    lines = services_file.read_text().strip().splitlines()
    assert len(lines) == 2  # header + 1 linha
    assert lines[1].split(",")[0] == "streaming"
    assert lines[1].split(",")[4] == "50.00"  # acceptance_rate = 1/2 = 50%


def test_cli_overrides_parse():
    from muar_sfc.main import _parse_bool, build_cli_parser, build_settings_overrides

    assert _parse_bool("on") is True
    assert _parse_bool("n") is False
    assert _parse_bool(None) is None

    args = build_cli_parser().parse_args([
        "--alg", "greedyb", "--n_sessions", "10", "--sfc", "on",
        "--verbose", "n", "--crash_at", "400", "520",
        "--service_mix", "muar,streaming", "--service_weights", "0.7,0.3",
    ])
    ov = build_settings_overrides(args)
    assert ov["alg"] == "greedyb"
    assert ov["n_sessions"] == 10
    assert ov["sfc"] is True
    assert ov["verbose"] is False
    assert ov["crash_at"] == [400.0, 520.0]
    assert ov["service_mix"] == "muar,streaming"
    assert ov["service_weights"] == "0.7,0.3"


def test_rebuild_preserves_service_type():
    from time import time

    from muar_sfc.controllers.modules.sfc_state_tracker import SFCStateTracker
    from muar_sfc.controllers.sfc_generator import SFCGenerator

    sfc = SFCGenerator({
        "name": "sfc_streaming_p1_9",
        "vnf_list": [
            {"type": VNFType.TYPE1, "name": "packetizer_1_9", "CPU": 15, "cache": 0,
             "in_bw": 40, "out_bw": 42, "service_type": "streaming", "shareable": False},
        ],
        "bandwidth": 40, "src_node": 0, "dst_node": "19", "closer_router": 5,
        "latency": 150, "duration": 120, "service_type": "streaming",
    }).generate()

    tracker = SFCStateTracker()
    tracker.sfcs_tracker["19"] = {
        "sfc_list": [sfc], "solution": {}, "duration": 120.0, "timer": time(),
    }
    rebuilt, remaining = tracker.rebuild_sfcs_for_requeue(
        [sfc], changed_location=False, new_location=None
    )
    assert len(rebuilt) == 1
    assert remaining > 1.0
    assert rebuilt[0].service_type == "streaming"  # F5: herança preservada no requeue


def _fake_substrate(g):
    import networkx as nx

    class FakeSubstrate:
        def __init__(self, graph):
            self.g = graph
            self._node = graph._node

        def edges(self, n):
            pass

        def get_node_cpu_free(self, n):
            return self.g.nodes[n]["cpu_capacity"] - self.g.nodes[n]["cpu_used"]

        def get_node_cache_free(self, n):
            return self.g.nodes[n]["cache_capacity"] - self.g.nodes[n]["cache_used"]

        def get_node_cpu_capacity(self, n):
            return self.g.nodes[n]["cpu_capacity"]

        def get_node_cache_capacity(self, n):
            return self.g.nodes[n]["cache_capacity"]

        def get_node_cpu_used(self, n):
            return self.g.nodes[n]["cpu_used"]

        def get_node_cache_used(self, n):
            return self.g.nodes[n]["cache_used"]

        def get_link_latency(self, u, v):
            return self.g[u][v]["latency"]

        def get_shortest_path_length(self, u, v):
            return nx.dijkstra_path_length(self.g, u, v, weight="latency")

        def get_shortest_path(self, u, v):
            return nx.dijkstra_path(self.g, u, v, weight="latency")

        def get_shortest_path_with_bw(self, u, v, bw):
            valid = nx.subgraph_view(
                self.g,
                filter_edge=lambda a, b: (
                    self.g[a][b]["bandwidth_capacity"] - self.g[a][b]["bandwidth_used"]
                ) >= bw,
            )
            try:
                return nx.dijkstra_path(valid, u, v, weight="latency")
            except nx.NetworkXException:
                return None

    return FakeSubstrate(g)


def test_greedy_supports_heterogeneous_chain():
    import networkx as nx

    from muar_sfc.algorithms.greedy_algorithm import GreedyAlgorithm
    from muar_sfc.controllers.sfc_generator import SFCGenerator

    g = nx.Graph()
    g.add_node(0, cpu_capacity=1000, cache_capacity=1000, cpu_used=0, cache_used=0, position=1)
    for n in (1, 2, 3, 4):
        g.add_node(n, cpu_capacity=1000, cache_capacity=1000,
                   cpu_used=0, cache_used=0, position=1)
    g.add_node("19", cpu_capacity=100, cache_capacity=100,
               cpu_used=0, cache_used=0, position=1)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, "19"), (0, 2), (2, 4)]:
        g.add_edge(u, v, bandwidth_capacity=1000, bandwidth_used=0, latency=1)

    sfc = SFCGenerator({
        "name": "sfc_voip_p1_9",
        "vnf_list": [
            {"type": VNFType.TYPE1, "name": "jitter_1_9", "CPU": 5, "cache": 0,
             "in_bw": 0.1, "out_bw": 0.1, "service_type": "voip", "shareable": False},
            {"type": VNFType.TYPE1, "name": "codec_1_9", "CPU": 8, "cache": 0,
             "in_bw": 0.1, "out_bw": 0.1, "service_type": "voip", "shareable": False},
        ],
        "bandwidth": 0.1, "src_node": 0, "dst_node": "19", "closer_router": 3,
        "latency": 1000, "duration": 120, "service_type": "voip",
    }).generate()

    alg = GreedyAlgorithm()
    alg.install_substrate_network(_fake_substrate(g))
    alg.install_SFC(sfc)
    assert alg.start_algorithm() is True  # F5: cadeia heterogênea (2 VNFs) não é mais fixada em 6
    route = alg.get_route_info()
    assert set(route) == {"src", "jitter_1_9", "codec_1_9", "dst"}
