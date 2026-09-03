"""Gera o dataset de treino RL heterogêneo (list_graph / list_sfc) para o REPLIC.

Cada amostra é composta por:
  - um grafo substrato "fresco" (recursos zerados) da topologia configurada, com
    um dispositivo móvel conectado a um roteador;
  - uma SFC de um serviço sorteado do catálogo (muar/streaming/voip/iot).

Uso:
    uv run python scripts/gerar_dataset_rl.py --n_samples 300

Saída: variaveis_salvas/list_graph_het.pkl e list_sfc_het.pkl
(consumidos por rl_saved_models/treinar_rl.py --env REPLIC)
"""

from __future__ import annotations

import argparse
import copy
import pickle
import random
from pathlib import Path

import numpy as np

from muar_sfc.config import SimulationSettings
from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.services.catalog import build_default_catalog
from muar_sfc.topology.instantiator import TopologyInstantiator

SAVE_DIR = Path("variaveis_salvas")


def _add_mobile(graph, mobile_id, closer_router):
    """Adiciona um dispositivo móvel ligado ao roteador (espelha add_mobile_user_to_graph)."""
    graph.add_node(
        mobile_id,
        type="mobile_device",
        cpu_capacity=25.0,
        cache_capacity=10.0,
        cpu_used=0.0,
        cache_used=0.0,
        ips=1e9,
        position=random.uniform(100, 5000),  # distância escalar (como na simulação)
        reuse=[],
        services={},
        sfcs_list=[],
        is_active=True,
    )
    router = graph.nodes[closer_router]
    wireless_free = router.get("w_channel_capacity", 0.0) - router.get("w_channel_used", 0.0)
    graph.add_edge(
        mobile_id,
        closer_router,
        bandwidth_capacity=max(wireless_free, 0.0),
        bandwidth_used=0.0,
        latency=1,
        services_in_transit={},
    )
    return graph


def _prepopulate_prior_player(graph, sfc, session, servers):
    """Simula a 1ª instância de um player anterior no mesmo lote/sessão.

    Cria oportunidades de REUSO (VNFs shareable) para o modelo aprender a
    compartilhar: o env verá is_reusable=True nos nós que já hospedam a VNF.
    """
    for vnf_dict in sfc.vnfs_dict:
        node = random.choice(servers)
        key = (vnf_dict["name"], session)
        services = graph.nodes[node].setdefault("services", {})
        services[key] = {
            "cpu": float(vnf_dict["CPU"]),
            "cache": float(vnf_dict["cache"]),
            "copys": 1,
        }
        graph.nodes[node]["cpu_used"] = round(
            graph.nodes[node].get("cpu_used", 0.0) + float(vnf_dict["CPU"]), 2
        )
        graph.nodes[node].setdefault("sfcs_list", []).append("prior_" + sfc.id)
        if key[0].startswith(("IA_DET_FT_", "MA_region_", "RE_region_")):
            graph.nodes[node].setdefault("reuse", []).append(key[0])
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dataset RL heterogêneo.")
    parser.add_argument("--n_samples", type=int, default=300, help="nº de amostras")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--service_mix", type=str, default="muar,streaming,voip,iot")
    parser.add_argument("--service_weights", type=str, default="0.4,0.2,0.2,0.2")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    settings = SimulationSettings()
    topo = TopologyInstantiator().instantiate_topology(settings.topology, settings.eco_effi_ratio)
    net = topo.generate_substrate_network()
    base_graph = net.graph
    routers = [n for n, d in base_graph.nodes(data=True)
               if d.get("type") == "router" and n != 0]
    if not routers:
        raise RuntimeError("Topologia sem roteadores para conectar o dispositivo móvel.")

    catalog = build_default_catalog()
    mix = [s.strip() for s in args.service_mix.split(",")]
    raw = [float(w.strip()) for w in args.service_weights.split(",")]
    weights = dict(zip(mix, raw, strict=True))

    list_graph: list = []
    list_sfc: list = []
    counts = {name: 0 for name in mix}

    for i in range(args.n_samples):
        graph = copy.deepcopy(base_graph)
        mobile_id = f"M{i}"
        closer = random.choice(routers)
        graph = _add_mobile(graph, mobile_id, closer)

        service = catalog.weighted_choice(weights=weights)
        counts[service.name] = counts.get(service.name, 0) + 1
        counter = str(i + 1)
        chain = service.build_chain(player=1, counter=counter)
        input_bw = chain[0]["in_bw"]

        sfc_dict = service.build_sfc_dict(
            sfc_name=f"sfc_{service.name}_p1_{counter}",
            src_node=0,
            dst_node=mobile_id,
            closer_router=closer,
            bandwidth=input_bw,
            duration=120,
            player=1,
            counter=counter,
        )
        sfc = SFCGenerator(sfc_dict).generate()

        # pré-popula instâncias de um player anterior (reuso) e ajusta lista de servidores
        servers = [n for n, d in graph.nodes(data=True)
                   if d.get("type") == "server" and not str(n).endswith(".1")]
        if servers:
            graph = _prepopulate_prior_player(graph, sfc, counter, servers)

        list_graph.append(graph)
        list_sfc.append(sfc)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / "list_graph_het.pkl").write_bytes(pickle.dumps(list_graph))
    (SAVE_DIR / "list_sfc_het.pkl").write_bytes(pickle.dumps(list_sfc))

    print(f"Dataset salvo em {SAVE_DIR}: {len(list_graph)} amostras")
    print(f"Distribuição por serviço: {counts}")


if __name__ == "__main__":
    main()
