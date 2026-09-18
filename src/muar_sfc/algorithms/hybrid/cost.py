"""Avaliador de custos híbrido (fusão dos três artigos).

- Custo de banda por saltos extras (Wu et al.): C_b(v) = Σ b_l · (hop - 1)
- Custo operacional + desgaste de ligar/desligar (Pham et al.):
    E(M) = α · Σ Q_m + β · |M - M'|
- Latência tripartida (Zhao et al.): T_comm + T_ins + T_proc, onde o
    agendamento contínuo α divide o custo de processamento (mais α = mais
    rápido).

Todas as funções são puras e reutilizam os utilitários de rede existentes
(``networkUtils``) e o modelo de potência do ``EnergyCalculator`` do projeto.
"""

from __future__ import annotations

from collections.abc import Iterable

from muar_sfc.algorithms.networkUtils import (
    calculate_computational_latency,
    calculate_latency_betwen_nodes,
)
from muar_sfc.utils.network_utils import EnergyCalculator

DEFAULT_NODE_POWER = 250.0


def compute_node_powers(graph, node_ids: Iterable | None = None) -> dict:
    """Calcula a potência (Watts) de cada nó usando o EnergyCalculator do projeto.

    O EnergyCalculator deriva a potência do ``level_server`` (a/b/c) e da
    utilização de CPU (interpolando entre os níveis low/medium/high). Nós sem
    o atributo caem no valor padrão.
    """
    calculator = EnergyCalculator()
    powers: dict = {}
    allowed = set(node_ids) if node_ids is not None else None
    for node_id in graph.nodes:
        if allowed is not None and node_id not in allowed:
            continue
        try:
            powers[node_id] = calculator._get_power_for_node(graph, node_id)
        except Exception:
            powers[node_id] = DEFAULT_NODE_POWER
    return powers


class HybridSFCCostEvaluator:
    """Avalia uma alocação de SFC com o custo unificado dos três artigos."""

    def __init__(
        self,
        weight_b: float = 0.3,
        weight_c: float = 0.3,
        weight_l: float = 0.4,
        alpha_power: float = 0.1,
        beta_wear: float = 0.1,
        success_bonus: float = 3.0,
        failure_penalty: float = -3.0,
        latency_target_ms: float = 15.0,
        sla_latency_ms: float = 20.0,
    ):
        self.wb = weight_b
        self.wc = weight_c
        self.wl = weight_l
        self.alpha_power = alpha_power
        self.beta_wear = beta_wear
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.latency_target_ms = latency_target_ms
        self.sla_latency_ms = sla_latency_ms

    def compute_bandwidth_cost(self, path: list, req_bw: float) -> float:
        """Custo de banda acumulado por saltos extras (Wu et al.).

        ``path`` é o caminho físico (já filtrado por banda) entre duas VNFs
        consecutivas. Cada salto além do primeiro (mínimo) é penalizado com a
        banda requisitada.
        """
        if not path or len(path) < 2:
            return 0.0
        hops = len(path) - 1
        return req_bw * max(0.0, hops - 1)

    def compute_operational_cost(
        self,
        active_nodes: set,
        prev_active_nodes: set,
        node_powers: dict,
    ) -> float:
        """Energia ativa + desgaste de transição (Pham et al.).

        ``E(M) = α · Σ_{m∈M} Q_m + β · |M − M'|``
        """
        power_cost = sum(node_powers.get(m, DEFAULT_NODE_POWER) for m in active_nodes)
        wear_cost = len(set(active_nodes).symmetric_difference(prev_active_nodes))
        return self.alpha_power * power_cost + self.beta_wear * wear_cost

    def compute_latency(self, graph, path: list, vnf, alpha: float, node) -> float:
        """Latência tripartida de comunicação, instanciação e processamento.

        - T_comm: soma das latências dos enlaces do caminho físico.
        - T_ins: atributo ``inst_time`` do nó (default 0.002 ms).
        - T_proc: latência computacional do projeto (``networkUtils``) dividida
          pela fração α — quanto menor o agendamento, maior o atraso.
        """
        t_comm = 0.0
        if path and len(path) > 1:
            for u, v in zip(path[:-1], path[1:], strict=False):
                t_comm += calculate_latency_betwen_nodes(graph, u, v, vnf)

        node_data = graph.nodes[node]
        t_ins = float(node_data.get("inst_time", 0.002))

        t_proc = calculate_computational_latency(graph, node, vnf)
        t_proc = t_proc / max(float(alpha), 1e-3)

        return t_comm + t_ins + t_proc

    def calculate_reward(
        self,
        c_banda: float,
        c_operacional: float,
        lat: float,
        sla_latency: float,
    ) -> tuple[float, dict[str, float]]:
        """Recompensa unificada com latência realista e penalidade de SLA.

        Componentes calibrados para faixa [-5 .. +6] por passo:
        - Banda: 2/(C_b+1) ∈ [0, 2] (Wu: minimiza hops)
        - Op. cost: 1/(C_op+1) ∈ (0, 1] (Pham: minimiza energia)
        - Latência: gradiente forte em torno do target (Zhao)
        - SLA: penalidade pesada se exceder o prazo efetivo (20ms)
        - Sucesso: +3, Falha: -3 (reduzido para não dominar)
        """
        target = self.latency_target_ms
        hard_sla = float(sla_latency) if sla_latency is not None else self.sla_latency_ms

        # Bandas e operacional: mesmos dos artigos originais
        r_bw = 2.0 / (c_banda + 1.0)
        r_op = 1.0 / (c_operacional + 1.0)

        # Latência: termo com gradiente real em torno do target
        r_lat = (
            2.0 * (1.0 - lat / max(target, 1e-3))
            if lat <= target
            else -0.3 * (lat - target)
        )

        # SLA hard: penalidade pesada se exceder o prazo máximo
        sla_penalty = 0.0
        if lat > hard_sla:
            sla_penalty = 1.5 * (lat - hard_sla)

        # Penalidade por consumo excessivo de recursos
        resource_penalty = 0.05 * c_operacional

        reward = (
            self.wb * r_bw
            + self.wc * r_op
            + self.wl * r_lat
            - sla_penalty
            - resource_penalty
        )

        metrics = {
            "bandwidth_cost": c_banda,
            "operational_cost": c_operacional,
            "latency": lat,
            "r_bw": r_bw,
            "r_op": r_op,
            "r_lat": r_lat,
            "sla_penalty": sla_penalty,
            "resource_penalty": resource_penalty,
            "reward": reward,
        }
        return reward, metrics
