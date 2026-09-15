"""Módulo híbrido: fusão dos artigos Pham (energia/desgaste), Wu (banda por saltos)
e Zhao (latência tripartida + agendamento contínuo) para alocação de SFCs.

Contém o avaliador de custos (cost.py), a arquitetura neural Multi-Head
(networks.py), o PPO custom (ppo.py) e o ambiente Gym híbrido (hybrid_env.py).
"""

from muar_sfc.algorithms.hybrid.cost import (
    DEFAULT_NODE_POWER,
    HybridSFCCostEvaluator,
    compute_node_powers,
)
from muar_sfc.algorithms.hybrid.hybrid_env import SFC_AllocationEnv_Hybrid
from muar_sfc.algorithms.hybrid.networks import GCNLayer, HybridSFCNetwork
from muar_sfc.algorithms.hybrid.ppo import HybridPPO

__all__ = [
    "DEFAULT_NODE_POWER",
    "HybridSFCCostEvaluator",
    "compute_node_powers",
    "SFC_AllocationEnv_Hybrid",
    "GCNLayer",
    "HybridSFCNetwork",
    "HybridPPO",
]
