"""Arquitetura neural Multi-Head (GCN + Seq2Seq + saídas Categorical e Beta).

- GCN: extração topológica das features dos nós (Wu et al.).
- LSTM: codificação da sequência de VNFs da SFC (Wu et al.).
- Cabeças de saída (Zhao et al.):
    * ``node_logits``  -> distribuição Categorical sobre os nós candidatos;
    * ``beta_params``  -> distribuição Beta para a fração contínua α;
    * ``value``        -> estimador de valor (critic) do PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Camada de propagação convolucional em grafo com normalização por grau."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = torch.sum(adj, dim=-1, keepdim=True) + 1e-5
        adj_norm = adj / deg
        out = torch.matmul(adj_norm, x)
        return F.relu(self.fc(out))


class HybridSFCNetwork(nn.Module):
    """Política/valor híbrido insensível ao número de nós (variável por topologia)."""

    def __init__(self, node_feat_dim: int, vnf_feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.vnf_feat_dim = vnf_feat_dim
        self.hidden_dim = hidden_dim

        self.gcn = GCNLayer(node_feat_dim, hidden_dim)
        self.lstm = nn.LSTM(vnf_feat_dim, hidden_dim, batch_first=True)

        self.discrete_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        adj_matrix: torch.Tensor,
        sfc_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (node_logits, beta_params, value).

        - ``node_feats``: [B, N, F]
        - ``adj_matrix``: [B, N, N]
        - ``sfc_seq``:    [B, T, V]
        """
        g_embed = self.gcn(node_feats, adj_matrix)

        _, (h_n, _) = self.lstm(sfc_seq)
        sfc_embed = h_n[-1].unsqueeze(1).expand(-1, g_embed.shape[1], -1)

        context = torch.cat([g_embed, sfc_embed], dim=-1)

        node_logits = self.discrete_head(context).squeeze(-1)
        beta_params = F.softplus(self.beta_head(context)) + 1.0
        value = self.value_head(context.mean(dim=1))

        return node_logits, beta_params, value
