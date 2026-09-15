import numpy as np
import torch

from muar_sfc.algorithms.hybrid.networks import GCNLayer, HybridSFCNetwork
from muar_sfc.algorithms.hybrid.ppo import HybridPPO


def _obs(n_nodes: int = 4, t_steps: int = 8):
    return {
        "node_feats": np.zeros((n_nodes, 8), dtype=np.float32),
        "adj_matrix": np.eye(n_nodes, dtype=np.float32),
        "sfc_seq": np.zeros((t_steps, 5), dtype=np.float32),
        "sfc_seq_len": np.array([2], dtype=np.float32),
    }


def test_gcn_output_shape():
    layer = GCNLayer(8, 16)
    x = torch.randn(1, 5, 8)
    adj = torch.eye(5).unsqueeze(0)
    out = layer(x, adj)
    assert out.shape == (1, 5, 16)


def test_network_forward_shapes():
    net = HybridSFCNetwork(node_feat_dim=8, vnf_feat_dim=5, hidden_dim=16)
    n_nodes, t_steps = 4, 8
    node_feats = torch.randn(1, n_nodes, 8)
    adj = torch.eye(n_nodes).unsqueeze(0)
    sfc_seq = torch.randn(1, t_steps, 5)

    node_logits, beta_params, value = net(node_feats, adj, sfc_seq)

    assert node_logits.shape == (1, n_nodes)
    assert beta_params.shape == (1, n_nodes, 2)
    assert value.shape == (1, 1)
    assert (beta_params >= 1.0).all()
    assert torch.isfinite(node_logits).all()


def test_ppo_predict_returns_hybrid_action():
    net = HybridSFCNetwork(node_feat_dim=8, vnf_feat_dim=5, hidden_dim=16)
    ppo = HybridPPO(network=net, lr=1e-3)

    masks = np.ones(4, dtype=np.int8)
    node_idx, alpha = ppo.predict(_obs(), masks)

    assert isinstance(node_idx, int)
    assert 0 <= node_idx < 4
    assert isinstance(alpha, float)
    assert 0.01 <= alpha <= 1.0


def test_ppo_evaluate_logp_finite():
    net = HybridSFCNetwork(node_feat_dim=8, vnf_feat_dim=5, hidden_dim=16)
    ppo = HybridPPO(network=net, lr=1e-3, minibatch_size=4, epochs=1)

    b, n, t = 4, 5, 8
    node_feats = torch.randn(b, n, 8)
    adj = torch.eye(n).unsqueeze(0).expand(b, n, n).clone()
    sfc_seq = torch.randn(b, t, 5)
    masks = torch.ones(b, n, dtype=torch.bool)
    node_idxs = torch.randint(0, n, (b,))
    alphas = torch.rand(b)

    logp, val, node_ent, beta_ent = ppo._evaluate(
        node_feats, adj, sfc_seq, node_idxs, alphas, masks
    )
    assert torch.isfinite(logp).all()
    assert torch.isfinite(val).all()
    assert (node_ent >= 0).all()
    assert torch.isfinite(beta_ent).all()  # entropia da Beta pode ser negativa


class _FakeEnv:
    """Ambiente mínimo que espelha a interface usada pelo PPO."""

    def __init__(self, n_nodes=4, max_steps=3):
        self.n_nodes = n_nodes
        self.max_steps = max_steps
        self.success = False
        self.latency_used = 0.0

    def reset(self, seed=None, options=None):
        self.steps = 0
        self.success = False
        self.latency_used = 0.0
        return _obs(self.n_nodes), {}

    def action_masks(self):
        return np.ones(self.n_nodes, dtype=np.int8)

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.max_steps
        self.success = done
        self.latency_used += 0.1
        reward = 1.0 if done else 0.1
        return _obs(self.n_nodes), reward, done, False, {}


def test_ppo_collect_and_update_smoke():
    net = HybridSFCNetwork(node_feat_dim=8, vnf_feat_dim=5, hidden_dim=16)
    ppo = HybridPPO(network=net, lr=1e-3, rollout_len=12, minibatch_size=4, epochs=1)

    env = _FakeEnv()
    buffer, stats = ppo._collect_rollout(env, 12)
    assert len(buffer["rewards"]) == 12
    assert stats["mean_reward"] > 0.0

    loss_stats = ppo.update(buffer)
    for key, value in loss_stats.items():
        assert np.isfinite(value), key


def test_ppo_save_load_roundtrip(tmp_path):
    net = HybridSFCNetwork(node_feat_dim=8, vnf_feat_dim=5, hidden_dim=16)
    ppo = HybridPPO(network=net, lr=1e-3)
    path = str(tmp_path / "model.pt")
    ppo.save(path)

    loaded = HybridPPO.load(path, device="cpu")
    state_before = ppo.network.state_dict()
    state_after = loaded.network.state_dict()
    assert set(state_before) == set(state_after)
    assert all(
        torch.equal(state_before[k], state_after[k]) for k in state_before
    )
