"""PPO custom em PyTorch para o agente híbrido de SFC.

Implementa coleta de trajetórias, Generalized Advantage Estimation (GAE),
atualizações com clipping e entropia das duas cabeças (Categorical e Beta).
A API espelha o fluxo do stable-baselines3 usado no restante do projeto
(``predict``, ``learn``, save/load via checkpoint), facilitando o treino e a
inferência dentro do MUAR-SFC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical
from torch.utils.tensorboard import SummaryWriter

from muar_sfc.algorithms.hybrid.networks import HybridSFCNetwork


def _to_tensor(value, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=dtype)


class HybridPPO:
    """PPO com cabeça discreta (Categorical) + contínua (Beta) e critic MLP."""

    def __init__(
        self,
        network: HybridSFCNetwork,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.05,
        beta_entropy_coef: float = 0.01,
        epochs: int = 10,
        minibatch_size: int = 128,
        rollout_len: int = 2048,
        device: str = "cpu",
    ):
        self.network = network.to(device)
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.beta_entropy_coef = beta_entropy_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.rollout_len = rollout_len

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Amostragem de ações (inferência)
    # ------------------------------------------------------------------

    def _obs_to_tensors(self, obs: dict):
        node_feats = _to_tensor(obs["node_feats"]).unsqueeze(0).to(self.device)
        adj = _to_tensor(obs["adj_matrix"]).unsqueeze(0).to(self.device)
        sfc_seq = _to_tensor(obs["sfc_seq"]).unsqueeze(0).to(self.device)
        return node_feats, adj, sfc_seq

    def _dist_from_obs(self, obs: dict, masks: np.ndarray):
        node_feats, adj, sfc_seq = self._obs_to_tensors(obs)
        node_logits, beta_params, value = self.network(node_feats, adj, sfc_seq)
        node_logits = self._apply_mask(node_logits, masks)
        return node_logits, beta_params, value

    @staticmethod
    def _apply_mask(node_logits: torch.Tensor, masks) -> torch.Tensor:
        mask_t = _to_tensor(masks).bool().to(node_logits.device).unsqueeze(0)
        if mask_t.sum() == 0:
            mask_t[0, -1] = True
        return node_logits.masked_fill(~mask_t, float("-inf"))

    def _sample_action(self, obs: dict, masks: np.ndarray, deterministic: bool = False):
        node_logits, beta_params, value = self._dist_from_obs(obs, masks)
        node_dist = Categorical(logits=node_logits)

        node_idx = node_dist.probs.argmax(dim=-1) if deterministic else node_dist.sample()

        selected_beta = beta_params.gather(
            1, node_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)
        ).squeeze(1)
        alpha_dist = Beta(selected_beta[..., 0], selected_beta[..., 1])

        alpha = alpha_dist.mean if deterministic else alpha_dist.sample()
        alpha = alpha.clamp(0.01, 1.0)

        node_logp = node_dist.log_prob(node_idx)
        alpha_logp = alpha_dist.log_prob(alpha.clamp(1e-6, 1.0 - 1e-6))
        logp = node_logp + alpha_logp

        return node_idx.item(), alpha.item(), logp, value

    def predict(self, obs: dict, masks: np.ndarray) -> tuple[int, float]:
        """Ação determinística (nó, alpha) para o loop de inferência."""
        with torch.no_grad():
            node_idx, alpha, _, _ = self._sample_action(obs, masks, deterministic=True)
        return node_idx, alpha

    # ------------------------------------------------------------------
    # Coleta de trajetórias
    # ------------------------------------------------------------------

    def _collect_rollout(self, env, n_steps: int):
        obs, _ = env.reset()
        buffer = {
            "obs": [],
            "node_idxs": [],
            "alphas": [],
            "logp": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "masks": [],
        }
        running_rewards: list[float] = []
        ep_rewards: list[float] = []
        ep_latency: list[float] = []
        fail_counts: dict[str, int] = {"resource": 0, "bandwidth": 0, "no_mask": 0}
        steps = 0
        episode_continues = False

        while steps < n_steps:
            masks = env.action_masks()

            if masks.sum() == 0:
                ep_rewards.append(0.0)
                ep_latency.append(0.0)
                fail_counts["no_mask"] += 1
                obs, _ = env.reset()
                running_rewards = []
                episode_continues = False
                continue

            node_idx, alpha, logp, value = self._sample_action(obs, masks)
            next_obs, reward, terminated, truncated, _ = env.step((node_idx, alpha))
            done = terminated or truncated

            buffer["obs"].append(obs)
            buffer["node_idxs"].append(node_idx)
            buffer["alphas"].append(alpha)
            buffer["logp"].append(logp)
            buffer["values"].append(value)
            buffer["rewards"].append(float(reward))
            buffer["dones"].append(done)
            buffer["masks"].append(np.asarray(masks, dtype=np.float32))

            steps += 1
            running_rewards.append(float(reward))

            if done:
                ep_rewards.append(sum(running_rewards))
                ep_latency.append(getattr(env, "latency_used", 0.0))
                reason = getattr(env, "fail_reason", None)
                if reason:
                    fail_counts[reason] = fail_counts.get(reason, 0) + 1
                obs, _ = env.reset()
                running_rewards = []
                episode_continues = False
            else:
                obs = next_obs
                episode_continues = True

        if running_rewards:
            ep_rewards.append(sum(running_rewards))
            ep_latency.append(getattr(env, "latency_used", 0.0))

        if episode_continues and obs is not None:
            with torch.no_grad():
                last_value = self.network(*self._obs_to_tensors(obs))[2].detach().squeeze()
            buffer["last_value"] = last_value
        else:
            buffer["last_value"] = torch.zeros(1)

        stats = {
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "mean_latency": float(np.mean(ep_latency)) if ep_latency else 0.0,
            "fail_reasons": fail_counts,
        }
        return buffer, stats

    # ------------------------------------------------------------------
    # GAE e atualização PPO
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, lam = self.gamma, self.gae_lambda
        n = len(rewards)
        returns = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value if not dones[t] else values.new_zeros(1)
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
            gae = delta + gamma * lam * (1 - dones[t].float()) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        return returns, advantages

    def _evaluate(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        sfc_seq: torch.Tensor,
        node_idxs: torch.Tensor,
        alphas: torch.Tensor,
        masks: torch.Tensor,
    ):
        node_logits, beta_params, values = self.network(node_feats, adj, sfc_seq)
        node_logits = node_logits.masked_fill(~masks.bool(), float("-inf"))
        node_dist = Categorical(logits=node_logits)

        node_logp = node_dist.log_prob(node_idxs)
        node_entropy = node_dist.entropy()

        selected_beta = beta_params.gather(
            1, node_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2)
        ).squeeze(1)
        alpha_dist = Beta(selected_beta[..., 0], selected_beta[..., 1])
        alpha_logp = alpha_dist.log_prob(alphas.clamp(1e-6, 1.0 - 1e-6))
        beta_entropy = alpha_dist.entropy()

        return (
            node_logp + alpha_logp,
            values.squeeze(-1),
            node_entropy,
            beta_entropy,
        )

    def update(self, buffer: dict) -> dict[str, float]:
        node_feats = torch.as_tensor(
            np.stack([o["node_feats"] for o in buffer["obs"]]), dtype=torch.float32
        ).to(self.device)
        adj = torch.as_tensor(
            np.stack([o["adj_matrix"] for o in buffer["obs"]]), dtype=torch.float32
        ).to(self.device)
        sfc_seq = torch.as_tensor(
            np.stack([o["sfc_seq"] for o in buffer["obs"]]), dtype=torch.float32
        ).to(self.device)
        node_idxs = torch.as_tensor(buffer["node_idxs"], dtype=torch.long).to(self.device)
        alphas = torch.as_tensor(buffer["alphas"], dtype=torch.float32).to(self.device)
        old_logp = torch.cat(buffer["logp"]).detach().to(self.device)
        values = torch.cat(buffer["values"]).detach().squeeze().to(self.device)
        rewards = torch.as_tensor(buffer["rewards"], dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(buffer["dones"], dtype=torch.bool).to(self.device)
        masks = torch.as_tensor(
            np.stack(buffer["masks"]), dtype=torch.bool
        ).to(self.device)

        last_value = buffer.get("last_value", torch.zeros(1)).detach().to(self.device)
        returns, advantages = self._compute_gae(rewards, values, dones, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(buffer["rewards"]))
        loss_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                mb = indices[start : start + self.minibatch_size]
                mb = torch.as_tensor(mb, dtype=torch.long).to(self.device)

                logp, val, node_entropy, beta_entropy = self._evaluate(
                    node_feats[mb],
                    adj[mb],
                    sfc_seq[mb],
                    node_idxs[mb],
                    alphas[mb],
                    masks[mb],
                )
                ratio = torch.exp(logp - old_logp[mb])
                adv = advantages[mb]

                pg_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv,
                ).mean()
                vf_loss = F.mse_loss(val, returns[mb])
                loss = (
                    pg_loss
                    + self.value_coef * vf_loss
                    - self.entropy_coef * node_entropy.mean()
                    - self.beta_entropy_coef * beta_entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                loss_stats["policy_loss"] += pg_loss.item()
                loss_stats["value_loss"] += vf_loss.item()
                loss_stats["entropy"] += node_entropy.mean().item() + beta_entropy.mean().item()

        n_updates = self.epochs * max(1, len(indices) // self.minibatch_size)
        for key in loss_stats:
            loss_stats[key] /= max(1, n_updates)

        self._step_count += len(buffer["rewards"])
        return loss_stats

    # ------------------------------------------------------------------
    # Loop de treinamento
    # ------------------------------------------------------------------

    def evaluate(self, env, n_episodes: int = 30) -> dict[str, float]:
        successes = 0
        total_reward = 0.0
        total_latency = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                masks = env.action_masks()
                node_idx, alpha = self.predict(obs, masks)
                obs, reward, terminated, truncated, _ = env.step((node_idx, alpha))
                done = terminated or truncated
                ep_reward += reward
            total_reward += ep_reward
            if env.success:
                successes += 1
                total_latency += env.latency_used
        return {
            "success_rate": successes / max(1, n_episodes),
            "mean_reward": total_reward / max(1, n_episodes),
            "mean_latency": total_latency / max(1, successes) if successes else 0.0,
        }

    def learn(
        self,
        env,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 4096,
        eval_episodes: int = 30,
        save_path: str | None = None,
        tb_log_dir: str = "tensorboard_logs",
        tb_log_name: str = "HybridSFC",
    ) -> dict[str, float]:
        writer = SummaryWriter(log_dir=f"{tb_log_dir}/{tb_log_name}")
        save_path = Path(save_path) if save_path else None
        steps = 0
        final_stats = {}

        while steps < total_timesteps:
            n = min(self.rollout_len, total_timesteps - steps)
            buffer, rollout_stats = self._collect_rollout(env, n)
            loss_stats = self.update(buffer)
            steps += n

            writer.add_scalar("train/reward", rollout_stats["mean_reward"], steps)
            writer.add_scalar("train/policy_loss", loss_stats["policy_loss"], steps)
            writer.add_scalar("train/value_loss", loss_stats["value_loss"], steps)
            writer.add_scalar("train/entropy", loss_stats["entropy"], steps)
            for reason, count in rollout_stats.get("fail_reasons", {}).items():
                writer.add_scalar(f"train/fail_{reason}", count, steps)
            final_stats = loss_stats

            if eval_env is not None and steps % eval_freq < n:
                eval_stats = self.evaluate(eval_env, eval_episodes)
                writer.add_scalar("eval/success_rate", eval_stats["success_rate"], steps)
                writer.add_scalar("eval/mean_reward", eval_stats["mean_reward"], steps)
                writer.add_scalar("eval/mean_latency", eval_stats["mean_latency"], steps)
                print(
                    f"[{steps}/{total_timesteps}] "
                    f"sucesso={eval_stats['success_rate']:.2f} "
                    f"recompensa={eval_stats['mean_reward']:.2f} "
                    f"latência={eval_stats['mean_latency']:.2f}"
                )

            if save_path is not None:
                self.save(save_path)

        writer.close()
        return final_stats

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.network.state_dict(),
            "node_feat_dim": self.network.node_feat_dim,
            "vnf_feat_dim": self.network.vnf_feat_dim,
            "hidden_dim": self.network.hidden_dim,
            "hyperparams": {
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "beta_entropy_coef": self.beta_entropy_coef,
            },
        }
        torch.save(checkpoint, str(path))

    @classmethod
    def load(cls, path, device: str = "cpu") -> HybridPPO:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        network = HybridSFCNetwork(
            node_feat_dim=checkpoint["node_feat_dim"],
            vnf_feat_dim=checkpoint["vnf_feat_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 64),
        )
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        hyper = checkpoint.get("hyperparams", {})
        ppo = cls(network=network, device=device, **hyper)
        return ppo
