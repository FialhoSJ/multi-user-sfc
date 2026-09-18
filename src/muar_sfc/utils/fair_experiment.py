"""Primitives for reproducible, paired algorithm experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the simulator and common ML backends."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_request_trace(settings: Any, topology: Any, seed: int) -> dict[str, Any]:
    """Create one immutable workload trace shared by all algorithms."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    services = [s.strip() for s in settings.service_mix.split(",") if s.strip()]
    weights = [float(w.strip()) for w in settings.service_weights.split(",") if w.strip()]
    routers = list(topology.get_topology_info()["routers"])
    cumulative = np.cumsum(np.asarray(weights, dtype=float) / sum(weights))

    requests = []
    for index in range(1, settings.n_sessions + 1):
        x = rng.random()
        service = services[int(np.searchsorted(cumulative, x, side="right"))]
        requests.append(
            {
                "request_index": index,
                "service_type": service,
                "closer_router": routers[rng.randrange(len(routers))],
                "duration": int(np_rng.poisson(settings.sfc_lifetime)),
            }
        )
    return {
        "schema_version": 1,
        "seed": seed,
        "topology": settings.topology,
        "n_sessions": settings.n_sessions,
        "n_players": settings.n_players,
        "service_mix": settings.service_mix,
        "service_weights": settings.service_weights,
        "requests": requests,
    }


def save_trace(trace: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_trace(path: str | Path) -> dict[str, Any]:
    trace = json.loads(Path(path).read_text(encoding="utf-8"))
    if trace.get("schema_version") != 1 or not isinstance(trace.get("requests"), list):
        raise ValueError(f"Trace inválido ou incompatível: {path}")
    return trace
