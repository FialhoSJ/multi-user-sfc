"""Run a paired experiment: one trace/topology seed for every algorithm."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from muar_sfc.config import SimulationSettings
from muar_sfc.topology.instantiator import TopologyInstantiator
from muar_sfc.utils.fair_experiment import build_request_trace, save_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", default=["hybrid", "msf", "musfico", "vegeta", "greedyb"])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-file", type=Path, default=Path("results/traces/fair_trace.json"))
    parser.add_argument("--n-sessions", type=int, default=50)
    parser.add_argument("--n-players", type=int, default=6)
    parser.add_argument("--topology", default="luxembourgv2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = SimulationSettings(
        n_sessions=args.n_sessions, n_players=args.n_players,
        topology=args.topology, seed=args.seed,
    )
    topology = TopologyInstantiator().instantiate_topology(settings.topology, settings.eco_effi_ratio)
    save_trace(build_request_trace(settings, topology, args.seed), args.trace_file)

    jobs = []
    for repetition in range(args.repetitions):
        run_seed = args.seed + repetition
        for algorithm in args.algorithms:
            jobs.append([
                sys.executable, "-m", "muar_sfc.main",
                "--alg", algorithm, "--topology", args.topology,
                "--n_sessions", str(args.n_sessions), "--n_players", str(args.n_players),
                "--seed", str(run_seed), "--repetition", str(repetition),
                "--trace_file", str(args.trace_file), "--alpha", str(args.alpha),
                "--verbose", "n",
            ])
    print(f"Trace comum: {args.trace_file} | jobs pareados: {len(jobs)}")
    if args.dry_run:
        for job in jobs:
            print(" ".join(job))
        return
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] algoritmo={job[4]} repetição={job[14]}")
        subprocess.run(job, check=True)


if __name__ == "__main__":
    main()
