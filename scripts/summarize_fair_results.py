"""Aggregate per-run CSVs with mean, sample SD and 95% CI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("success", "latency", "decision_time_ms", "path_bandwidth_cost", "path_hops")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, default=Path("results/results_flows"), nargs="?")
    parser.add_argument("--output", type=Path, default=Path("results/fair_summary.csv"))
    args = parser.parse_args()

    run_values = []
    for csv_path in args.root.glob("**/*.csv"):
        if "service_summary" in csv_path.name:
            continue
        # A pasta contém execuções históricas. No protocolo justo, cada CSV
        # precisa ter o metadata JSON com o mesmo stem; assim não misturamos
        # resultados antigos ou execuções feitas com outro workload.
        meta_path = csv_path.parent / f"experiment_metadata_{csv_path.stem}.json"
        if not meta_path.exists():
            continue
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            continue
        present = [metric for metric in METRICS if metric in frame]
        if not present:
            continue
        meta = {}
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for metric in present:
            values = pd.to_numeric(frame[metric], errors="coerce").dropna()
            if values.empty:
                continue
            # Cada CSV é uma repetição. Primeiro reduzimos os fluxos daquela
            # execução a uma observação; o IC final usa a quantidade de runs.
            run_values.append({
                "algorithm": meta.get("algorithm", csv_path.parent.name.split("_s_")[0]),
                "seed": meta.get("seed"), "repetition": meta.get("repetition"),
                "metric": metric, "run_mean": values.mean(),
            })
    rows = []
    if run_values:
        runs = pd.DataFrame(run_values)
        for (algorithm, metric), group in runs.groupby(["algorithm", "metric"], dropna=False):
            values = group["run_mean"]
            n = len(values)
            mean = values.mean()
            sd = values.std(ddof=1) if n > 1 else 0.0
            rows.append({
                "algorithm": algorithm, "metric": metric, "n_repetitions": n,
                "mean": mean, "std": sd,
                "ci95_low": mean - 1.96 * sd / (n ** 0.5),
                "ci95_high": mean + 1.96 * sd / (n ** 0.5),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Resumo salvo em {args.output} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
