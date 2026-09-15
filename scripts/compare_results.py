import glob

import numpy as np
import pandas as pd

RESULTS = "results/results_flows"
algos = {
    "hybrid_s_50_p_6_a_0.99_c_3": "Hybrid",
    "vegeta_s_50_p_6_a_0.99_c_3": "VEGETA",
    "greedyb_s_50_p_6_a_0.99_c_3": "GreedyB",
    "musfico_s_50_p_6_a_0.99_c_3": "MUSFiCO",
    "msf_s_50_p_6_a_0.99_c_3": "MSF",
}

cols = [
    "acceptance_rate", "latency", "cpu_utilization", "cache_utilization",
    "bandwidth_utilization", "avg_sfc_reliability", "total_energy_consumption",
]


def mean_last(df, col):
    v = df[col].dropna()
    if len(v) == 0:
        return 0.0
    # média da última metade do run (estado estacionário)
    return v.iloc[len(v) // 2:].mean()


print(f"{'Algo':10s} | {'Aceit%':>7s} | {'Lat(ms)':>7s} | {'CPU':>6s} | {'Cache':>6s} | {'Banda':>6s} | {'Conf':>6s} | {'Pot(W)':>7s}")
for key, name in algos.items():
    csvs = sorted(glob.glob(f"{RESULTS}/{key}/*.csv"))
    csvs = [c for c in csvs if "service_summary" not in c]
    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p, sep=",")
            if len(df) > 10:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        print(f"{name}: NO DATA")
        continue
    m = {}
    for col in cols:
        m[col] = np.mean([mean_last(f, col) for f in frames])
    print(
        f"{name:10s} | {m['acceptance_rate']:6.1f}% | {m['latency']:6.2f} | "
        f"{m['cpu_utilization']:.4f} | {m['cache_utilization']:.4f} | "
        f"{m['bandwidth_utilization']:.4f} | {m['avg_sfc_reliability']:.4f} | "
        f"{m['total_energy_consumption']:6.0f}"
    )