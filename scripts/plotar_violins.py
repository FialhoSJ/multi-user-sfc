"""Violin plots por faixa de tempo (comparação de algoritmos).

Representação baseada em análise exploratória de dados: um violino por
algoritmo em cada faixa de tempo (200s, 400s, ..., 1000s) mostra a forma
completa da distribuição (assimetria, multimodalidade), com mediana (linha
cheia), média (linha tracejada) e extremos — mais informativo que box plots.

Outliers do decision_time são limitados ao "bigode 6x" (mesma lógica do
notebook plots/2025/davidgn) para não dominar o eixo Y.

Uso:
    uv run python scripts/plotar_violins.py --out DIR \
        "REPLIC=results/results_flows/replic_s_50_p_4_a_0.99_c_3/FLOWS.csv" \
        "MSF=results/results_flows/msf_s_50_p_4_a_0.99_c_3/FLOWS.csv"
"""
from __future__ import annotations

import argparse
import csv
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    from scripts.cores_algoritmos import cor_para, rotulo_curto
except ModuleNotFoundError:
    from cores_algoritmos import cor_para, rotulo_curto

BUCKET = 200.0
CAP = 1000.0

# (coluna, rótulo Y, nome do arquivo de saída, substituir_outliers?)
METRICS = [
    ("acceptance_rate", "Acceptance Rate (%)", "acceptance_rate", False),
    ("latency", "Latency (ms)", "latency", False),
    ("decision_time_ms", "Decision Time (ms)", "decision_time", True),
    ("shared_vnfs", "Shared SFs", "shared_sfs", False),
    ("bandwidth_utilization", "Bandwidth Utilization (%)", "uso_banda", False),
    ("cpu_utilization", "CPU Utilization (%)", "uso_cpu", False),
]


def load_csv(path: str) -> dict[str, list[str]]:
    cols: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for col in reader.fieldnames or []:
            cols[col] = []
        for row in reader:
            for col in reader.fieldnames or []:
                cols[col].append(row[col])
    return cols


def bucket_of(t: str) -> float:
    """Mapeia um tempo (s) para a faixa de 200s correspondente (teto), cap em 1000."""
    t = float(t)
    b = math.ceil(t / BUCKET) * BUCKET
    return min(CAP, b)


def substituir_outliers_por_bigode_6x(values: list[str]) -> np.ndarray:
    """Mesma lógica de tratamento_csv.py: limita outliers em 6x o bigode superior."""
    dados = np.array([float(v) for v in values])
    q1, q3 = np.percentile(dados, [25, 75])
    iqr = q3 - q1
    limite_superior = q3 + 1.5 * iqr
    dentro = dados[dados <= limite_superior]
    bigode_superior = float(dentro.max()) if len(dentro) > 0 else float(limite_superior)
    limite = 6 * bigode_superior
    return np.where(dados > limite, limite, dados)


def to_float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None or str(v).strip() in ("", "None", "nan", "N/A"):
            continue
        try:
            out.append(float(v))
        except ValueError:
            continue
    return out


def parse_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        label, _, path = item.partition("=")
        pairs.append((label.strip(), path.strip()))
    return pairs


def _style_line(parts: dict, key: str, color: str, lw: float, ls: str | None = None) -> None:
    """Estiliza uma parte do violino (LineCollection ou lista de Line2D)."""
    item = parts[key]
    targets = [item] if isinstance(item, LineCollection) else item
    for t in targets:
        t.set_color(color)
        t.set_linewidth(lw)
        if ls is not None:
            t.set_linestyle(ls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="diretório de saída dos PNGs")
    parser.add_argument("--scenario", default="", help="sufixo opcional no nome dos arquivos")
    parser.add_argument("--points", action="store_true", default=True,
                        help="sobrepor pontos (strip jittered) sobre os violinos")
    parser.add_argument("--no-points", action="store_false", dest="points",
                        help="não sobrepor pontos")
    parser.add_argument("items", nargs="+", help='"LABEL=path/to/flows.csv" ...')
    args = parser.parse_args()

    pairs = parse_pairs(args.items)
    try:
        matplotlib.font_manager.findfont("Arial")
        plt.rcParams["font.family"] = ["Arial"]
    except Exception:
        plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.titleweight"] = "normal"

    data_by_metric: dict[str, dict[str, dict[float, list[float]]]] = {
        metric: {label: {} for label, _ in pairs} for metric, *_ in METRICS
    }

    for label, path in pairs:
        cols = load_csv(path)
        time_vals = [bucket_of(t) for t in cols["time_seconds"]]
        for metric, _y, _f, use_outlier in METRICS:
            raw = to_float_list(cols[metric])
            if use_outlier:
                raw = [float(v) for v in substituir_outliers_por_bigode_6x([str(v) for v in raw])]
            bucket_data = data_by_metric[metric][label]
            for b, v in zip(time_vals, raw, strict=False):
                bucket_data.setdefault(b, []).append(v)

    import os

    os.makedirs(args.out, exist_ok=True)
    colors = [cor_para(label, j) for j, (label, _p) in enumerate(pairs)]
    rng = np.random.default_rng(42)

    for metric, ylabel, fname, _use_outlier in METRICS:
        buckets = sorted({b for d in data_by_metric[metric].values() for b in d})
        n = len(pairs)
        step = n + 1
        fig, ax = plt.subplots(figsize=(14, 7))
        for i, b in enumerate(buckets):
            base = i * step
            for j, (label, _p) in enumerate(pairs):
                values = data_by_metric[metric][label].get(b, [])
                pos = base + j
                if not values:
                    continue
                parts = ax.violinplot(
                    [values],
                    positions=[pos],
                    widths=0.82,
                    showmeans=True,
                    showmedians=True,
                    showextrema=True,
                )
                body = parts["bodies"][0]
                body.set_facecolor(colors[j])
                body.set_alpha(0.45)
                body.set_edgecolor(colors[j])
                body.set_linewidth(1.1)
                _style_line(parts, "cmins", "black", 0.8)
                _style_line(parts, "cmaxes", "black", 0.8)
                _style_line(parts, "cmedians", "black", 1.4)
                _style_line(parts, "cmeans", "black", 1.1, ls="--")
                if args.points:
                    xs = pos + rng.uniform(-0.18, 0.18, size=len(values))
                    ax.scatter(xs, values, s=4, alpha=0.35, color=colors[j],
                               linewidths=0, zorder=3)
        ax.set_xticks([i * step + (n - 1) / 2 for i in range(len(buckets))])
        ax.set_xticklabels([f"{int(b)}" for b in buckets], fontsize=13)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xlabel("Time (s)", fontsize=15)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[j], alpha=0.45,
                          edgecolor=colors[j], label=rotulo_curto(label))
            for j, (label, _p) in enumerate(pairs)
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  ncol=min(n, 6), fontsize=14, frameon=False)
        out_name = fname if not args.scenario else f"{fname}_{args.scenario}"
        out_path = os.path.join(args.out, f"{out_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"salvo: {out_path}")


if __name__ == "__main__":
    main()
