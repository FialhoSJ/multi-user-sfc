"""Comparativo de aceitação por serviço entre os 6 mixes (pasta slides/).

Lê as tabelas `slides/mix*/tabela_resumo.csv` e gera:
  - slides/comparativo_mixes.png  (4 painéis, um por serviço)
  - slides/tabela_comparativo_mixes.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts.cores_algoritmos import cor_para, rotulo_curto
except ModuleNotFoundError:
    from cores_algoritmos import cor_para, rotulo_curto

SERVICOS = ["muar", "streaming", "voip", "iot"]
ALGOS = ["replic", "msf", "kuririn", "darsppo", "hephaestus"]
MIXES = [
    ("mix1_equilibrado", "Equil."),
    ("mix2_streaming", "Stream"),
    ("mix3_voip", "VoIP"),
    ("mix4_iot", "IoT"),
    ("mix5_muar", "MUAR"),
    ("mix6_sem_muar", "noMUAR"),
]


def ler_tabela(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    try:
        matplotlib.font_manager.findfont("Arial")
        plt.rcParams["font.family"] = ["Arial"]
    except Exception:
        plt.rcParams["font.family"] = ["DejaVu Sans"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="slides")
    args = parser.parse_args()
    root = Path(args.root)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    n_mixes = len(MIXES)
    largura = 0.8 / len(ALGOS)

    linhas = []
    for idx, svc in enumerate(SERVICOS):
        ax = axes[idx // 2][idx % 2]
        for k, algo in enumerate(ALGOS):
            valores = []
            for mix, _rotulo in MIXES:
                taxa = 0.0
                for row in ler_tabela(root / mix / "tabela_resumo.csv"):
                    if row["algoritmo"] == rotulo_curto(algo) and row["servico"] == svc:
                        taxa = float(row["aceitacao_%"])
                valores.append(taxa)
                linhas.append([mix, rotulo_curto(algo), svc, f"{taxa:.2f}"])
            pos = np.arange(n_mixes) + (k - (len(ALGOS) - 1) / 2) * largura
            ax.bar(pos, valores, width=largura, label=rotulo_curto(algo),
                   color=cor_para(algo, k), edgecolor="black", linewidth=0.8)
        ax.set_title(svc, fontsize=15, fontweight="bold")
        ax.set_xticks(range(n_mixes))
        ax.set_xticklabels([r for _, r in MIXES], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Aceitação (%)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=cor_para(a, k), edgecolor="black",
                      label=rotulo_curto(a))
        for k, a in enumerate(ALGOS)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Taxa de aceitação por serviço em cada mix", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(root / "comparativo_mixes.png", format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {root / 'comparativo_mixes.png'}")

    with open(root / "tabela_comparativo_mixes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mix", "algoritmo", "servico", "aceitacao_%"])
        w.writerows(linhas)
    print(f"salvo: {root / 'tabela_comparativo_mixes.csv'}")


if __name__ == "__main__":
    main()
