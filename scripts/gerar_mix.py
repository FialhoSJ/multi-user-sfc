"""Gera as figuras padronizadas de UM mix de serviços (pasta slides/mix_<tag>/).

Uso:
    uv run python scripts/gerar_mix.py --tag mix1_equilibrado \
        --flows 'REPLIC=path1,MSF=path2,Kuririn=path3,DARSPPO=path4,Hephaestus=path5' \
        --sums 'path_s1,path_s2,...'
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

SLA_SERVICOS = {"muar": 1000.0, "streaming": 150.0, "voip": 60.0, "iot": 200.0}
SERVICOS_ORDEM = ["muar", "streaming", "voip", "iot"]


def _estilo():
    try:
        matplotlib.font_manager.findfont("Arial")
        plt.rcParams["font.family"] = ["Arial"]
    except Exception:
        plt.rcParams["font.family"] = ["DejaVu Sans"]


def parse_pares(spec: str) -> list[tuple[str, Path]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        rotulo, _, caminho = item.partition("=")
        out.append((rotulo.strip(), Path(caminho.strip())))
    return out


def ler_flows(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for col in r.fieldnames or []:
            cols[col] = []
        for row in r:
            for col in r.fieldnames or []:
                try:
                    cols[col].append(float(row[col]))
                except (ValueError, TypeError):
                    cols[col].append(float("nan"))
    return cols


def ler_resumo(path: Path) -> dict[str, dict[str, float]]:
    dados: dict[str, dict[str, float]] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            dados[row["service_type"]] = {
                k: (float(v) if v else 0.0) for k, v in row.items() if k != "service_type"
            }
    return dados


def salvar(fig, out: Path, nome: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    p = out / f"{nome}.png"
    fig.savefig(p, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {p}")


def fig_visao_geral(pars, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for k, (rotulo, caminho) in enumerate(pars):
        cor = cor_para(rotulo, k)
        d = ler_flows(caminho)
        ax1.plot(d["time_seconds"], d["acceptance_rate"], label=rotulo_curto(rotulo),
                 color=cor, linewidth=2.2)
        ax2.plot(d["time_seconds"], d["cpu_saved"], label=rotulo_curto(rotulo),
                 color=cor, linewidth=2.2)
    for ax, ylab in ((ax1, "Taxa de aceitação (%)"), (ax2, "CPU economizado (sharing)")):
        ax.grid(linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=11)
        ax.set_xlabel("Tempo de simulação (s)", fontsize=13, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    ax1.set_ylim(0, 105)
    fig.suptitle("Visão geral", fontsize=16, fontweight="bold")
    salvar(fig, out, "visao_geral")


def fig_aceitacao_por_servico(sums, out):
    fig, ax = plt.subplots(figsize=(14, 5))
    servicos = [s for s in SERVICOS_ORDEM if any(s in d for _, d in sums)]
    n = len(sums)
    largura = 0.8 / n
    for k, (rotulo, dados) in enumerate(sums):
        valores = [dados.get(s, {}).get("acceptance_rate", 0.0) for s in servicos]
        pos = np.arange(len(servicos)) + (k - (n - 1) / 2) * largura
        ax.bar(pos, valores, width=largura, label=rotulo_curto(rotulo),
               color=cor_para(rotulo, k), edgecolor="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(servicos)))
    ax.set_xticklabels(servicos, fontsize=12, fontweight="bold")
    ax.set_ylabel("Taxa de aceitação (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle("Taxa de aceitação por serviço", fontsize=16, fontweight="bold")
    salvar(fig, out, "aceitacao_por_servico")


def fig_latencia_vs_sla(sums, out):
    fig, ax = plt.subplots(figsize=(14, 5))
    servicos = [s for s in SERVICOS_ORDEM if any(s in d for _, d in sums)]
    n = len(sums)
    largura = 0.8 / n
    for k, (rotulo, dados) in enumerate(sums):
        valores = [dados.get(s, {}).get("avg_latency_ms", 0.0) for s in servicos]
        pos = np.arange(len(servicos)) + (k - (n - 1) / 2) * largura
        ax.bar(pos, valores, width=largura, label=rotulo_curto(rotulo),
               color=cor_para(rotulo, k), edgecolor="black", linewidth=1.0)
    for i, s in enumerate(servicos):
        sla = SLA_SERVICOS.get(s)
        if sla is not None:
            ax.plot([i - 0.5, i + 0.5], [sla, sla], color="red", linestyle="--",
                    linewidth=1.8)
    ax.set_xticks(np.arange(len(servicos)))
    ax.set_xticklabels(servicos, fontsize=12, fontweight="bold")
    ax.set_ylabel("Latência média (ms)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle("Latência média por serviço × SLA", fontsize=16, fontweight="bold")
    salvar(fig, out, "latencia_por_servico_vs_sla")


def fig_uso_recursos(pars, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for k, (rotulo, caminho) in enumerate(pars):
        cor = cor_para(rotulo, k)
        d = ler_flows(caminho)
        ax1.plot(d["time_seconds"], d["bandwidth_utilization"], label=rotulo_curto(rotulo),
                 color=cor, linewidth=2.2)
        ax2.plot(d["time_seconds"], d["cpu_utilization"], label=rotulo_curto(rotulo),
                 color=cor, linewidth=2.2)
    ax1.set_ylabel("Uso de banda (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Uso de CPU (%)", fontsize=13, fontweight="bold")
    for ax in (ax1, ax2):
        ax.grid(linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=11)
        ax.set_xlabel("Tempo de simulação (s)", fontsize=13, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    fig.suptitle("Utilização de recursos", fontsize=16, fontweight="bold")
    salvar(fig, out, "uso_recursos")


def escrever_tabela(sums, out):
    linhas = [["algoritmo", "servico", "requisicoes", "aceitas", "rejeitadas",
               "aceitacao_%", "latencia_ms", "cpu_saved"]]
    for rotulo, dados in sums:
        for s, v in dados.items():
            linhas.append([rotulo, s, str(int(v["requests"])), str(int(v["accepted"])),
                           str(int(v["rejected"])), f"{v['acceptance_rate']:.2f}",
                           f"{v['avg_latency_ms']:.4f}", f"{v['cpu_saved']:.2f}"])
    out.mkdir(parents=True, exist_ok=True)
    p = out / "tabela_resumo.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows(linhas)
    print(f"  salvo: {p}")


def main() -> None:
    _estilo()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--flows", required=True, help="'ROTULO=path,...'")
    parser.add_argument("--sums", required=True, help="paths dos resumos separados por vírgula")
    args = parser.parse_args()

    pars = parse_pares(args.flows)
    sums = [
        (rotulo_curto(Path(p).parent.name.split("_")[0]), ler_resumo(Path(p)))
        for p in args.sums.split(",") if p.strip()
    ]
    out = Path("slides") / args.tag

    fig_visao_geral(pars, out)
    fig_aceitacao_por_servico(sums, out)
    fig_latencia_vs_sla(sums, out)
    fig_uso_recursos(pars, out)
    escrever_tabela(sums, out)


if __name__ == "__main__":
    main()
