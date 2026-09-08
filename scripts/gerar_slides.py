"""Gera os gráficos padronizados para a apresentação (pasta slides/).

Estilo científico do `comparativo_linhas`: figuras de 2 painéis, linhas/barras
limpas, paleta canônica por algoritmo, grid suave, rótulos em negrito.

Uso:
    uv run python scripts/gerar_slides.py \
        --out slides \
        --padrao  'REPLIC=flows_base.csv,MSF=flows_base.csv,...' \
        --hetero  'REPLIC=flows_hetero.csv,MSF=flows_hetero.csv,...' \
        --sum-padrao 'resumo_base_replic.csv,resumo_base_msf.csv,...' \
        --sum-hetero 'resumo_hetero_replic.csv,resumo_hetero_msf.csv,...'
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


def novo_ax(fig):
    ax = fig.add_subplot(111)
    ax.grid(linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    return ax


def salvar(fig, out: Path, nome: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    p = out / f"{nome}.png"
    fig.savefig(p, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo: {p}")


def fig_visao_geral(pars, titulo, out, nome):
    """2 painéis: aceitação (%) e CPU economizado ao longo do tempo."""
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
    fig.suptitle(titulo, fontsize=16, fontweight="bold")
    salvar(fig, out, nome)


def fig_aceitacao_por_servico(sums, out, nome):
    """Barras agrupadas: taxa de aceitação (%) por serviço × algoritmo."""
    fig, ax = plt.subplots(figsize=(14, 5))
    servicos = [s for s in SERVICOS_ORDEM if any(s in d for _, d in sums)]
    n = len(sums)
    largura = 0.8 / n
    for k, (rotulo, dados) in enumerate(sums):
        cores = [dados.get(s, {}).get("acceptance_rate", 0.0) for s in servicos]
        pos = np.arange(len(servicos)) + (k - (n - 1) / 2) * largura
        ax.bar(pos, cores, width=largura, label=rotulo_curto(rotulo),
               color=cor_para(rotulo, k), edgecolor="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(servicos)))
    ax.set_xticklabels(servicos, fontsize=12, fontweight="bold")
    ax.set_ylabel("Taxa de aceitação (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.suptitle("Taxa de aceitação por serviço (cenário heterogêneo)",
                 fontsize=16, fontweight="bold")
    salvar(fig, out, nome)


def fig_latencia_por_servico_vs_sla(sums, out, nome):
    """Barras de latência média por serviço + linha do SLA de cada serviço."""
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
    fig.suptitle("Latência média por serviço × SLA (cenário heterogêneo)",
                 fontsize=16, fontweight="bold")
    salvar(fig, out, nome)


def fig_violins(pars, coluna, ylab, titulo, out, nome):
    """Violinos por algoritmo (distribuição da métrica) — 1 painel."""
    fig, ax = plt.subplots(figsize=(14, 5))
    posicoes = []
    rotulos = []
    rng = np.random.default_rng(7)
    for k, (rotulo, caminho) in enumerate(pars):
        d = ler_flows(caminho)
        valores = np.array(d[coluna])
        valores = valores[~np.isnan(valores)]
        pos = k + 0.5
        ax.violinplot([valores], positions=[pos], widths=0.6, showmedians=True,
                      showextrema=True)
        xs = pos + rng.uniform(-0.12, 0.12, size=len(valores))
        ax.scatter(xs, valores, s=3, alpha=0.25, color=cor_para(rotulo, k),
                   linewidths=0, zorder=3)
        posicoes.append(pos)
        rotulos.append(rotulo_curto(rotulo))
    ax.set_xticks(posicoes)
    ax.set_xticklabels(rotulos, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.suptitle(titulo, fontsize=16, fontweight="bold")
    salvar(fig, out, nome)


def fig_uso_recursos(pars, out, nome):
    """2 painéis: uso de banda (%) e de CPU (%) ao longo do tempo."""
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
    fig.suptitle("Utilização de recursos (cenário heterogêneo)", fontsize=16,
                 fontweight="bold")
    salvar(fig, out, nome)


def main() -> None:
    _estilo()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="slides")
    parser.add_argument("--padrao", required=True, help="'ROTULO=path,...' flows baseline")
    parser.add_argument("--hetero", required=True, help="'ROTULO=path,...' flows heterogêneo")
    parser.add_argument("--sum-hetero", required=True, help="paths dos resumos heterogêneo")
    args = parser.parse_args()

    out = Path(args.out)
    pars_padrao = parse_pares(args.padrao)
    pars_hetero = parse_pares(args.hetero)
    sums_hetero = [
        (rotulo_curto(Path(p).parent.name.split("_")[0]), ler_resumo(Path(p)))
        for p in args.sum_hetero.split(",") if p.strip()
    ]

    fig_visao_geral(pars_padrao, "Visão geral — cenário padrão (100% MUAR)",
                    out, "fig1_visao_geral_baseline")
    fig_visao_geral(pars_hetero, "Visão geral — cenário heterogêneo (mix 0.5/0.2/0.15/0.15)",
                    out, "fig2_visao_geral_heterogeneo")
    fig_aceitacao_por_servico(sums_hetero, out, "fig3_aceitacao_por_servico")
    fig_latencia_por_servico_vs_sla(sums_hetero, out, "fig4_latencia_por_servico_vs_sla")
    fig_violins(pars_hetero, "decision_time_ms", "Tempo de decisão (ms)",
                "Distribuição do tempo de decisão (heterogêneo)",
                out, "fig5_tempo_decisao")
    fig_violins(pars_hetero, "latency", "Latência (ms)",
                "Distribuição da latência (heterogêneo)", out, "fig6_latencia")
    fig_uso_recursos(pars_hetero, out, "fig7_uso_recursos")


if __name__ == "__main__":
    main()
