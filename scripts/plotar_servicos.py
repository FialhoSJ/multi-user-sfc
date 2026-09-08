"""Gera os gráficos de heterogeneidade de serviços para o artigo (F4).

Lê um arquivo `service_summary_*.csv` (produzido pelo simulador ao final de cada
execução) e plota dois gráficos em PDF:
  1) Taxa de aceitação por tipo de serviço;
  2) Latência média por tipo de serviço.

Uso:
    uv run python scripts/plotar_servicos.py [caminho_do_service_summary.csv]
Se nenhum caminho for passado, o script usa o CSV de resumo mais recente em results/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.cores_algoritmos import cor_para, rotulo_curto
except ModuleNotFoundError:
    from cores_algoritmos import cor_para, rotulo_curto

# ==============================================================================
# ESTILO CIENTÍFICO (consistente com plotar_recompensas.py)
# ==============================================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "serif"],
    }
)

CORES_SERVICOS = {
    "muar": "#017507",
    "streaming": "#E24A0E",
    "voip": "#2C16AC",
    "iot": "#EA4AF9",
}

# SLA de latência (ms) por serviço, para referência no gráfico
SLA_LATENCIA_MS = {
    "muar": 1000.0,
    "streaming": 150.0,
    "voip": 60.0,
    "iot": 200.0,
}


def _latest_summary() -> Path:
    """Localiza o CSV de resumo por serviço mais recente dentro de results/."""
    candidates = sorted(Path("results").rglob("service_summary_*.csv"))
    if not candidates:
        sys.exit("Nenhum service_summary_*.csv encontrado em results/.")
    return candidates[-1]


def carregar_resumo(caminho: Path) -> dict:
    """Carrega o CSV e devolve {service: {coluna: valor}}."""
    dados: dict[str, dict] = {}
    with open(caminho, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dados[row["service_type"]] = {
                k: (float(v) if v else 0.0) for k, v in row.items() if k != "service_type"
            }
    return dados


def plotar(caminho: Path, out_dir: Path | None = None) -> None:
    dados = carregar_resumo(caminho)
    if not dados:
        sys.exit("Resumo vazio: nada a plotar.")

    servicos = sorted(dados)
    aceitacao = np.array([dados[s]["acceptance_rate"] for s in servicos])
    latencia = np.array([dados[s]["avg_latency_ms"] for s in servicos])
    requests = np.array([dados[s]["requests"] for s in servicos])
    cores = [CORES_SERVICOS.get(s, "#999999") for s in servicos]
    slas = [SLA_LATENCIA_MS.get(s) for s in servicos]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    # --- 1. Taxa de aceitação por serviço ---
    ax1.bar(servicos, aceitacao, color=cores, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Taxa de aceitação (%)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # --- 2. Latência média por serviço + SLA ---
    ax2.bar(servicos, latencia, color=cores, edgecolor="black", linewidth=1.2)
    ax2.set_ylabel("Latência média (ms)", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    for s, sla in zip(servicos, slas, strict=True):
        if sla is not None:
            cor = cores[servicos.index(s)]
            ax2.axhline(y=sla, color=cor, linestyle="--", alpha=0.6, linewidth=1.0)

    # --- 3. Nº de requisições por serviço (reflete o mix/weights) ---
    ax3.bar(servicos, requests, color=cores, edgecolor="black", linewidth=1.2)
    ax3.set_ylabel("Requisições", fontsize=13, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    for i, v in enumerate(requests):
        ax3.text(i, v, int(v), ha="center", va="bottom", fontsize=10, fontweight="bold")

    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis="x", labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.suptitle("Heterogeneidade de serviços", fontsize=16, fontweight="bold")
    fig.tight_layout()

    pdf = caminho.with_name("grafico_servicos.pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"Gráfico salvo: {pdf}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "grafico_servicos.png"
        fig.savefig(png, format="png", bbox_inches="tight", pad_inches=0.05, dpi=150)
        print(f"Gráfico salvo: {png}")


def plotar_comparativo(caminhos: list[Path], out_dir: Path | None = None) -> None:
    """Compara múltiplos algoritmos: aceitação (%) e CPU economizado por serviço."""
    dados_algos = []
    for caminho in caminhos:
        dados = carregar_resumo(caminho)
        if not dados:
            sys.exit(f"Resumo vazio: {caminho}")
        rotulo = rotulo_curto(caminho.parent.name.split("_")[0])
        dados_algos.append((rotulo, dados))

    servicos = sorted({s for _, d in dados_algos for s in d})

    largura = 0.8 / len(dados_algos)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1. Taxa de aceitação por serviço, por algoritmo ---
    for k, (rotulo, dados) in enumerate(dados_algos):
        valores = np.array([dados.get(s, {}).get("acceptance_rate", 0.0) for s in servicos])
        pos = np.arange(len(servicos)) + (k - (len(dados_algos) - 1) / 2) * largura

        ax1.bar(pos, valores, width=largura, label=rotulo, color=cor_para(rotulo, k),
                edgecolor="black", linewidth=1.2)

    ax1.set_xticks(np.arange(len(servicos)))
    ax1.set_xticklabels(servicos)
    ax1.set_ylabel("Taxa de aceitação (%)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)

    # --- 2. CPU economizado por serviço, por algoritmo ---
    for k, (rotulo, dados) in enumerate(dados_algos):
        valores = np.array([dados.get(s, {}).get("cpu_saved", 0.0) for s in servicos])
        pos = np.arange(len(servicos)) + (k - (len(dados_algos) - 1) / 2) * largura

        ax2.bar(pos, valores, width=largura, label=rotulo, color=cor_para(rotulo, k),
                edgecolor="black", linewidth=1.2)

    ax2.set_xticks(np.arange(len(servicos)))
    ax2.set_xticklabels(servicos)
    ax2.set_ylabel("CPU economizado (sharing)", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="best", fontsize=9, framealpha=0.9)

    # --- AJUSTE VISUAL PARA BARRAS: Impede barras gigantes se houver só 1 serviço ---
    if len(servicos) == 1:
        ax1.set_xlim(-0.8, 0.8)
        ax2.set_xlim(-0.8, 0.8)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.suptitle("Comparação de algoritmos por serviço", fontsize=16, fontweight="bold")
    fig.tight_layout()

    pdf = caminhos[0].with_name("grafico_comparativo.pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"Gráfico comparativo salvo: {pdf}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "comparativo_servicos.png"
        fig.savefig(png, format="png", bbox_inches="tight", pad_inches=0.05, dpi=150)
        print(f"Gráfico comparativo salvo: {png}")


def plotar_comparativo_linhas(
    fluxos: list[Path], rotulos: list[str], out_dir: Path | None = None
) -> None:
    """Compara algoritmos com GRÁFICOS DE LINHA ao longo do tempo (fluxos CSV).

    Painéis: (1) taxa de aceitação (%) e (2) CPU economizado (cumulativo),
    ambos vs tempo de simulação — uma linha por algoritmo.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for k, (caminho, rotulo_orig) in enumerate(zip(fluxos, rotulos, strict=False)):
        rotulo = rotulo_curto(rotulo_orig)
        cor = cor_para(rotulo_orig, k)
        tempos, aceit, saved = [], [], []
        with open(caminho, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tempos.append(float(row["time_seconds"]))
                aceit.append(float(row["acceptance_rate"]))
                saved.append(float(row["cpu_saved"]))

        # Gráficos mais limpos: linhas totalmente sólidas (linestyle="-") e sem marcadores
        ax1.plot(tempos, aceit, label=rotulo, color=cor, linewidth=2.0, linestyle="-")
        ax2.plot(tempos, saved, label=rotulo, color=cor, linewidth=2.0, linestyle="-")

    ax1.set_xlabel("Tempo de simulação (s)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Taxa de aceitação (%)", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(linestyle="--", alpha=0.3)
    # Legenda inteligente (procura o espaço mais vazio) e sutil
    ax1.legend(loc="best", fontsize=9, framealpha=0.9)

    ax2.set_xlabel("Tempo de simulação (s)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("CPU economizado (sharing)", fontsize=13, fontweight="bold")
    ax2.grid(linestyle="--", alpha=0.3)
    # Legenda inteligente (procura o espaço mais vazio) e sutil
    ax2.legend(loc="best", fontsize=9, framealpha=0.9)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    fig.suptitle("Comparação de algoritmos (heterogeneidade de serviços)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()

    pdf = fluxos[0].with_name("grafico_comparativo_linhas.pdf")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"Gráfico de linhas salvo: {pdf}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / "comparativo_linhas.png"
        fig.savefig(png, format="png", bbox_inches="tight", pad_inches=0.05, dpi=150)
        print(f"Gráfico de linhas salvo: {png}")


def _parse_compare_lines(spec: str) -> tuple[list[Path], list[str]]:
    """Converte 'rotulo1=path1,rotulo2=path2' em (paths, rótulos)."""
    fluxos, rotulos = [], []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            rotulo, caminho = item.split("=", 1)
        else:
            rotulo, caminho = Path(item).parent.name.split("_")[0], item
        fluxos.append(Path(caminho))
        rotulos.append(rotulo)
    return fluxos, rotulos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota o resumo por tipo de serviço.")
    parser.add_argument("caminho", nargs="?", type=Path, default=None,
                        help="CSV service_summary gerado pelo simulador.")
    parser.add_argument("--compare", type=str, default=None,
                        help="Dois CSVs separados por vírgula para comparar algoritmos "
                             "(ex.: --compare 'replic.csv,msf.csv')")
    parser.add_argument("--compare-lines", type=str, default=None,
                        help="Flows CSV dos algoritmos com rótulo, separados por vírgula "
                             "(ex.: --compare-lines 'REPLIC=replic.csv,MSF=msf.csv')")
    parser.add_argument("--out", type=Path, default=None,
                        help="Diretório de saída (salva PNG além do PDF).")
    args = parser.parse_args()

    if args.compare_lines:
        fluxos, rotulos = _parse_compare_lines(args.compare_lines)
        plotar_comparativo_linhas(fluxos, rotulos, out_dir=args.out)
    elif args.compare:
        caminhos = [Path(p.strip()) for p in args.compare.split(",") if p.strip()]
        plotar_comparativo(caminhos, out_dir=args.out)
    else:
        plotar(args.caminho or _latest_summary(), out_dir=args.out)
