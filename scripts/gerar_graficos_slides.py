"""Gera automaticamente todos os gráficos disponíveis em results/ para slides/."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

try:
    from scripts import gerar_slides, plotar_servicos
except ModuleNotFoundError:
    import gerar_slides
    import plotar_servicos


def mais_recente(pasta: Path, padrao: str) -> Path | None:
    arquivos = sorted(pasta.glob(padrao), key=lambda p: p.stat().st_mtime)
    return arquivos[-1] if arquivos else None


def main() -> None:
    raiz = Path("results/results_flows")
    out = Path("slides")
    out.mkdir(parents=True, exist_ok=True)

    pares_fluxos = []
    pares_resumos = []
    for pasta in sorted(raiz.glob("*_s_*")):
        algoritmo = pasta.name.split("_s_", 1)[0]
        fluxos = [p for p in pasta.glob("*.csv") if not p.name.startswith("service_summary_")]
        fluxo = max(fluxos, key=lambda p: p.stat().st_mtime) if fluxos else None
        resumo = mais_recente(pasta, "service_summary_*.csv")
        if fluxo is None or resumo is None:
            continue
        pares_fluxos.append((algoritmo, fluxo))
        pares_resumos.append((algoritmo, gerar_slides.ler_resumo(resumo)))

    if not pares_fluxos:
        raise SystemExit("Nenhum fluxo CSV com service_summary foi encontrado em results/results_flows.")

    gerar_slides._estilo()
    gerar_slides.fig_visao_geral(
        pares_fluxos,
        "Visão geral dos algoritmos",
        out,
        "fig1_visao_geral",
    )
    gerar_slides.fig_aceitacao_por_servico(pares_resumos, out, "fig2_aceitacao_por_servico")
    gerar_slides.fig_latencia_por_servico_vs_sla(
        pares_resumos, out, "fig3_latencia_por_servico_vs_sla"
    )
    gerar_slides.fig_violins(
        pares_fluxos,
        "decision_time_ms",
        "Tempo de decisão (ms)",
        "Distribuição do tempo de decisão",
        out,
        "fig4_tempo_decisao",
    )
    gerar_slides.fig_violins(
        pares_fluxos,
        "latency",
        "Latência (ms)",
        "Distribuição da latência",
        out,
        "fig5_latencia",
    )
    gerar_slides.fig_uso_recursos(pares_fluxos, out, "fig6_uso_recursos")

    plotar_servicos.plotar_comparativo(
        [pasta / resumo.name for algoritmo, _ in pares_fluxos
         for pasta in [raiz / f"{algoritmo}_s_50_p_6_a_0.99_c_3"]
         for resumo in [mais_recente(pasta, "service_summary_*.csv")]
         if resumo is not None],
        out_dir=out,
    )
    plotar_servicos.plotar_comparativo_linhas(
        [fluxo for _, fluxo in pares_fluxos],
        [algoritmo for algoritmo, _ in pares_fluxos],
        out_dir=out,
    )

    with (out / "manifesto_graficos.csv").open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["algoritmo", "fluxo", "resumo"])
        for algoritmo, fluxo in pares_fluxos:
            resumo = mais_recente(fluxo.parent, "service_summary_*.csv")
            writer.writerow([algoritmo, fluxo.as_posix(), resumo.as_posix() if resumo else ""])
    print(f"Manifesto salvo: {out / 'manifesto_graficos.csv'}")


if __name__ == "__main__":
    main()
