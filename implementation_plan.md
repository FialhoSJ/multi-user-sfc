# Análise dos Resultados e Melhoria dos Gráficos

## 1. Análise de Validação dos Resultados

### Dados Reais Extraídos dos CSVs (1 run cada)

| Métrica | Híbrido | VEGETA | GreedyB | MUSFiCO | MSF |
|---------|---------|--------|---------|---------|-----|
| **Aceitação** | 97.97% | 93.20% | 82.42% | 93.39% | 96.92% |
| **Latência** | 8.62ms | 10.67ms | 4.84ms | 8.39ms | 7.93ms |
| **CPU util.** | 0.0517 | 0.0418 | 0.0978 | 0.0775 | 0.1032 |
| **Cache util.** | 0.0748 | 0.1746 | 0.4490 | 0.2994 | 0.2994 |
| **Banda util.** | 0.0240 | 0.0533 | 0.0484 | 0.0830 | 0.1178 |
| **Energia (W)** | 5228 | 5178 | 4913 | 5279 | 5826 |
| **Confiabilidade** | 0.987 | 0.927 | 0.917 | 0.801 | 0.742 |
| **SFCs ativas** | 54 | 74 | 72 | 61 | 53 |

### Discrepâncias vs. Resumo do Usuário

> [!WARNING]
> Os dados reais dos CSVs **divergem significativamente** do resumo apresentado na mensagem. Isto pode indicar que os CSVs atuais são de um run diferente dos que geraram o resumo, ou que houve mais uma rodada de correções.

| Métrica | Resumo do Usuário | CSV Real | Status |
|---------|-------------------|----------|--------|
| Aceitação Híbrido | 98.1% | 97.97% | ≈ OK |
| Latência Híbrido | **6.49ms** | **8.62ms** | ⚠️ Diverge! |
| CPU Híbrido | 0.057 | 0.0517 | ≈ OK |
| Cache Híbrido | 0.117 | 0.0748 | ⚠️ Diverge! |
| Banda Híbrido | 0.023 | 0.024 | ≈ OK |
| Energia Híbrido | **4455W** | **5228W** | ⚠️ Diverge! |
| Confiabilidade Híbrido | 0.943 | **0.987** | ⚠️ Diverge (melhor!) |
| Latência VEGETA | **6.79ms** | **10.67ms** | ⚠️ Diverge! |
| Energia VEGETA | **4623W** | **5178W** | ⚠️ Diverge! |
| Confiabilidade VEGETA | 0.957 | **0.927** | ⚠️ Diverge! |

> [!IMPORTANT]
> **Ação necessária**: Os CSVs atuais no diretório `results/results_flows/` **NÃO** correspondem aos números do resumo. É preciso:
> 1. Confirmar se os CSVs corretos (pós-correções) foram salvos, ou
> 2. Re-rodar os experimentos com as correções aplicadas, ou
> 3. Verificar se há outro diretório de resultados mais recente.

---

## 2. Análise Crítica (baseada nos dados REAIS dos CSVs)

### ✅ Pontos Fortes do Híbrido (comprovados pelos dados reais)

1. **Aceitação mais alta (97.97%)**: supera todos os baselines — VEGETA 93.2%, MSF 96.9%
2. **Banda MUITO mais eficiente (0.024)**: vs VEGETA 0.053, MUSFiCO 0.083, MSF 0.118 — **redução de 55% vs VEGETA e 80% vs MSF**. Este é o ponto mais forte.
3. **Confiabilidade líder (0.987)**: MUITO acima de todos — VEGETA 0.927, GreedyB 0.917, MUSFiCO 0.801, MSF 0.742
4. **Cache eficiente (0.075)**: vs VEGETA 0.175 (57% menos), GreedyB 0.449 (83% menos)

### ⚠️ Pontos que Necessitam Esclarecimento

1. **Latência (8.62ms)**: **NÃO vence VEGETA** nos dados atuais (10.67 vs 8.62 — vence), MAS **perde para GreedyB (4.84ms) e MSF (7.93ms)**. O GreedyB tem latência 44% menor.
2. **Energia (5228W)**: **NÃO vence** — GreedyB (4913W) e VEGETA (5178W) gastam menos. Isto contradiz o claim do artigo.
3. **CPU (0.052)**: menor que VEGETA (0.042) mas próximo — isto é bom (eficiência), mas o VEGETA usa MENOS CPU.

### 🔍 O Problema Fundamental: Latência vs Energia

O Híbrido aceita mais SFCs (98% vs 93%), mas cada SFC individual tem latência e energia comparáveis ou ligeiramente maiores. O GreedyB tem a menor latência (4.84ms) e energia (4913W), mas a pior aceitação (82.4%). Há um trade-off claro:

**Mais aceitação = mais carga = mais energia e latência por SFC**.

### Interpretação para o Artigo

Para o objetivo declarado (economizar banda, baixa latência, energia):

| Objetivo | Resultado | Avaliação |
|----------|-----------|-----------|
| **Economizar banda** | ✅ Melhor de todos (0.024) | **Ponto principal do artigo** |
| **Latência baixa** | ⚠️ 2° lugar (8.62ms, perde p/ MSF 7.93 e GreedyB 4.84) | Precisa argumentar trade-off |
| **Energia** | ⚠️ 3° lugar (5228W, perde p/ GreedyB 4913 e VEGETA 5178) | Precisa normalizar por SFC |

> [!TIP]
> **Métrica-chave para o artigo**: Considere **normalizar energia e latência pelo número de SFCs aceitas**. O Híbrido roda 54 SFCs com 5228W = **96.8 W/SFC**, enquanto GreedyB roda 72 com 4913W = **68.2 W/SFC**. VEGETA roda 74 com 5178W = **69.9 W/SFC**. Isso mostra que o Híbrido é menos eficiente per-SFC. Mas o Híbrido aceita 98% vs 82% do GreedyB — ele atende mais USUÁRIOS com recurso de banda drasticamente menor.

---

## 3. Sugestões de Melhoria dos Resultados

1. **Normalizar métricas por taxa de aceitação**: "Custo por SFC aceita" é uma métrica justa que nivela as comparações
2. **Destacar banda como diferencial principal**: redução de 55-80% vs baselines é fortíssimo
3. **Re-rodar com 3-5 seeds** para ter desvio padrão e significância estatística (atualmente N=1)

---

## 4. Proposta de Gráficos Melhorados

### Gráficos a REMOVER (redundantes ou pouco informativos)
- `bar_cpu_saved.png`, `bar_cache_saved.png` — métricas absolutas sem contexto
- `bar_comp_latency.png`, `bar_comm_latency.png` — detalhes; o total basta
- `grouped_latency.png` — substitído pelo heatmap
- `line_comp_latency.png`, `line_comm_latency.png` — detalhes; o line_latency basta
- `line_running_sfcs.png` — métrica interna, não é objetivo do artigo

### Gráficos NOVOS propostos (8 gráficos focados)

1. **📊 Tabela-Resumo Visual (Heatmap)**: Heatmap normalizado com TODAS as métricas — cada célula coloreada por percentil. Mostra de relance quem é melhor em quê.

2. **🕸️ Radar (melhorado)**: Radar com 7 eixos (Aceitação, Banda, Latência, Energia, CPU, Cache, Confiabilidade), normalizado [0,1] onde 1 = melhor, com os eixos focados nos objetivos do artigo.

3. **📈 Barras Horizontais com % de Melhoria**: Barra horizontal mostrando "% de melhoria do Híbrido vs VEGETA" para cada métrica. Valores positivos = Híbrido melhor. Visual impactante.

4. **📉 Evolução da Banda (line chart)**: O gráfico de linha de banda já existe e mostra claramente a superioridade. Manter mas melhorar a estética.

5. **📊 Barras de Aceitação + Confiabilidade (dual-axis)**: Um gráfico combinado mostrando aceitação e confiabilidade lado a lado — os dois pontos fortes do Híbrido.

6. **📊 Barras de Utilização de Recursos (grouped)**: CPU + Cache + Banda agrupados por algoritmo. Manter mas melhorar estética.

7. **📈 Evolução da Latência (line chart)**: Manter mas com moving average para suavizar ruído.

8. **📈 Evolução da Energia (line chart)**: Manter mas melhorar estética.

### Melhorias Estéticas Gerais
- Fundo branco limpo com grade sutil
- Fontes profissionais (DejaVu Sans / Liberation Sans)
- Paleta de cores harmônica: Híbrido em **vermelho escuro #C0392B** (destaque), baselines em tons frios
- DPI 300 para qualidade de publicação
- Anotações com os valores exatos nos gráficos de barra
- Legenda posicionada consistentemente
- Labels pt-BR em todos os eixos

---

## Open Questions

> [!IMPORTANT]
> 1. **Os CSVs atuais são os definitivos?** Os números extraídos dos CSVs divergem muito do resumo que você apresentou. Preciso saber qual versão é a correta para gerar os gráficos.
> 
> 2. **Quantos runs/seeds foram feitos?** Atualmente há apenas 1 run por algoritmo. Para publicação científica, o ideal é 3-5 runs com desvio padrão. Os gráficos atuais mostram bandas de confiança zeradas.
>
> 3. **Devo gerar os gráficos com os dados ATUAIS dos CSVs, ou você vai re-rodar os experimentos primeiro?**

## Plano de Execução

1. Reescrever `slides/generate_charts.py` com os 8 gráficos propostos
2. Estética profissional de publicação (DPI 300, fontes limpas, paleta coesa)
3. Gerar HTML de apresentação atualizado
4. Todos os textos em pt-BR

## Verificação

- Rodar o script e verificar que todos os 8 gráficos são gerados corretamente
- Verificar que os dados nos gráficos correspondem aos CSVs
