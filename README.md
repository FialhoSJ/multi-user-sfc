# MUAR-SFC (Multi-User Service Function Chaining Simulator)

Um simulador de redes avançado e de alto desempenho, focado em resiliência, mobilidade e alocação adaptativa de Service Function Chains (SFC) para serviços imersivos.

O MUAR-SFC integra infraestrutura de topologia de redes (**NetworkX**), simulação de tráfego urbano realístico (**Eclipse SUMO**) e inteligência artificial de ponta (**Deep Reinforcement Learning** via Stable-Baselines3) para orquestrar serviços de rede sob condições severas de falhas e mobilidade.

---

## ✨ Diferenciais de Engenharia

Diferente de simuladores acadêmicos convencionais, o MUAR-SFC foi refatorado seguindo padrões industriais de software:

-   **Performance Atômica:** Lógica de falhas e backups otimizada com complexidade $O(1)$, substituindo varreduras lineares por acessos diretos a conjuntos e dicionários.
-   **Arquitetura Robusta:** Organizado em `src-layout` para garantir isolamento total e evitar a "ilusão de corretude" no desenvolvimento.
-   **Resiliência Baseada em EAFP:** Tratamento de erros idiomático (*Easier to Ask for Forgiveness than Permission*), eliminando verificações lentas e silenciamento de bugs críticos.
-   **Multiplataforma Nativa:** Gerenciamento de caminhos via `Pathlib`, garantindo que o simulador rode sem alterações em Linux, macOS ou Windows.
-   **Configuração Centralizada:** Baseado na metodologia *Twelve-Factor App* através do `Pydantic-Settings`.

---

## 🛠️ Tecnologias e Dependências

-   **Runtime:** Python >= 3.12 (Aproveitando melhorias de performance e tipagem).
-   **Gerenciador:** `uv` (Hiper-velocidade escrita em Rust).
-   **IA/ML:** stable-baselines3, sb3-contrib (PPO/MaskablePPO), tensorboard.
-   **Rede:** networkx, simpy.
-   **Tráfego:** Eclipse SUMO (TraCI).
-   **Qualidade:** Ruff (Linting), Pyright (Tipagem Estática).

---

## 🚀 Instalação e Setup

O projeto utiliza o `uv` para garantir que o seu ambiente seja **exatamente igual** ao dos desenvolvedores.

1. **Instale o uv**
    ```bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```

2. **Prepare o ambiente**
    ```bash
    git clone [https://github.com/seu-usuario/multi-user-sfc.git](https://github.com/seu-usuario/multi-user-sfc.git)
    cd multi-user-sfc
    uv sync
    ```

3. **Configuração (.env)**
    Crie o arquivo de variáveis de ambiente:
    ```bash
    cp .env.example .env
    ```
    *Ajuste algoritmos, níveis de confiabilidade e topologia diretamente no `.env`.*

---

## 🖥️ Como Executar

Como o projeto agora é um pacote instalado, você não precisa mais caçar scripts em pastas. Use os comandos registrados:

-   **Simulação Única:**
    ```bash
    uv run muar-sim
    ```
-   **Simulação em Lote (Paralelo):**
    ```bash
    uv run muar-sim-paralelo
    ```
-   **Execução Sequencial:**
    ```bash
    uv run muar-sim-seq
    ```

### Heterogeneidade de Serviços (F1–F5)

O simulador sorteia o tipo de serviço de cada sessão conforme `service_mix` e `service_weights`
(disponíveis no `.env` ou via CLI):

```bash
# Mix heterogêneo (70% MUAR, 10% streaming, 10% VoIP, 10% IoT)
uv run muar-sim --alg replic \
    --service_mix "muar,streaming,voip,iot" \
    --service_weights "0.7,0.1,0.1,0.1"

# Baseline 100% MUAR
uv run muar-sim --service_mix "muar" --service_weights "1.0"
```

Ao final de cada execução é gerado um resumo agregado por serviço
(`results/results_flows/.../service_summary_*.csv`). Para plotar os gráficos do artigo:

```bash
# Gráfico de um run (3 painéis: aceitação, latência+SLA, nº de requisições)
uv run python scripts/plotar_servicos.py                      # usa o CSV mais recente
uv run python scripts/plotar_servicos.py caminho/do/arquivo.csv

# Comparar 2+ algoritmos (barras por serviço)
uv run python scripts/plotar_servicos.py --compare 'replic.csv,msf.csv'

# Comparar algoritmos com GRÁFICOS DE LINHA ao longo do tempo
uv run python scripts/plotar_servicos.py --compare-lines \
  'REPLIC=results/results_flows/replic_s_50_p_4_a_0.99_c_3/FLOWS.csv,MSF=results/results_flows/msf_s_50_p_4_a_0.99_c_3/FLOWS.csv'
```

### Violin plots por faixa de tempo (análise exploratória)

Compara os algoritmos com **violin plots** agrupados por faixa de tempo (200 s, 400 s, …,
1000 s) para as 6 métricas: aceitação, latência, tempo de decisão, SFCs compartilhados, uso
de banda e CPU. O violino mostra a **forma completa da distribuição** (assimetria,
multimodalidade), com mediana (linha cheia), média (linha tracejada), extremos e pontos
brutos sobrepostos — mais informativo que box plots.

```bash
uv run python scripts/plotar_violins.py --out plots/2025/fialhosj/graficos/cenario_padrao \
  'REPLIC=results/results_flows/replic_s_50_p_4_a_0.99_c_3/FLOWS.csv' \
  'MSF=results/results_flows/msf_s_50_p_4_a_0.99_c_3/FLOWS.csv' \
  'Kuririn=results/results_flows/kuririnMaskablePPO_s_50_p_4_a_0.99_c_3/FLOWS.csv' \
  'DARSPPO=results/results_flows/darsppoMaskablePPO_s_50_p_4_a_0.99_c_3/FLOWS.csv' \
  'Hephaestus=results/results_flows/hephaestusMaskablePPO_s_50_p_4_a_0.99_c_3/FLOWS.csv'
```

Saída: `acceptance_rate.png`, `latency.png`, `decision_time.png`, `shared_sfs.png`,
`uso_banda.png`, `uso_cpu.png`. (No `decision_time`, outliers são limitados ao "bigode 6×",
igual ao notebook `plots/2025/davidgn/tratamento_csv.ipynb`.)

### Parâmetros de QoS por serviço (e suas fontes)

Os requisitos de cada serviço (SLA de latência, banda e CPU por VNF) estão definidos em
`src/muar_sfc/core/services/catalog.py`. Os valores são embasados em normas ITU-T/3GPP e
recomendações de mercado:

| Serviço | SLA latência | Banda da SFC | CPU/VNF | Fonte |
|---|---|---|---|---|
| **VoIP** | 60 ms | 0.1 Mbps | 5 / 8 | ITU-T G.114 (voz ≤ 150 ms; orçamento de borda) · ITU-T G.711 (codec 64 kbps + overhead) |
| **Streaming** | 150 ms | **15 Mbps** (4K UHD) | 15 / 45 / 35 | ITU-T G.1010 (live) · Netflix (4K = 15 Mbps) |
| **IoT** | 200 ms | **0.5 Mbps** (agregado de sensores) | 2 / 6 | ITU-T Y.1541 Classe 3 (≤ 400 ms) · 3GPP NB-IoT / LoRaWAN |
| **MUAR** | 1000 ms | 150 Mbps (legado veicular) | 22 / 10 / 8 / 7 | ITU-T Y.1541 Classe 4 (≤ 1 s) · cenário original |

O tráfego ao longo da cadeia é transformado por VNF (`factor`): encoder expande (×1.4) e
transcoder reduz (×0.7) no streaming (F1). A banda solicitada de cada SFC = `in_bw` da 1ª VNF.
Para alterar basta editar o catálogo — sem mexer em código dos algoritmos.

### Retreinamento dos modelos RL para o cenário heterogêneo

Os modelos RL pré-treinados (REPLIC, Kuririn, DARSPPO, Hephaestus) conhecem **só o MUAR**.
Para compará-los na heterogeneidade, é preciso retreinar:

```bash
# 1. Gera o dataset heterogêneo (grafos + SFCs muar/streaming/voip/iot).
#    O gerador pré-popula instâncias de um "player anterior" para o RL aprender reuso.
uv run python scripts/gerar_dataset_rl.py --n_samples 500 --service_weights "0.5,0.2,0.15,0.15"

# 2. Treina do zero (--reset-model apaga o modelo antigo). NÃO tem flag --service_weights aqui;
#    os pesos vêm do dataset gerado no passo 1.
uv run python rl_saved_models/treinar_rl.py --env REPLIC --timesteps 100000 --reset-model
```

Observações:
- O gerador salva em `variaveis_salvas/list_graph_het.pkl` e `list_sfc_het.pkl`.
- `valid_nodes` (servidores + sentinela da ação `dst`) está alinhado entre treino
  e inferência via `build_valid_nodes` (em `algorithms/environments/env_replic.py`).
- O mesmo fluxo vale para `--env` `darsppo`/`hephaestus`/`default` (Kuririn) — cada
  algoritmo precisa do próprio modelo re-treinado; sem isso, os resultados deles no mix
  heterogêneo ficam fora de distribuição (aceitação baixa).

### Resultados de referência (50 sessões, `--n_players 4 --sfc_lifetime 60`)

**Cenário SEM heterogeneidade (baseline 100% MUAR, `--service_mix "muar"`):**

| Algoritmo | muar | CPU economizado (muar) |
|---|---|---|
| **REPLIC** (re-treinado) | 100% | 29380 |
| **MSF** | 100% | 8382 |
| **Kuririn** (re-treinado) | 52% | 5267 |
| **DARSPPO** (re-treinado) | 6.5% | 0 |
| **Hephaestus** (re-treinado) | 0% | 0 |

**Cenário COM heterogeneidade (mix `0.5/0.2/0.15/0.15`, bandas corrigidas: streaming 15 Mbps 4K, IoT 0.5 Mbps):**

| Algoritmo | muar | streaming | iot | voip |
|---|---|---|---|---|
| **REPLIC** (re-treinado) | 100% | **88.6%** | 100% | 100% |
| **MSF** | 100% | **100%** | 100% | 100% |
| **Kuririn** (re-treinado) | 68.2% | 0% | 0% | 0% |
| **DARSPPO** (re-treinado) | 5% | 0% | 0% | 0% |
| **Hephaestus** (re-treinado) | 0% | 0% | 0% | 0% |

*REPLIC e MSF dominam ambos os cenários; os demais modelos RL, mesmo re-treinados com
100k passos, não generalizam bem para o mix heterogêneo — precisam de mais treino/tuning.
Os box plots e gráficos de linha/barra estão em `plots/2025/fialhosj/graficos/`
(`cenario_padrao/` = baseline 100% MUAR; `cenario_heterogeneo/` = mix de serviços).*

### Experimento: variação do mix de serviços (M1–M6)

Para avaliar o impacto da composição do tráfego, foram executados **6 mixes** diferentes
(50 sessões, `--n_players 4 --sfc_lifetime 60`, 5 algoritmos). Tabela de **aceitação do
streaming** (único serviço onde há diferença; muar/voip/iot ficam em ~100% para REPLIC/MSF):

| Mix | pesos (muar/stream/voip/iot) | REPLIC (stream) | MSF (stream) |
|---|---|---|---|
| M1 · Equilibrado | `0.25/0.25/0.25/0.25` | 90.6% | 90.9% |
| M2 · Streaming-heavy | `0.2/0.5/0.15/0.15` | 63.8% | 83.7% |
| M3 · VoIP-heavy | `0.2/0.15/0.5/0.15` | 83.3% | 92.9% |
| M4 · IoT-heavy | `0.2/0.15/0.15/0.5` | 100% | 100% |
| M5 · MUAR-heavy | `0.7/0.1/0.1/0.1` | 87.5% | 100% |
| M6 · Sem-MUAR | `0/0.5/0.25/0.25` | 61.6% | 79.8% |

Observações dos experimentos:
- **Kuririn** aceita apenas MUAR parcialmente (59–89%); **DARSPPO/Hephaestus** ~0% em todos
  os serviços não-MUAR — mesmo re-treinados, não generalizam.
- **M6 (sem MUAR):** `cpu_saved = 0` para todos os algoritmos — o **compartilhamento de VNFs
  só existe no MUAR** (único serviço com `shareable=True` no catálogo).
- **O streaming é o gargalo do sistema:** quando domina o tráfego (M2) ou não há MUAR para
  dividir a rede (M6), a aceitação cai — e o heurístico MSF se mostra mais robusto que o RL.
- Figuras padronizadas para a apresentação em `slides/mix*/` (visão geral, aceitação por
  serviço, latência × SLA, uso de recursos, tabela) e comparativo geral em
  `slides/comparativo_mixes.png`.

Para reproduzir um mix (ex.: M2 streaming-heavy):

```bash
uv run muar-sim --n_sessions 50 --n_players 4 --sfc_lifetime 60 \
  --alg replic --service_mix "muar,streaming,voip,iot" --service_weights "0.2,0.5,0.15,0.15"
# Depois, gerar as figuras daquele mix:
uv run python scripts/gerar_mix.py --tag mix2_streaming \
  --flows 'REPLIC=results/results_flows/replic_s_50_p_4_a_0.99_c_3/FLOWS.csv,MSF=...' \
  --sums 'results/results_flows/replic_s_50_p_4_a_0.99_c_3/service_summary_*.csv,...'
```

---

## 🏗️ Estrutura do Repositório

```text
multi-user-sfc/
├── src/
│   └── muar_sfc/           # Código-fonte principal (Pacote)
│       ├── main.py         # Entrypoint da simulação
│       ├── algorithms/     # REPLIC, DARSPPO, Hephaestus, etc.
│       ├── controllers/    # Orchestrator, SFCManager, Crasher
│       ├── core/           # Configurações (Pydantic) e Net_V2
│       └── utils/          # Helpers de rede e falhas
├── tests/                  # Testes automatizados (Pytest)
├── scripts/                # Geradores de dataset, gráficos e treino
├── rl_saved_models/        # Modelos de rede neural (.zip)
├── pyproject.toml          # Definições de CLI e dependências
├── uv.lock                 # Trava determinística de versões
└── .env                    # Configurações locais de simulação

🧑‍💻 Desenvolvimento e Qualidade
Mantenha a integridade do código com as ferramentas integradas:

Linting e Estilo: uv run ruff check . --fix

Verificação de Tipagem: uv run pyright

Bateria de Testes: uv run pytest

Autores

Hugo Leonardo — hugosantos@ufpa.br

David Galhego — david.galhego@icen.ufpa.br

Matheus Morais de Brito

Erick

Felipe Fialho Nascimento