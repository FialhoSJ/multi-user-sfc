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

**Cenário COM heterogeneidade (mix `0.5/0.2/0.15/0.15`):**

| Algoritmo | muar | streaming | iot | voip |
|---|---|---|---|---|
| **REPLIC** (re-treinado) | 100% | **81.8%** | 100% | 100% |
| **MSF** | 100% | **97.6%** | 100% | 100% |
| **Kuririn** (re-treinado) | 62% | 0% | 0% | 0% |
| **DARSPPO** (re-treinado) | 2.3% | 0% | 0% | 0% |
| **Hephaestus** (re-treinado) | 0% | 0% | 0% | 0% |

*REPLIC e MSF dominam ambos os cenários; os demais modelos RL, mesmo re-treinados com
100k passos, não generalizam bem para o mix heterogêneo — precisam de mais treino/tuning.
Gráficos de linha gerados por `--compare-lines`:
`grafico_comparativo_linhas_baseline.pdf` e `grafico_comparativo_linhas_heterogeneo.pdf`.*

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