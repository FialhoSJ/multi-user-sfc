MUAR-SFC (Multi-User Service Function Chaining Simulator)

  Um simulador de redes avançado, focado em resiliência, mobilidade e
  alocação inteligente de Service Function Chains (SFC) para serviços
  imersivos.

O MUAR-SFC integra infraestrutura de topologia de redes (NetworkX),
simulação de tráfego urbano realístico (Eclipse SUMO) e algoritmos de
inteligência artificial de ponta (Deep Reinforcement Learning via
Stable-Baselines3 e Algoritmos Genéticos via DEAP) para orquestrar e
recuperar serviços de rede sob falhas e mobilidade.

------------------------------------------------------------------------

✨ Principais Funcionalidades

-   Alocação Inteligente (DRL): Suporte nativo a agentes Proximal Policy
    Optimization (PPO) e MaskablePPO para alocação adaptativa de
    recursos.
-   Gerenciamento de Mobilidade: Integração robusta com a API TraCI do
    Eclipse SUMO (https://eclipse.dev/sumo/) para simular o movimento de
    usuários de ponta a ponta.
-   Orquestração de Resiliência: Sistema de Crashers e Backup Managers
    para simular falhas em servidores/links e avaliar
    latência/degradação e recuperação de SFCs.
-   Engenharia de Software de Alto Nível:
    -   Arquitetura blindada em src-layout evitando colisões e
        dependências fantasmas.
    -   Tolerância a falhas idiomática baseada no padrão EAFP (zero
        bare-excepts mascarados).
    -   Telemetria e observabilidade centralizada via Logging e
        orientada a objetos via Pathlib.

------------------------------------------------------------------------

🛠️ Tecnologias e Dependências

O projeto obedece aos padrões declarativos modernos da PEP 621 e é
gerenciado pelo ecossistema de hiper-velocidade uv (escrito em Rust).

-   Python: >= 3.12
-   Core: numpy, scipy, pandas, networkx
-   Machine Learning: stable-baselines3, sb3-contrib, tensorboard
-   Otimização: deap, shapely
-   Simulação de Tráfego: sumo, traci
-   Qualidade e Linting: Ruff, pytest

------------------------------------------------------------------------

🚀 Instalação (Ambiente Moderno)

Diga adeus ao arcaico conda e requirements.txt. O projeto utiliza o
gerenciador oficial uv para garantir resolução determinística em
milissegundos.

1. Instale o uv

    # MacOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

2. Clone o repositório

    git clone https://github.com/seu-usuario/multi-user-sfc.git
    cd multi-user-sfc

3. Instale as dependências

    uv pip install -e .

------------------------------------------------------------------------

🖥️ Como Executar

Para garantir que o seu código utilize o interpretador correto sem
ativar o ambiente manualmente:

    uv run python scripts/main.py

------------------------------------------------------------------------

🏗️ Estrutura do Projeto

    multi-user-sfc/
    ├── src/
    │   └── muar_sfc/
    │       ├── algorithms/
    │       ├── controllers/
    │       │   └── modules/
    │       ├── core/
    │       ├── sumo/
    │       └── utils/
    ├── pyproject.toml
    ├── uv.lock
    ├── rl_saved_models/
    └── README.md

------------------------------------------------------------------------

🧑‍💻 Desenvolvimento e Qualidade Estática

Para verificar a qualidade do código:

    uv run ruff check .

Para aplicar correções automáticas:

    uv run ruff check . --fix

------------------------------------------------------------------------

Autores

-   Hugo Leonardo — hugosantos@ufpa.br
-   David Galhego - david.galhego@icen.ufpa.br
-   Matheus Morais de Brito 
-   Erick
