````markdown
# MUAR-SFC (Multi-User Service Function Chaining Simulator)

> Um simulador de redes avançado, focado em resiliência, mobilidade e alocação inteligente de *Service Function Chains* (SFC) para serviços imersivos.

O **MUAR-SFC** integra infraestrutura de topologia de redes (NetworkX), simulação de tráfego urbano realístico (Eclipse SUMO) e algoritmos de inteligência artificial de ponta (Deep Reinforcement Learning via Stable-Baselines3 e Algoritmos Genéticos via DEAP) para orquestrar e recuperar serviços de rede sob falhas e mobilidade.

---

## ✨ Principais Funcionalidades

- **Alocação Inteligente (DRL):** Suporte nativo a agentes *Proximal Policy Optimization* (PPO) e *MaskablePPO* para alocação adaptativa de recursos.
- **Gerenciamento de Mobilidade:** Integração robusta com a API TraCI do [Eclipse SUMO](https://eclipse.dev/sumo/) para simular o movimento de usuários de ponta a ponta.
- **Orquestração de Resiliência:** Sistema de *Crashers* e *Backup Managers* para simular falhas em servidores/links e avaliar latência/degradação e recuperação de SFCs.
- **Engenharia de Software de Alto Nível:**
  - Arquitetura blindada em **`src-layout`** evitando colisões e dependências fantasmas.
  - Tolerância a falhas idiomática baseada no padrão **EAFP** (zero `bare-excepts` mascarados).
  - Telemetria e observabilidade centralizada via **Logging** e orientada a objetos via **Pathlib**.

---

## 🛠️ Tecnologias e Dependências

O projeto obedece aos padrões declarativos modernos da PEP 621 e é gerenciado pelo ecossistema de hiper-velocidade **`uv`** (escrito em Rust).

* **Python:** `>= 3.12`
* **Core:** `numpy`, `scipy`, `pandas`, `networkx`
* **Machine Learning:** `stable-baselines3`, `sb3-contrib`, `tensorboard`
* **Otimização:** `deap`, `shapely`
* **Simulação de Tráfego:** `sumo`, `traci`
* **Qualidade e Linting:** `Ruff`, `pytest`

---

## 🚀 Instalação (Ambiente Moderno)

Diga adeus ao arcaico `conda` e `requirements.txt`. O projeto utiliza o gerenciador oficial **`uv`** para garantir resolução determinística em milissegundos.

**1. Instale o `uv` no seu sistema (se ainda não tiver):**
```bash
# MacOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
````

*(Para Windows ou outras opções, veja a documentação do uv)*

**2. Clone o repositório:**

```bash
git clone https://github.com/seu-usuario/multi-user-sfc.git
cd multi-user-sfc
```

**3. Sincronize o ambiente e instale o pacote em modo editável:**
O comando abaixo cria automaticamente o ambiente virtual isolado, baixa os pacotes descritos no `pyproject.toml` e trava as versões pelo `uv.lock`.

```bash
uv pip install -e .
```

---

## 🖥️ Como Executar

Para garantir que o seu código acesse o interpretador encapsulado correto sem precisar ativar o ambiente manualmente, utilize o prefixo `uv run`:

**Executar o simulador principal:**

```bash
uv run python src/muar_sfc/main.py
```

*(Certifique-se de que os traces e configurações do SUMO na pasta `traces/` ou `sumo/` estão devidamente configurados).*

---

## 🏗️ Estrutura do Projeto

A infraestrutura segue as rigorosas diretrizes de isolamento de domínio:

```text
multi-user-sfc/
├── src/
│   └── muar_sfc/                # Código-fonte principal da aplicação
│       ├── algorithms/          # RL (PPO, Maskable), Genéticos, K-Shortest
│       ├── controllers/         # Orquestradores da Rede Substrato e Módulos
│       │   └── modules/         # Backup, Mobility, SFC Instantiators
│       ├── core/                # Infraestrutura de Grafos (Net, Net2)
│       ├── sumo/                # Scripts de Integração TraCI/SUMO
│       └── utils/               # Geradores de Trace, Métricas e Arquivos
├── pyproject.toml               # Fonte Única de Verdade (Dependências e Metadados)
├── uv.lock                      # Hash Criptográfico e Controle Determinístico
├── rl_saved_models/             # Scripts e Modelos Treinados (TensorBoard/Zip)
└── README.md                    # Este arquivo
```

---

## 🧑‍💻 Desenvolvimento e Qualidade Estática

Este projeto preza pela excelência estrutural e possui `0` tolerância a antipadrões ou vazamento de exceções (`E722`). Para validar a saúde do código, rodamos a suíte do **Ruff**.

Para verificar a conformidade estática:

```bash
uv run ruff check .
```

Para aplicar correções automáticas de estilo (como importações ociosas):

```bash
uv run ruff check . --fix
```

---

**Autores e Mantenedores:**

* **Hugo Leonardo** - *[hugosantos@ufpa.br](mailto:hugosantos@ufpa.br)*
* **Sua Equipe/Nome aqui**

```
```
