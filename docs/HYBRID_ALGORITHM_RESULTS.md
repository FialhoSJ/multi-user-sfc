# Algoritmo Hybrid: arquitetura, fundamentos e resultados

## 1. Visão geral

O `Hybrid` é um algoritmo de alocação de Service Function Chains (SFCs) em redes NFV/edge computing. Ele foi desenvolvido combinando ideias de três trabalhos científicos com os modelos, métricas e algoritmos já existentes neste projeto.

O objetivo do algoritmo é tomar decisões de alocação considerando simultaneamente:

- aceitação das requisições;
- uso de banda;
- latência de comunicação e processamento;
- custo operacional;
- congestionamento;
- confiabilidade dos nós;
- disponibilidade de CPU e outros recursos.

O nome `Hybrid` (temporario) vem principalmente da ação composta usada pelo agente:

```text
ação = (nó escolhido, fração contínua de CPU alpha)
```

Assim, o agente decide tanto **onde** instalar a VNF quanto **quanto recurso computacional** alocar para ela.

## 2. Fundamentos científicos

### 2.1. Embedding de SFC com aprendizado por reforço

O artigo de Wang et al. formula o embedding de SFC como um problema de decisão sequencial. O agente observa o estado atual da rede, escolhe uma ação de alocação e recebe uma recompensa.

Essa ideia foi incorporada ao projeto por meio do ambiente `SFC_AllocationEnv_Hybrid`, que mantém o estado de:

- CPU disponível;
- banda disponível;
- latência;
- topologia;
- VNFs ainda não alocadas;
- nós candidatos;
- recursos já utilizados por outras SFCs.

O artigo serviu principalmente como base conceitual para a modelagem MDP do problema. O Hybrid não reproduz diretamente o algoritmo distribuído do artigo: a implementação atual usa um agente PPO centralizado.

### 2.2. Ação discreta e contínua

O artigo de Zhao et al. utiliza uma ação composta para decidir simultaneamente o posicionamento da VNF e a quantidade de recurso computacional alocada.

No Hybrid, essa ideia aparece como:

```python
action = (node_idx, alpha)
```

Onde:

- `node_idx` é uma ação discreta que seleciona um servidor;
- `alpha` é uma ação contínua entre 0,01 e 1, representando a fração de CPU alocada.

O artigo usa MAPPO com CTDE, isto é, treinamento centralizado e execução descentralizada. O Hybrid adapta o conceito de ação composta, mas utiliza PPO de agente único. Portanto, a implementação não é um MAPPO completo.

### 2.3. Otimização consciente de banda

O artigo de Wu et al. propõe uma abordagem de SFC consciente de banda. A proposta considera explicitamente o custo dos enlaces usados entre VNFs, em vez de avaliar somente a CPU dos nós.

O Hybrid incorpora essa ideia em três níveis:

1. custo de banda por saltos extras;
2. penalização de pressão sobre enlaces com pouca capacidade disponível;
3. penalização pelo número de saltos do caminho.

O custo básico usado pelo ambiente é:

```text
C_banda = banda_requisitada * max(numero_de_saltos - 1, 0)
```

Essa fórmula busca evitar caminhos desnecessariamente longos e preservar banda para requisições futuras.

## 3. Arquitetura do Hybrid

### 3.1. Estado de entrada

O estado entregue à rede neural possui três componentes principais:

```text
node_feats  - características dos nós
adj_matrix  - topologia da rede
sfc_seq     - sequência de VNFs da SFC atual
```

As características dos nós representam recursos e condições locais, incluindo informações relacionadas a CPU, cache, banda e utilização.

### 3.2. GCN para a topologia

O arquivo `src/muar_sfc/algorithms/hybrid/networks.py` implementa uma camada `GCNLayer`.

A GCN propaga informações entre nós vizinhos utilizando a matriz de adjacência. Dessa forma, o agente não analisa somente a capacidade isolada de um servidor. Ele também pode considerar o contexto topológico ao redor do servidor.

Isso é importante porque uma escolha aparentemente boa em termos de CPU pode ser ruim em termos de caminho, banda ou congestionamento.

### 3.3. LSTM para a sequência de VNFs

A sequência de VNFs é codificada por uma LSTM. A ordem da cadeia é importante porque diferentes ordens produzem diferentes requisitos de comunicação e processamento.

Por exemplo, a decisão para uma cadeia que começa com uma função de filtragem não precisa ser igual à decisão para uma cadeia que começa com uma função de transcodificação.

A LSTM usada no projeto é uma adaptação da ideia de representação sequencial do artigo de Wu et al. Ela não é uma implementação completa do Seq2Seq original.

### 3.4. Cabeças de decisão

A rede produz três saídas:

1. `node_logits`: pontuações para os nós candidatos;
2. `beta_params`: parâmetros da distribuição Beta para gerar `alpha`;
3. `value`: estimativa de valor usada pelo critic do PPO.

O nó é escolhido por uma distribuição categórica. O valor de `alpha` é amostrado de uma distribuição Beta e limitado ao intervalo `[0,01, 1,0]`.

## 4. Relação entre alpha, CPU e latência

O parâmetro `alpha` cria o principal compromisso do algoritmo:

```text
alpha menor  -> menor consumo de CPU, maior latência de processamento
alpha maior  -> maior consumo de CPU, menor latência de processamento
```

No ambiente, a CPU efetivamente alocada é proporcional a `alpha`:

```text
CPU efetiva = CPU requisitada * alpha
```

Já a latência de processamento é calculada aproximadamente como:

```text
latência de processamento = latência computacional / alpha
```

Consequentemente, o Hybrid pode economizar recursos e, ao mesmo tempo, apresentar latência maior. Esse comportamento é esperado pela formulação atual e não representa necessariamente um erro de implementação.

## 5. Função de custo e recompensa

### 5.1. Componentes básicos

O avaliador `HybridSFCCostEvaluator` combina três custos principais:

#### Custo de banda

Penaliza saltos extras no caminho entre VNFs.

#### Custo operacional

Combina potência dos nós ativos e desgaste de ativação/desativação:

```text
C_operacional = alpha_power * potência_ativa
                + beta_wear * alterações_de_nós_ativos
```

#### Custo de latência

A latência é composta por:

```text
latência = comunicação + instanciação + processamento
```

### 5.2. Recompensa básica

Os pesos padrão do avaliador são:

```text
peso de banda       = 0,3
peso operacional    = 0,3
peso de latência    = 0,4
```

A recompensa combina os retornos de banda, operação e latência, além de penalizar violações de SLA e consumo operacional excessivo.

O ambiente adiciona ainda:

- bônus por completar a SFC;
- penalidade de falha;
- penalidade de congestionamento;
- recompensa de confiabilidade;
- penalidade de pressão de banda;
- penalidade por quantidade de saltos.

Portanto, o Hybrid não otimiza latência isoladamente. Ele resolve um problema multiobjetivo.

## 6. O que veio dos algoritmos existentes

O Hybrid reutiliza a infraestrutura já existente no projeto:

- modelo de SFC e VNF;
- topologia física;
- alocador de recursos;
- compartilhamento de VNFs;
- busca de caminhos;
- cálculo de utilização de CPU, cache e banda;
- cálculo de latência;
- registro dos resultados em CSV;
- mecanismos de comparação entre algoritmos.

Também foram incorporadas ideias relacionadas aos algoritmos anteriores:

- controle de nós válidos e máscaras de ação;
- prevenção de alocações inviáveis;
- balanceamento por congestionamento;
- reutilização de VNFs compartilháveis;
- preservação de recursos para requisições futuras.

A principal contribuição específica do Hybrid é integrar essas preocupações em uma única política PPO com ação discreta-contínua.

## 7. Resultados observados

Os resultados abaixo foram calculados a partir dos CSVs atuais em `results/results_flows`, usando uma execução disponível de cada algoritmo.

| Algoritmo | Aceitação final | Latência média | Tempo de decisão | Uso de banda |
|---|---:|---:|---:|---:|
| Hybrid | 93,80% | 11,84 ms | 30,51 ms | 4,12% |
| MSF | 92,84% | 4,81 ms | 4,92 ms | 12,65% |
| VEGETA | 92,71% | 6,65 ms | 3,21 ms | 0,70% |
| MUSFiCO | 90,04% | 3,57 ms | 11,03 ms | 9,27% |
| GreedyB | 77,82% | 3,41 ms | 9,30 ms | 4,22% |

Os valores de banda são frações no CSV e foram convertidos para porcentagem nesta tabela.

## 8. Interpretação dos resultados

### 8.1. Pontos fortes

#### Aceitação equilibrada

O Hybrid apresentou 93,8% de aceitação final. Além disso, aceitou 100% das requisições de IoT, Streaming e VoIP na tabela por serviço. Em MUAR, obteve 92,59%.

Esse comportamento indica robustez diante de uma mistura heterogênea de serviços.

#### Baixa pressão de banda em comparação com alguns algoritmos

O Hybrid utilizou aproximadamente 4,12% da capacidade total de banda, abaixo do MSF, com 12,65%, e do MUSFiCO, com 9,27%.

Entretanto, o VEGETA foi superior nesse indicador, com aproximadamente 0,70%. Portanto, o resultado correto é que o Hybrid reduziu significativamente a pressão de banda, mas não foi o melhor algoritmo nesse critério.

#### Cumprimento dos SLAs

As latências médias por serviço do Hybrid ficaram abaixo dos SLAs usados no ambiente:

| Serviço | Latência média | SLA |
|---|---:|---:|
| MUAR | 14,11 ms | 1000 ms |
| Streaming | 1,92 ms | 150 ms |
| VoIP | 0,006 ms | 60 ms |
| IoT | 0,086 ms | 200 ms |

O algoritmo cumpriu os limites de SLA, mas isso não significa que tenha minimizado a latência.

### 8.2. Limitações observadas

#### Latência média elevada

O Hybrid apresentou a maior latência média entre os algoritmos comparados, com 11,84 ms.

Isso é coerente com a ação `alpha`: valores menores de alocação de CPU reduzem o consumo de recursos, mas aumentam o tempo de processamento. A recompensa atual também prioriza aceitação, confiabilidade e preservação de recursos, não apenas a menor latência.

#### Tempo de decisão elevado

O tempo médio de decisão foi aproximadamente 30,51 ms, superior aos demais algoritmos.

Esse custo decorre da combinação de:

- GCN;
- LSTM;
- política categórica;
- distribuição Beta;
- avaliação de custos;
- cálculo de congestionamento;
- cálculo de confiabilidade;
- PPO customizado.

Esse valor representa o custo computacional da tomada de decisão e não deve ser confundido com a latência do serviço na rede.

#### CPU economizada não foi o melhor resultado

O Hybrid não foi o algoritmo com maior economia de CPU em todos os indicadores. Isso é esperado porque ele prioriza um compromisso entre recursos, aceitação, banda e confiabilidade.

## 9. Conclusão científica

Os resultados validam parcialmente a proposta do Hybrid.

O algoritmo demonstrou:

- boa taxa geral de aceitação;
- melhor equilíbrio entre tipos de serviço;
- baixa pressão de banda em relação ao MSF e ao MUSFiCO;
- cumprimento dos SLAs observados;
- capacidade de considerar topologia, sequência da SFC e alocação contínua de CPU.

Por outro lado, ele ainda não demonstrou:

- menor latência média;
- menor tempo de decisão;
- menor uso de banda em comparação com todos os algoritmos;
- superioridade estatística definitiva.

A conclusão recomendada é:

> O Hybrid é um algoritmo multiobjetivo de alocação de SFCs que prioriza robustez, aceitação heterogênea e controle de recursos. Ele reduz a pressão de banda e mantém os serviços dentro dos SLAs, mas ainda apresenta um custo elevado de decisão e uma latência média maior. O principal compromisso observado é entre economia de CPU, preservação de banda e latência de processamento.

## 10. Limitações experimentais

Os CSVs atuais correspondem a execuções individuais e não necessariamente utilizam exatamente a mesma sequência de requisições para todos os algoritmos. A quantidade de requisições por serviço também varia entre execuções.

Para uma comparação estatística final, recomenda-se:

1. usar a mesma topologia para todos os algoritmos;
2. usar a mesma sequência de requisições;
3. fixar as mesmas sementes aleatórias;
4. executar cada algoritmo várias vezes;
5. calcular média, desvio padrão e intervalo de confiança;
6. registrar explicitamente o custo de banda por caminho, número de saltos e valores de `alpha`.

## 11. Como reproduzir os gráficos

Os gráficos padronizados podem ser regenerados com:

```bash
uv run python scripts/gerar_graficos_slides.py
```

Os arquivos são salvos em:

```text
slides/
```

O manifesto dos arquivos utilizados fica em:

```text
slides/manifesto_graficos.csv
```

## 12. Referências utilizadas

1. Wang, W.; Chen, S.; Zhang, P.; Liu, K. *Reinforcement-Learning-Assisted Service Function Chain Embedding Algorithm in Edge Computing Networks*. Electronics, 2024, 13, 3007. DOI: [10.3390/electronics13153007](https://doi.org/10.3390/electronics13153007).
2. Zhao, T.; Tian, B.; Wang, L.; Ma, W.; Wei, B. *Intelligent Service Chain Orchestration and Resource Allocation in End-Edge Collaborative IIoT Using Multi-Agent Proximal Policy Optimization*. Sensors, 2026, 26, 3583. DOI: [10.3390/s26113583](https://doi.org/10.3390/s26113583).
3. Wu, Y.-J.; Hwang, S.-H.; Hwang, W.-S.; Cheng, M.-H. *A Deep Reinforcement Learning-Based Approach for Bandwidth-Aware Service Function Chaining*. Electronics, 2026, 15, 227. DOI: [10.3390/electronics15010227](https://doi.org/10.3390/electronics15010227).
