# Proactive Mobility-Aware SFC Migration in Edge Networks

## Plano de artigo e implementação

## 1. Título provisório

**Proactive Mobility-Aware SFC Migration in Edge Networks: Balancing Service Continuity and Migration Cost**

## 2. Ideia central

Investigar quando é vantajoso migrar uma Service Function Chain (SFC) de forma proativa, antes que o usuário móvel complete um handover, em vez de esperar a mudança de roteador para iniciar uma migração reativa.

A política proposta deve equilibrar:

- continuidade do serviço;
- latência fim a fim;
- tempo de interrupção;
- número de migrações;
- banda usada na migração;
- consumo de CPU, cache e energia;
- carga do servidor de destino;
- risco de violação de SLA.

A contribuição científica não será apenas executar SFCs sobre uma trajetória móvel. O foco será modelar a decisão de migração como uma escolha entre o benefício esperado da antecipação e o custo de migrar uma cadeia que pode não ser mais necessária no destino previsto.

## 3. Motivação e problema

Em uma rede edge, o usuário pode se afastar do servidor que hospeda sua SFC. Isso aumenta a latência entre o dispositivo e as VNFs e pode causar interrupção durante o handover.

A migração proativa pode reduzir a latência e a interrupção, mas também pode:

- consumir banda e recursos computacionais;
- gerar migrações desnecessárias quando a previsão estiver errada;
- sobrecarregar o próximo servidor edge;
- aumentar o consumo energético;
- causar indisponibilidade durante a transferência do estado da SFC.

### Pergunta principal

> Dado um usuário móvel e sua SFC, quando e para onde migrar a cadeia para reduzir a interrupção sem gerar custo excessivo?

### Perguntas secundárias

1. Em quais velocidades a migração proativa supera a migração reativa?
2. Qual é o custo de uma previsão incorreta?
3. Um limiar adaptativo é melhor que o limiar fixo de handover?
4. Em que condições a migração parcial seria preferível à migração completa?
5. Como a carga do servidor de destino altera a decisão de migrar?

## 4. Hipóteses

### H1 - Continuidade

A migração proativa reduz a latência e o tempo de interrupção em comparação com a migração reativa.

### H2 - Custo

A migração proativa reduz a interrupção, mas aumenta o consumo de banda, recursos e energia.

### H3 - Política adaptativa

Uma política que considera benefício e custo apresenta melhor compromisso entre continuidade e custo do que uma política baseada somente na distância ou no handover.

### H4 - Velocidade

A vantagem da migração proativa aumenta com a velocidade do usuário e com a frequência de handovers.

### H5 - Previsibilidade

A migração proativa é mais eficiente em trajetórias previsíveis e pode gerar migrações desnecessárias em trajetórias imprevisíveis.

## 5. Escopo inicial

Para manter o primeiro estudo controlável, a versão inicial deve utilizar:

- uma topologia edge;
- uma classe homogênea de SFC;
- usuários móveis simulados pelo SUMO;
- uma SFC por usuário;
- migração completa da cadeia;
- falhas da infraestrutura desativadas na primeira etapa;
- política centralizada;
- predição baseada em posição, velocidade e direção;
- várias sementes aleatórias.

A heterogeneidade de serviços, falhas, backups e migração parcial podem ser adicionados em uma segunda etapa ou como trabalho futuro.

## 6. Modelo do sistema

### 6.1. Componentes

- usuários móveis simulados pelo SUMO/TraCI;
- roteadores de acesso;
- servidores edge;
- VNFs que compõem a SFC;
- enlaces com capacidade, banda utilizada e latência;
- controlador central de rede;
- módulo de previsão de mobilidade;
- política de decisão de migração;
- módulo de coleta de métricas.

### 6.2. Estado do usuário

Para cada usuário `u`, o sistema deve acompanhar:

```text
posição atual
posição anterior
velocidade
 direção
roteador conectado
próximo roteador previsto
tempo estimado até handover
distância ao servidor atual
última migração
número de migrações
estado da migração
```

### 6.3. Estimativa de movimento

A posição do usuário no instante `t` pode ser representada por:

$$
p_u(t) = (x_u(t), y_u(t))
$$

A velocidade estimada é:

$$
v_u(t) = \frac{p_u(t) - p_u(t-\Delta t)}{\Delta t}
$$

Uma previsão cinemática simples para um horizonte `tau` é:

$$
\hat{p}_u(t+\tau) = p_u(t) + v_u(t)\tau
$$

O próximo roteador pode ser obtido selecionando o roteador mais próximo da posição prevista.

O preditor simples deve ser implementado antes de técnicas de aprendizado de máquina. Isso permite validar o protocolo e estabelecer um baseline interpretável.

## 7. Políticas de comparação

### 7.1. Sem migração

A SFC permanece no local original durante todo o experimento.

Essa política mede o custo de não adaptar a alocação à mobilidade.

### 7.2. Migração reativa

A migração acontece somente depois que o usuário muda de roteador ou quando o novo roteador apresenta uma melhoria de distância acima de um limiar.

Essa política representa o comportamento atual de handover/redeploy do simulador.

### 7.3. Migração proativa simples

A migração é iniciada quando o próximo roteador previsto é diferente do roteador atual e o tempo estimado até o handover está abaixo de um limiar.

```text
se tempo_estimado_ate_handover < limiar:
    iniciar migração
```

### 7.4. Migração proativa adaptativa

A política compara o benefício esperado com o custo de migração:

$$
B_{latencia} + B_{SLA} + B_{interrupcao}
>
C_{migracao} + C_{banda} + C_{recursos} + C_{energia}
$$

Uma função de utilidade possível é:

$$
U = w_1\Delta L + w_2\Delta I + w_3\Delta SLA - w_4C_m - w_5C_b - w_6C_e
$$

Onde:

- `Delta L` é a redução esperada de latência;
- `Delta I` é a redução esperada de interrupção;
- `Delta SLA` é a redução esperada de violações de SLA;
- `C_m` é o custo de migração;
- `C_b` é o custo de banda;
- `C_e` é o custo energético;
- `w_i` são os pesos da política.

A migração deve ocorrer somente quando `U > 0` e o destino possuir recursos suficientes.

## 8. Alterações necessárias no simulador

### 8.1. Corrigir o contrato de mobilidade

Existe uma inconsistência a verificar:

- `SimulationSettings` define `mobility` como booleano;
- `MobilityManager` verifica `args.mobility == "y"`.

O contrato deve ser padronizado para booleano, por exemplo:

```python
self.activated = bool(args.mobility)
```

Depois, deve ser criado um teste ou log que confirme que o SUMO foi iniciado quando `mobility=True`.

### 8.2. Estender o `MobilityManager`

O `players_tracker` deve armazenar, além do roteador conectado:

- histórico curto de posições;
- velocidade estimada;
- direção estimada;
- roteador previsto;
- tempo estimado até handover;
- timestamp da última decisão;
- timestamp da última migração;
- estado atual da migração;
- contador de migrações necessárias e desnecessárias.

### 8.3. Criar o preditor

A primeira versão deve:

1. coletar posições consecutivas do veículo;
2. calcular velocidade e direção;
3. projetar a posição futura;
4. determinar o roteador previsto;
5. estimar o tempo até o handover;
6. informar a confiança ou o erro da previsão.

Possíveis extensões:

- média móvel;
- regressão linear;
- Random Forest;
- LSTM;
- uso direto da rota conhecida pelo SUMO.

### 8.4. Modelar o custo de migração

Uma formulação inicial é:

$$
C_{mig} = \alpha D_{state} + \beta H_{path} + \gamma T_{downtime} + \delta R_{resource}
$$

Onde:

- `D_state` é o volume de estado transferido;
- `H_path` é o número de saltos entre origem e destino;
- `T_downtime` é o tempo de indisponibilidade;
- `R_resource` é o recurso reservado no destino.

O custo pode ser aproximado com base em:

- número de VNFs;
- CPU requisitada;
- cache requisitado;
- banda do caminho de migração;
- distância entre origem e destino;
- carga atual do destino.

### 8.5. Modelar estados de migração

A migração não deve ser tratada como uma troca instantânea. Utilize estados explícitos:

```text
STABLE
PREDICTED
MIGRATING
COMMITTING
COMPLETED
CANCELLED
FAILED
```

Durante a migração devem ser registrados latência, interrupção, banda transferida e recursos ocupados na origem e no destino.

### 8.6. Evitar migrações instáveis

A política deve incluir:

- tempo mínimo entre migrações;
- histerese;
- benefício mínimo para iniciar uma migração;
- cancelamento quando a previsão mudar;
- bloqueio de destinos congestionados;
- validação de capacidade antes da reserva.

Uma decisão pode ser bloqueada quando:

```text
tempo_desde_ultima_migracao < cooldown
ou beneficio_estimado < beneficio_minimo
ou capacidade_destino insuficiente
```

## 9. Integração arquitetural

O fluxo esperado é:

```text
SUMO atualiza a posição
        |
MobilityManager atualiza o estado
        |
Preditor estima o próximo roteador
        |
MigrationPolicy calcula benefício e custo
        |
Controlador decide manter, migrar ou cancelar
        |
SFCManager executa a migração
        |
OutputWriter registra as métricas
```

Principais pontos de integração:

- `MobilityManager`: posição, velocidade, previsão e handover;
- `SubstrateNetworkController`: ciclo periódico e chamada da política;
- `SFCManager`: remoção e redeploy da SFC;
- `SFCInstantiator`: nova alocação no destino;
- `OutputWriter`: métricas de mobilidade e migração;
- `BackupManager`: inicialmente desabilitado para não misturar os efeitos.

## 10. Métricas

### 10.1. Continuidade

- tempo total de interrupção;
- número de interrupções;
- duração média da interrupção;
- taxa de sessões sem interrupção;
- disponibilidade da SFC;
- violações de SLA.

### 10.2. Mobilidade

- número de handovers;
- distância percorrida;
- velocidade média;
- tempo entre handovers;
- erro de previsão do próximo roteador;
- taxa de previsões corretas.

### 10.3. Migração

- número total de migrações;
- migrações necessárias;
- migrações desnecessárias;
- migrações canceladas;
- migrações falhas;
- tempo médio de migração;
- volume de estado transferido.

### 10.4. Rede e recursos

- latência fim a fim;
- banda consumida;
- CPU utilizada;
- cache utilizado;
- energia;
- utilização dos servidores;
- congestionamento dos enlaces;
- taxa de aceitação.

### 10.5. Eficiência da migração

Uma métrica composta possível é:

$$
E_{migration} = \frac{reducao\ de\ interrupcao}{custo\ adicional\ de\ migracao}
$$

Essa métrica deve ser apresentada junto das métricas originais, não como substituta delas.

## 11. Cenários experimentais

### Cenário A - Velocidade

- baixa;
- média;
- alta.

### Cenário B - Densidade de usuários

- poucos usuários;
- densidade média;
- muitos usuários.

### Cenário C - Previsibilidade

- trajetórias previsíveis;
- mudanças frequentes de direção;
- mobilidade aleatória.

### Cenário D - Frequência de decisão

- verificação a cada 1 segundo;
- verificação a cada 5 segundos;
- verificação a cada 10 segundos.

### Cenário E - Capacidade da rede

- servidores folgados;
- servidores moderadamente carregados;
- servidores congestionados.

### Conjunto mínimo

| Cenário | Velocidade | Usuários | Políticas |
|---|---:|---:|---|
| S1 | baixa | baixa | todas |
| S2 | média | média | todas |
| S3 | alta | média | todas |
| S4 | alta | alta | todas |
| S5 | média | alta | todas |

Cada configuração deve ser executada com pelo menos 5 sementes. Para resultados de publicação, preferir 10 ou mais repetições quando o custo computacional permitir.

## 12. Experimentos de ablação

Remover componentes da política adaptativa para identificar a contribuição de cada um:

1. sem previsão;
2. previsão sem custo de migração;
3. previsão sem carga do destino;
4. previsão com limiar fixo;
5. previsão adaptativa completa;
6. política adaptativa sem cooldown;
7. política adaptativa sem histerese.

Uma conclusão desejável seria demonstrar que a previsão reduz interrupções, mas que o custo de migração, cooldown e carga do destino são necessários para evitar migrações excessivas.

## 13. Resultados esperados

O resultado mais interessante não precisa ser uma vitória em todas as métricas. É esperado um compromisso:

- **sem migração:** menor custo de adaptação, mas maior latência;
- **reativa:** menos migrações, mas maior interrupção;
- **proativa simples:** menor interrupção, porém mais migrações;
- **proativa adaptativa:** equilíbrio entre continuidade e custo.

Uma conclusão possível é:

> A migração proativa é vantajosa principalmente para usuários rápidos e trajetórias previsíveis. Em mobilidade imprevisível, a política adaptativa reduz migrações desnecessárias ao considerar o custo e a incerteza da previsão.

Essa conclusão deve ser confirmada pelos experimentos, não assumida antecipadamente.

## 14. Estrutura do artigo

### 1. Introdução

- crescimento do edge computing;
- mobilidade dos usuários;
- problema da localização fixa da SFC;
- limitação da migração reativa;
- proposta e contribuições.

### 2. Trabalhos relacionados

- embedding de SFC;
- mobilidade em edge computing;
- migração de VNFs;
- previsão de trajetória;
- otimização com aprendizado por reforço.

### 3. Modelo do sistema

- rede substrata;
- usuários e mobilidade;
- SFC;
- handover;
- migração;
- hipóteses do modelo.

### 4. Política proposta

- estado observado;
- preditor;
- estimativa de custo;
- função de utilidade;
- decisão de migração;
- mecanismos de cooldown e histerese.

### 5. Implementação

- SUMO/TraCI;
- simulador MUAR-SFC;
- controladores utilizados;
- parâmetros;
- baselines;
- protocolo reprodutível.

### 6. Avaliação

- cenários;
- métricas;
- número de sementes;
- intervalos de confiança;
- resultados;
- análise de trade-offs;
- ablação.

### 7. Limitações

- modelo abstrato de migração;
- preditor inicialmente baseado em velocidade;
- ausência inicial de tráfego real;
- controlador centralizado;
- ausência de migração real de contêineres e estado de aplicação.

### 8. Conclusão

- quando a migração proativa compensa;
- redução de interrupção;
- custo introduzido;
- cenários em que a política falha;
- extensões futuras.

## 15. Ordem de implementação

1. Corrigir e validar a ativação da mobilidade.
2. Registrar posição, velocidade e roteador atual.
3. Implementar o baseline sem migração.
4. Medir o comportamento reativo atual.
5. Implementar o preditor cinemático.
6. Implementar a migração proativa simples.
7. Adicionar custo de migração.
8. Adicionar cooldown e histerese.
9. Registrar métricas específicas de mobilidade.
10. Executar cenários com sementes controladas.
11. Executar os experimentos de ablação.
12. Analisar resultados e escrever o artigo.

## 16. Critérios de sucesso

O trabalho estará pronto para submissão quando:

- as quatro políticas forem executáveis no mesmo simulador;
- as trajetórias e sementes forem reproduzíveis;
- o custo de migração estiver explicitamente modelado;
- interrupção e latência forem medidas durante a migração;
- houver pelo menos 5 repetições por configuração;
- os resultados apresentarem média, dispersão e intervalo de confiança;
- a política adaptativa for comparada com baselines simples;
- os experimentos de ablação identificarem a contribuição de cada componente;
- as limitações do modelo forem descritas claramente.

## 17. Decisão metodológica recomendada

A primeira versão não deve começar com aprendizado por reforço. Uma política baseada em regras é mais fácil de validar, explicar e comparar.

Depois que o modelo de migração, os custos e as métricas estiverem estáveis, o Hybrid ou PPO pode ser usado para aprender os pesos ou substituir a função de decisão. Nesse caso, o aprendizado por reforço será uma extensão experimental apoiada por um protocolo já validado, e não a única fonte de comportamento do sistema.
