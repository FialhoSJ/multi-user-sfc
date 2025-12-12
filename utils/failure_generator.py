import random
import logging
from typing import List, Tuple

# Configuração básica de log para exibir avisos no console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calcular_janelas_falha(
    duracao_simulacao: float,
    num_falhas: int,
    duracao_minima_falha: float,
    confiabilidade: float
) -> List[Tuple[float, float]]:
    """
    Gera uma lista cronológica de janelas de falha (início, duração).
    
    A função garante que a soma das durações das falhas corresponda ao orçamento
    de 'downtime' definido pela confiabilidade, respeitando a duração mínima de cada falha.

    Args:
        duracao_simulacao (float): Tempo total da simulação em segundos.
        num_falhas (int): Quantidade de eventos de falha a serem agendados.
        duracao_minima_falha (float): Tempo mínimo (em segundos) que uma falha deve durar.
        confiabilidade (float): Índice de confiabilidade (0.0 a 1.0). 
                                Ex: 0.95 significa 95% do tempo sem falhas agregadas.

    Returns:
        List[Tuple[float, float]]: Uma lista de tuplas ordenadas pelo tempo de início.
                                   Formato: [(inicio_s, duracao_s), ...]

    Raises:
        ValueError: Se os parâmetros forem inconsistentes (ex: confiabilidade exige
                    menos tempo de falha do que a soma dos mínimos).
    """
    
    # 1. Validações de Guarda (Fail Fast)
    if num_falhas <= 0:
        # logger.warning("Solicitado 0 falhas. Retornando lista vazia.")
        return []
    
    if not (0.0 <= confiabilidade <= 1.0):
        raise ValueError(f"Confiabilidade deve estar entre 0 e 1. Recebido: {confiabilidade}")

    # 2. Cálculo do Orçamento de Downtime
    # Total de tempo que o sistema "pode" falhar
    tempo_total_downtime = duracao_simulacao * (1.0 - confiabilidade)
    
    # Custo mínimo obrigatório (Hard Constraint)
    custo_minimo_necessario = num_falhas * duracao_minima_falha
    
    # Verificação de Viabilidade
    if tempo_total_downtime < custo_minimo_necessario:
        msg = (
            f"Conflito de Parâmetros: A confiabilidade ({confiabilidade}) permite apenas "
            f"{tempo_total_downtime:.2f}s de falha, mas {num_falhas} falhas de "
            f"{duracao_minima_falha}s exigem no mínimo {custo_minimo_necessario:.2f}s."
        )
        # Opção A: Levantar erro (mais seguro para garantir a matemática)
        # raise ValueError(msg)
        
        # Opção B: Emitir aviso e ajustar o budget para o mínimo (mais resiliente para a simulação não parar)
        logger.warning(f"{msg} Ajustando budget para o mínimo necessário.")
        tempo_total_downtime = custo_minimo_necessario

    # 3. Distribuição do Excedente (Stick Breaking)
    # Quanto tempo sobra para distribuir aleatoriamente além do mínimo?
    excedente = tempo_total_downtime - custo_minimo_necessario
    
    # Gerar pesos aleatórios para distribuir o excedente
    pesos_aleatorios = [random.random() for _ in range(num_falhas)]
    soma_pesos = sum(pesos_aleatorios)
    
    cronograma: List[Tuple[float, float]] = []

    for i in range(num_falhas):
        # Evita divisão por zero se o gerador de random for muito peculiar (raro)
        peso_normalizado = (pesos_aleatorios[i] / soma_pesos) if soma_pesos > 0 else (1 / num_falhas)
        
        # Duração = Mínimo Obrigatório + Fatia do Excedente
        duracao_evento = duracao_minima_falha + (peso_normalizado * excedente)
        duracao_evento = round(duracao_evento, 2)
        
        # Definir Início Aleatório
        # O evento deve começar e terminar dentro da simulação.
        # Janela válida: [0, Duração_Simulação - Duração_Evento]
        janela_maxima_inicio = duracao_simulacao - duracao_evento
        
        if janela_maxima_inicio < 0:
            # Fallback caso a falha seja maior que a simulação inteira
            inicio_evento = 0.0
            duracao_evento = duracao_simulacao
        else:
            inicio_evento = random.uniform(0, janela_maxima_inicio)
            
        inicio_evento = round(inicio_evento, 2)
        
        cronograma.append((inicio_evento, duracao_evento))

    # 4. Ordenação e Retorno
    # Ordenar pelo tempo de início facilita o processamento sequencial na simulação
    cronograma.sort(key=lambda x: x[0])
    
    return cronograma
