import random
import logging
from typing import List, Tuple

# Configuração básica de log
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
    de 'downtime' definido pela confiabilidade, respeitando a duração mínima.

    Args:
        duracao_simulacao (float): Tempo total da simulação em segundos.
        num_falhas (int): Quantidade de eventos de falha.
        duracao_minima_falha (float): Tempo mínimo de uma falha (em segundos).
        confiabilidade (float): Índice de confiabilidade (0.0 a 1.0).

    Returns:
        List[Tuple[float, float]]: Lista ordenada [(inicio_s, duracao_s), ...].
    """
    
    # ==========================================================================
    # 1. VALIDAÇÕES (Guarda)
    # ==========================================================================
    if num_falhas <= 0:
        return []
    
    if not (0.0 <= confiabilidade <= 1.0):
        raise ValueError(f"Confiabilidade deve estar entre 0 e 1. Recebido: {confiabilidade}")

    # ==========================================================================
    # 2. CÁLCULO DO ORÇAMENTO DE DOWNTIME
    # ==========================================================================
    tempo_total_downtime = duracao_simulacao * (1.0 - confiabilidade)
    custo_minimo_necessario = num_falhas * duracao_minima_falha
    
    # Verifica se o orçamento cobre o custo mínimo
    if tempo_total_downtime < custo_minimo_necessario:
        msg = (
            f"Conflito de Parâmetros: A confiabilidade ({confiabilidade}) permite apenas "
            f"{tempo_total_downtime:.2f}s de falha, mas {num_falhas} falhas exigem "
            f"no mínimo {custo_minimo_necessario:.2f}s."
        )
        logger.warning(f"{msg} Ajustando budget para o mínimo necessário.")
        tempo_total_downtime = custo_minimo_necessario

    # ==========================================================================
    # 3. PREPARAÇÃO DA DISTRIBUIÇÃO (Stick Breaking)
    # ==========================================================================
    excedente = tempo_total_downtime - custo_minimo_necessario
    
    # Gera pesos aleatórios para distribuir o tempo excedente
    pesos_aleatorios = [random.random() for _ in range(num_falhas)]
    soma_pesos = sum(pesos_aleatorios)
    
    cronograma: List[Tuple[float, float]] = []

    # ==========================================================================
    # 4. GERAÇÃO DOS EVENTOS
    # ==========================================================================
    for i in range(num_falhas):
        # Normalização do peso (evita divisão por zero)
        peso_normalizado = (pesos_aleatorios[i] / soma_pesos) if soma_pesos > 0 else (1 / num_falhas)
        
        # Define duração: Mínimo Obrigatório + Fatia do Excedente
        duracao_evento = duracao_minima_falha + (peso_normalizado * excedente)
        duracao_evento = round(duracao_evento, 2)
        
        # Define início aleatório
        # Janela válida: [0, Duração_Simulação - Duração_Evento]
        janela_maxima_inicio = duracao_simulacao - duracao_evento
        
        if janela_maxima_inicio < 0:
            # Fallback: falha maior que a simulação
            inicio_evento = 0.0
            duracao_evento = duracao_simulacao
        else:
            inicio_evento = random.uniform(0, janela_maxima_inicio)
            
        inicio_evento = round(inicio_evento, 2)
        cronograma.append((inicio_evento, duracao_evento))

    # ==========================================================================
    # 5. ORDENAÇÃO E RETORNO
    # ==========================================================================
    # Ordena cronologicamente pelo tempo de início
    cronograma.sort(key=lambda x: x[0])
    
    return cronograma