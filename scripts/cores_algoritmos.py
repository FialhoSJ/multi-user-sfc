"""Paleta padrão de cores por algoritmo (usada por TODOS os scripts de gráfico).

Garante que REPLIC/MSF/Kuririn/DARSPPO/Hephaestus tenham a MESMA cor em
box plots, barras e linhas, nos cenários padrão e heterogêneo.
"""
from __future__ import annotations

CORES_ALGORITMOS = {
    "replic": "#2E8B57",  # verde
    "msf": "#577590",  # azul-acinzentado
    "kuririn": "#F9844A",  # laranja
    "darsppo": "#6A4C93",  # roxo
    "hephaestus": "#E63946",  # vermelho
}

ROTULOS_CANONICOS = {
    "replic": "REPLIC",
    "msf": "MSF",
    "kuririn": "Kuririn",
    "darsppo": "DARSPPO",
    "hephaestus": "Hephaestus",
}

DEFAULT_CORES = ["#4D774E", "#577590", "#F9844A", "#90BE6D", "#6A4C93", "#E63946"]


def normalizar(label: str) -> str:
    """Normaliza um rótulo para a chave canônica (case-insensitive, tolera sufixos)."""
    chave = label.lower()
    for nome in CORES_ALGORITMOS:
        if nome in chave:
            return nome
    chave_so_letras = "".join(c for c in chave if c.isalpha())
    for nome in CORES_ALGORITMOS:
        if nome in chave_so_letras:
            return nome
    return chave


def cor_para(label: str, indice: int = 0) -> str:
    """Cor canônica do algoritmo; fallback para a paleta padrão por índice."""
    chave = normalizar(label)
    if chave in CORES_ALGORITMOS:
        return CORES_ALGORITMOS[chave]
    return DEFAULT_CORES[indice % len(DEFAULT_CORES)]


def rotulo_curto(label: str) -> str:
    """Rótulo de exibição curto do algoritmo (ex.: 'kuririnMaskablePPO' -> 'Kuririn')."""
    chave = normalizar(label)
    return ROTULOS_CANONICOS.get(chave, label)
