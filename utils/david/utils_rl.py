import re

def subtrair_valor_padrao(texto_original: str) -> str:
    """
    Encontra o padrão "p{num1}_{num2}" em uma string e subtrai 6 de num2.

    Args:
        texto_original: A string de entrada.

    Returns:
        A string modificada.
    """
    # O padrão regex para encontrar "p", seguido por dígitos, um underscore, e mais dígitos.
    # Os parênteses ( ) criam "grupos de captura" para os números.
    # r"p(\d+)_(\d+)"
    #   p         -> Encontra o caractere literal 'p'
    #   (\d+)     -> Grupo 1: Encontra e captura um ou mais dígitos (o primeiro número)
    #   _         -> Encontra o caractere literal '_'
    #   (\d+)     -> Grupo 2: Encontra e captura um ou mais dígitos (o segundo número)
    pattern = r"p(\d+)_(\d+)"

    # A função re.sub pode aceitar uma outra função (ou uma expressão lambda)
    # para determinar pelo que substituir o padrão encontrado.
    # 'match' é um objeto que contém as partes capturadas pelo padrão.
    # match.group(1) é o primeiro número (como string)
    # match.group(2) é o segundo número (como string)
    texto_modificado = re.sub(
        pattern,
        lambda match: f"p{match.group(1)}_{int(match.group(2)) - 6}",
        texto_original
    )

    return texto_modificado