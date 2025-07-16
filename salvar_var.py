import os
import pickle

def salvar_variavel(variavel, nome_lista, pasta='variaveis_salvas', valor_unico=False):
    """
    Salva uma variável dentro de uma lista nomeada em arquivo .pkl.
    
    Parâmetros:
    - variavel: objeto Python a ser salvo
    - nome_lista: string com o nome da lista de variáveis (será o nome do arquivo)
    - pasta: diretório onde os arquivos serão armazenados
    - valor_unico: se True, impede que a lista tenha mais de um item, ou seja, não adiciona um novo valor
    """
    # Cria a pasta se não existir
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")

    # Carrega a lista existente se o arquivo já existir
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, 'rb') as f:
            lista = pickle.load(f)
    else:
        lista = []

    # Se valor_unico for True, não adiciona o novo valor se já houver algum item na lista
    if valor_unico:
        if len(lista) > 0:
            print("A lista já contém um valor. Nenhuma modificação foi feita.")
            return  # Não adiciona nada
        else:
            lista.append(variavel)  # Adiciona o valor se a lista estiver vazia
    else:
        # Adiciona a nova variável se ela não estiver já na lista
        if variavel not in lista:
            lista.append(variavel)

    # Salva a lista (ou o único elemento) de volta no arquivo
    with open(caminho_arquivo, 'wb') as f:
        pickle.dump(lista, f)

    print(f"Variável salva em {caminho_arquivo}. Total de itens: {len(lista)}.")


def carregar_lista(nome_lista, pasta='variaveis_salvas'):
    """
    Carrega a lista salva em um arquivo .pkl.
    
    Parâmetros:
    - nome_lista: string com o nome da lista de variáveis (nome do arquivo)
    - pasta: diretório onde os arquivos são armazenados
    
    Retorna:
    - Lista de variáveis ou uma lista vazia caso o arquivo não exista.
    """
    caminho_arquivo = os.path.join(pasta, f"{nome_lista}.pkl")
    
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, 'rb') as f:
            lista = pickle.load(f)
        # print(f"Lista carregada de {caminho_arquivo}. Total de itens: {len(lista)}.")
        return lista
    else:
        # print(f"Arquivo {caminho_arquivo} não encontrado.")
        return []  # Retorna uma lista vazia caso o arquivo não exista
