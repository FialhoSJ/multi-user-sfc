import networkx as nx
import math
import copy 
import random
import numpy as np

def calculate_5g_latency(
    data,
    distancia_m=750,
    potencia_transmissao_dbm=30.0,
    largura_banda_hz=100e6,
    temperatura_kelvin=290,
    figura_ruido_db=10.0,
    eficiencia_codec=0.5,
    snr_minimo_db=0.0,
    freq_portadora_hz=3.5e9,
    sigma_shadowing_db=0#6.00, #8.00
):
    """
    Calcula latência (ms) para uma dada distância em 5G, considerando path loss com shadowing.

    Parâmetro:
    - distancia_m: distância em metros (float ou lista/tupla de floats)

    Retorna latência em ms (float ou lista de floats, conforme input)
    """
    BOLTZMANN = 1.380649e-23

    # Função de perda de caminho com shadowing
    def path_loss_5g(distancia_m):
        pl_db = 28.0 + 22 * math.log10(distancia_m) + 20 * math.log10(freq_portadora_hz / 1e9)
        pl_db += random.gauss(0, sigma_shadowing_db)
        return 10 ** (-pl_db / 10)  # ganho linear

    def calcular_latencia_um_ponto(dado):
        ganho = path_loss_5g(distancia_m)
        potencia_w = 10 ** (potencia_transmissao_dbm / 10) / 1000
        ruido_w_hz = BOLTZMANN * temperatura_kelvin * (10 ** (figura_ruido_db / 10))
        snr_linear = (ganho * potencia_w) / (ruido_w_hz * largura_banda_hz)
        snr_linear = max(snr_linear, 10 ** (snr_minimo_db / 10))
        taxa_bps = largura_banda_hz * math.log2(1 + snr_linear) * eficiencia_codec
        latencia_ms = (dado / taxa_bps) * 1000  
        return latencia_ms
    return calcular_latencia_um_ponto(data)

def calculate_computational_latency(graph,node,vnf):
    ips = graph.nodes[node]['ips']
    packet = vnf.get_income_interface_bandwidth()/60 * 1e6
    return packet * 10 * 1000/ips

def calculate_latency_betwen_nodes(graph,node1,node2,vnf):            
    data_packet = (vnf.get_outcome_interface_bandwidth()/60)*1e6 
    if is_mobile_node(node1):
        return calculate_5g_latency(data_packet,graph.nodes[node1]['position'])  
    elif is_mobile_node(node2):
        return calculate_5g_latency(data_packet,graph.nodes[node2]['position'])  
    else:
        return get_link_latency(graph,node1,node2)
    
def is_mobile_node(node):
    if isinstance(node,str):
        return True
    else:
        return False

def pre_get_single_source_minimum_latency_path(graph):
    """Pre-calculate the shortest paths for all nodes in the network based on latency."""
    single_source_minimum_latency_path = {}
    for node in graph.nodes():
        single_source_minimum_latency_path[node] = \
            nx.single_source_dijkstra(graph, source=node, cutoff=None, weight='latency')
    return single_source_minimum_latency_path

def add_node_to_graph(graph, node_id, node_attributes):
    """Add a node to the graph with specific attributes."""
    graph.add_node(node_id, **node_attributes)

def add_edge_to_graph(graph, node1, node2, bandwidth_capacity=500, latency=1):
    """Add an edge between two nodes in the graph with bandwidth and latency."""
    graph.add_edge(node1, node2, bandwidth_capacity=bandwidth_capacity, latency=latency)

def get_shortest_path_length(graph, source, target):
    """Get the shortest path length from the source node to the target node based on latency."""
    try:
        return nx.dijkstra_path_length(graph, source, target, weight='latency')
    except nx.NetworkXNoPath:
        return float('inf')

def get_shortest_path(graph, source, target):
    """Get the shortest path from the source node to the target node based on latency."""
    try:
        return nx.dijkstra_path(graph, source, target, weight='latency')
    except nx.NetworkXNoPath:
        return []

import networkx as nx

def get_available_shortest_path(graph, source, target, bandwidth_required, rounded=False, latencia_saltos=False):
    """
    Obtém o caminho mais curto entre source e target minimizando latência,
    considerando apenas arestas com banda disponível >= bandwidth_required.
    """
    try:
        # Cria subgrafo com nós do grafo original e apenas arestas com banda suficiente
        subgraph = nx.Graph()
        subgraph.add_nodes_from(graph.nodes(data=True))
        for u, v, d in graph.edges(data=True):
            if d.get('bandwidth_capacity', 0) - d.get('bandwidth_used', 0) >= bandwidth_required:
                subgraph.add_edge(u, v, **d)

        # Executa Dijkstra no subgrafo filtrado
        if rounded:
            return nx.dijkstra_path(subgraph, source, target, weight=latency_rounded)
        elif latencia_saltos:
            return nx.dijkstra_path(subgraph, source, target)  # peso padrão: 1 por salto
        else:
            return nx.dijkstra_path(subgraph, source, target, weight='latency')
    except nx.NetworkXNoPath:
        # Ocorre se não houver caminho com peso finito entre source e target
        return []
    except nx.NodeNotFound:
        return []
    

import networkx as nx
import numpy as np # Importe o numpy para usar o infinito

def get_available_shortest_path_fast(graph, source, target, bandwidth_required, rounded=False, latencia_saltos=False):
    """
    Obtém o caminho mais curto entre source e target de forma eficiente,
    usando uma função de peso que "poda" links sem banda durante a busca do Dijkstra.
    """

    def weight_func(u, v, d):
        """
        Função de peso customizada para o Dijkstra.
        'd' é o dicionário de atributos da aresta (edge).
        """
        edge_data = graph.edges[u, v]
        available_bw = edge_data.get('bandwidth_capacity', 0) - edge_data.get('bandwidth_used', 0)

        # Se a banda for insuficiente, o custo deste link é "infinito",
        # fazendo com que o Dijkstra o evite.
        if available_bw < bandwidth_required:
            return np.inf

        # Se a banda for suficiente, retorna o peso de latência desejado.
        if rounded:
            # Você precisaria definir a função latency_rounded em algum lugar
            # return latency_rounded(u, v, d) 
            # Assumindo 'latency' por enquanto
            return edge_data.get('latency', 1)
        elif latencia_saltos:
            return 1 # Peso padrão de 1 por salto
        else:
            return edge_data.get('latency', 1)

    try:
        # Executa o Dijkstra no grafo original, mas com a lógica de poda
        # embutida na função de peso.
        path = nx.dijkstra_path(graph, source, target, weight=weight_func)
        return path
    except nx.NetworkXNoPath:
        # Esta exceção agora é levantada se todos os caminhos possíveis
        # tiverem um link com peso infinito (sem banda).
        return []
    except nx.NodeNotFound:
        return []


def get_available_shortest_path_optimized(graph, source, target, bandwidth_required, rounded=False, latencia_saltos=False):
    """
    Obtém o caminho mais curto de forma otimizada, usando uma função de peso
    para desconsiderar arestas sem banda disponível.
    """

    def weight_function(u, v, d):
        """
        Função de peso para o Dijkstra. Retorna o peso real se a banda for
        suficiente, ou infinito caso contrário.
        """
        # Verifica se a aresta tem banda suficiente
        if d.get('bandwidth_capacity', 0) - d.get('bandwidth_used', 0) >= bandwidth_required:
            # Se sim, retorna o peso apropriado baseado nos parâmetros da função principal
            if rounded:
                # Supondo que latency_rounded() seja uma função externa, como no seu código original
                return latency_rounded(u, v, d) 
            elif latencia_saltos:
                return 1  # Peso padrão para contar saltos
            else:
                return d.get('latency', 1) # Usa o atributo de latência
        
        # Se não tem banda, retorna um valor infinito para que Dijkstra ignore esta aresta
        return math.inf

    try:
        # Executa Dijkstra DIRETAMENTE no grafo original usando a função de peso customizada
        return nx.dijkstra_path(graph, source, target, weight=weight_function)
    
    except nx.NetworkXNoPath:
        # Ocorre se não houver caminho com peso finito entre source e target
        return []
    except nx.NodeNotFound:
        return []
    

def latency_rounded(u,v,data):
    return int(round(data['latency'],3)*1000)
    
def get_link_latency(graph, node1, node2):
    """Get the latency of the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['latency']

def calcular_latencia_total(caminho, graph):
    """
    Calcula a latência total de um caminho dado entre os nós, somando as latências das arestas,
    utilizando a função get_link_latency já existente.

    Parâmetros:
    - caminho: uma lista de nós que formam o caminho
    - graph: o grafo onde as arestas possuem o atributo 'latency'

    Retorna:
    - latência total do caminho em milissegundos (float)
    """
    if len(caminho) < 2:
        return 0  # Se o caminho tem menos de dois nós, não há latência

    latencia_total = 0
    for i in range(len(caminho) - 1):
        node1, node2 = caminho[i], caminho[i + 1]
        
        # Usando a função get_link_latency para obter a latência da aresta entre node1 e node2
        latencia_total += get_link_latency(graph, node1, node2)
    
    return latencia_total
def get_node_cpu_used(graph, node_id):
    """Get the CPU used by a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cpu_used']

def get_node_cpu_free(graph, node_id):
    """Get the free CPU capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cpu_capacity'] - graph.nodes[node_id]['cpu_used']

def get_node_cpu_capacity(graph, node_id):
    """Get the total CPU capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cpu_capacity']

def get_node_cache_used(graph, node_id):
    """Get the cache used by a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cache_used']

def get_node_cache_free(graph, node_id):
    """Get the free cache capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cache_capacity'] - graph.nodes[node_id]['cache_used']

def get_node_cache_capacity(graph, node_id):
    """Get the total cache capacity of a node."""
    if node_id not in graph:
        raise ValueError(f"Nó {node_id} não existe na topologia.")
    return graph.nodes[node_id]['cache_capacity']

def get_link_bandwidth_used(graph, node1, node2):
    """Get the bandwidth used by the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['bandwidth_used']

def get_link_bandwidth_free(graph, node1, node2):
    """Get the free bandwidth capacity of the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['bandwidth_capacity'] - graph.edges[node1, node2]['bandwidth_used']

def get_link_bandwidth_capacity(graph, node1, node2):
    """Get the total bandwidth capacity of the link between two nodes."""
    if not graph.has_edge(node1, node2):
        raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
    return graph.edges[node1, node2]['bandwidth_capacity']


def calcular_energia_computacional_mobile(
    d_fk_in_mbits: float,
    x_fk_u: int,
    delta_v_comp : float = 2.5*10**-9,
    omega_cycles_per_mbit: float = 10**6
) -> float:
   
    # x_fk_u: variavel binaria que controla a presença ou não da SF especifica
    # omega_cycles: ciclos por megabits, 10^6 no tcc
    # d_fk_in_mbits: quantidade de mbits de input da sf analisada

    total_ciclos_cpu = omega_cycles_per_mbit * d_fk_in_mbits  


    energia_base = total_ciclos_cpu * delta_v_comp  # [cite: 170]

    E_fk_v_comp = x_fk_u * energia_base

    return E_fk_v_comp


def calcular_energia_movel_total(
    lista_sfs: list,
    omega_cycles_per_mbit: float = 10**6
) -> float:

    total_energia_movel = 0.0

    # Extrai os parâmetros fixos do dispositivo
    delta_u_comp = 2.5e-9
    delta_u_comm = 2.6
    P_u = 30

    # O somatório principal (SUM_fk_in_Fc) é implementado como um loop
    for sf in lista_sfs:
        d_in = sf['d_in_mbits']
        x_u = sf['x_u']
        x_a = sf['x_a']
        R_ua = sf.get('R_ua', 1.0)  # Pega R_ua, ou 1.0 para evitar divisão por zero

        # --- Componente 1: Energia Computacional Local (E_fk,u^comp) ---
        # Reutiliza a função da Eq. 6
        # Nota: passamos delta_u_comp (eficiência do móvel)
        comp_1_energia_local = calcular_energia_computacional_mobile(
            d_fk_in_mbits=d_in,
            delta_v_comp=delta_u_comp,
            x_fk_u=x_u,
            omega_cycles_per_mbit=omega_cycles_per_mbit
        )


        comp_2_energia_transmissao = 0.0
        if x_a == 1 and R_ua > 0:
            tempo_transmissao = d_in / R_ua  # (d_fk,in / R_u,a)
            # delta_u^comm * (tempo) * P_u
            comp_2_energia_transmissao = delta_u_comm * tempo_transmissao * P_u

        # Soma os dois componentes para esta SF específica
        total_energia_movel += comp_1_energia_local + comp_2_energia_transmissao

    return total_energia_movel