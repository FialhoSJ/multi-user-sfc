import random
from typing import Dict, List, Optional, Any, Tuple, Union
import networkx as nx

class NetworkTopology:
    """Gerenciador da topologia da rede (Wrapper sobre NetworkX).
    
    Mantém dois grafos separados (Infraestrutura Fixa e Dispositivos Móveis)
    conforme design original, mas fornece uma interface unificada.
    """

    def __init__(self):
        self._infra_graph = nx.Graph()
        self._mobile_graph = nx.Graph()

    # =========================================================================
    # GERENCIAMENTO DE NÓS
    # =========================================================================

    def add_server_node(
        self, 
        node_id: Union[str, int], 
        node_type: str, 
        cpu_capacity: float = 0.0, 
        cache_capacity: float = 0.0, 
        ips: float = 0.0,
        position: Tuple[float, float] = (0, 0)
    ) -> None:
        """Adiciona um nó de servidor ou roteador à infraestrutura fixa.
        
        Args:
            node_id: Identificador único do nó.
            node_type: Tipo do nó (ex: 'server_default', 'router').
            cpu_capacity: Capacidade de CPU.
            cache_capacity: Capacidade de Cache.
            ips: Instruções por segundo (base). Será multiplicado por 10e10 internamente.
            position: Coordenadas (x, y).
        """
        # Extração de nível do servidor (ex: server_levelA -> levelA)
        clean_type = node_type.split("_")[0]
        server_level = node_type.split("_")[-1] if '_' in node_type else 'default'

        # Definição dos atributos base
        attrs = {
            'type': clean_type,
            'cpu_capacity': float(cpu_capacity),
            'cache_capacity': float(cache_capacity),
            'cpu_used': 0.00,
            'cache_used': 0.00,
            'ips': ips * 10e10,  # Mantendo a escala original do Net2
            'position': position,
            'reuse': [],         # Lista de VNFs reutilizáveis
            'services': {},      # Dicionário de serviços alocados
            'sfcs_list': [],     # Lista de SFCs passando pelo nó
            'level_server': server_level,
            'is_active': True,
            'original_cpu_capacity': float(cpu_capacity), # Para recuperação de falhas
            'original_cache_capacity': float(cache_capacity)
        }

        # Tratamento específico para Roteadores (Atributos Wireless)
        if clean_type == 'router':
            # Nota: w_channel_capacity não foi passado nos args originais explicitamente,
            # mas é usado. Assumindo 0.0 ou deve ser passado via **kwargs se necessário.
            attrs.update({
                'w_channel_capacity': 0.0, # Deveria vir dos argumentos se variável
                'w_channel_used': 0.0,
                'w_services': {}
            })

        self._infra_graph.add_node(node_id, **attrs)

    def add_mobile_node(
        self, 
        node_id: Union[str, int], 
        cpu_capacity: float, 
        cache_capacity: float, 
        ips: float = 0.0,
        position: Tuple[float, float] = (0, 0)
    ) -> None:
        """Adiciona um dispositivo móvel com variação estocástica de hardware.
        
        Args:
            node_id: ID do usuário/dispositivo.
            cpu_capacity: Capacidade base de CPU.
        """
        # Lógica de variação aleatória (Simulação de heterogeneidade de dispositivos)
        # Mantendo fiel ao original: 33% fraco, 33% normal, 33% forte
        sorteio = random.random()
        factor = 1.0
        
        if sorteio <= 0.33:
            factor = 0.75
        elif sorteio > 0.67:
            factor = 1.25

        final_cpu = cpu_capacity * factor
        final_cache = cache_capacity * factor
        final_ips = (ips * factor) * 10e10

        attrs = {
            'type': 'mobile_device',
            'cpu_capacity': final_cpu,
            'cache_capacity': final_cache,
            'cpu_used': 0.00,
            'cache_used': 0.00,
            'ips': final_ips,
            'position': position,
            'reuse': [],
            'services': {},
            'sfcs_list': [],
            'is_active': True
        }
        
        self._mobile_graph.add_node(node_id, **attrs)

    def remove_node(self, node_id: Any) -> None:
        """Remove um nó de qualquer um dos grafos (EAFP)."""
        if self._mobile_graph.has_node(node_id):
            self._mobile_graph.remove_node(node_id)
        elif self._infra_graph.has_node(node_id):
            self._infra_graph.remove_node(node_id)

    # =========================================================================
    # GERENCIAMENTO DE LINKS (ARESTAS)
    # =========================================================================

    def add_wired_link(
        self, 
        node1: Any, 
        node2: Any, 
        bandwidth: float = 1000.0, 
        latency: float = 1.0
    ) -> None:
        """Cria um link cabeado entre dois nós da infraestrutura."""
        if not self._infra_graph.has_node(node1) or not self._infra_graph.has_node(node2):
            raise ValueError(f"Nós {node1} ou {node2} não existem na infraestrutura para criar link.")

        self._infra_graph.add_edge(
            node1, 
            node2,
            bandwidth_capacity=float(bandwidth),
            bandwidth_used=0.00,
            bandwidth_reserved=0.00,
            latency=latency,
            services_in_transit={},
            original_bw=float(bandwidth), # Para falhas
            original_lat=latency
        )

    def add_wireless_link(
        self, 
        mobile_id: Any, 
        router_id: Any, 
        bandwidth: float = 100.0, 
        latency_base: float = 5.0
    ) -> None:
        """Conecta um dispositivo móvel a um roteador (Link Híbrido).
        
        Nota: No Net2 original, links wireless eram arestas no grafo principal?
        O código original fazia `graph.add_edge` mesmo para mobile. 
        Aqui, vamos assumir que se conecta ao grafo principal.
        """
        # Verifica existências
        if not self._mobile_graph.has_node(mobile_id):
            raise ValueError(f"Mobile Node {mobile_id} não encontrado.")
        if not self._infra_graph.has_node(router_id):
            raise ValueError(f"Router Node {router_id} não encontrado.")

        # ATENÇÃO: NetworkX não permite arestas entre grafos diferentes nativamente
        # se forem instâncias separadas.
        # Design Pattern Adapter: Vamos criar uma abstração lógica ou
        # adicionar o nó móvel ao grafo de infraestrutura APENAS para fins de roteamento?
        # Pelo código original, parecia haver uma mistura.
        #
        # SOLUÇÃO SEGURA: Mantemos a conexão lógica no grafo de infraestrutura para permitir Dijkstra.
        
        # Copia dados essenciais do mobile para o grafo principal para fins de roteamento
        # (Isso simula a conexão física na topologia global)
        mobile_data = self._mobile_graph.nodes[mobile_id]
        self._infra_graph.add_node(mobile_id, **mobile_data)
        
        self._infra_graph.add_edge(
            mobile_id, 
            router_id,
            bandwidth_capacity=bandwidth,
            bandwidth_used=0.0,
            latency=latency_base,
            type='wireless',
            services_in_transit={}
        )

    # =========================================================================
    # CONSULTAS (GETTERS)
    # =========================================================================

    def get_node_data(self, node_id: Any) -> Dict[str, Any]:
        """Retorna os atributos de um nó, buscando em ambos os grafos."""
        if self._infra_graph.has_node(node_id):
            return self._infra_graph.nodes[node_id]
        elif self._mobile_graph.has_node(node_id):
            return self._mobile_graph.nodes[node_id]
        else:
            raise KeyError(f"Nó {node_id} não encontrado na topologia.")

    def get_link_data(self, u: Any, v: Any) -> Dict[str, Any]:
        if self._infra_graph.has_edge(u, v):
            return self._infra_graph.edges[u, v]
        raise KeyError(f"Link {u}-{v} não encontrado.")

    def get_all_nodes(self) -> List[Any]:
        """Retorna todos os IDs de nós (Infra + Mobile)."""
        return list(self._infra_graph.nodes()) + list(self._mobile_graph.nodes())

    def check_link_exists(self, u: Any, v: Any) -> bool:
        return self._infra_graph.has_edge(u, v)

    def is_mobile(self, node_id: Any) -> bool:
        """Verifica se o nó é um dispositivo móvel original."""
        return self._mobile_graph.has_node(node_id)
    
    @property
    def graph(self) -> nx.Graph:
        """Retorna o grafo principal (Infra) para algoritmos de roteamento.
        
        Aviso: Use com cuidado. Idealmente os algoritmos deveriam pedir
        subgrafos específicos ou iteradores.
        """
        return self._infra_graph