from core.net_v2 import Net2

class EnergyCalculator:
    """
    Objeto responsável por calcular o consumo de energia de uma rede com base
    nas regras do artigo e nas modificações solicitadas.
    """
    def __init__(self):
        """
        Inicializa o calculador com os dados de potência (em Watts) da Tabela 7 do artigo.
        A estrutura armazena os níveis de potência como 'low', 'medium' e 'high' para cada tipo de nó.
        """
        self._server_specs = {
            'a': {
                'low': 30,     # Frequência 2.0 GHz
                'medium': 40,  # Frequência 2.5 GHz
                'high': 50,    # Frequência 3.0 GHz
            },
            'b': {
                'low': 75,     # Frequência 4.0 GHz
                'medium': 100, # Frequência 5.0 GHz
                'high': 120,   # Frequência 6.5 GHz
            },
            'c': {
                'low': 150,    # Frequência 8.0 GHz
                'medium': 175, # Frequência 10.0 GHz
                'high': 220,   # Frequência 12.0 GHz
            }
        }

    def _get_power_for_node(self,graph ,node) -> float:
        """
        Calcula a potência de um único nó com base na sua utilização de CPU.
        Esta é a implementação da sua regra de negócio.
        """
        node_data = graph.nodes[node]
        if node_data['cpu_capacity'] == 0:
            return 0
        # Calcula a porcentagem de utilização da CPU
        cpu_utilization = node_data["cpu_used"] / node_data["cpu_capacity"]

        if node_data["type"] == "mobile_device":
            return 7.5

        # Seleciona o nível de potência com base na utilização
        node_type_specs = self._server_specs[node_data["level_server"]]

        

        if 0 <= cpu_utilization <= 0.33:
            # Uso baixo: até 33%, usa o menor valor da categoria
            return node_type_specs['low']
        elif 0.33 < cpu_utilization <= 0.67:
            # Uso médio: entre 33% e 67%, usa o valor do meio
            return node_type_specs['medium']
        else:
            # Uso alto: acima de 67%, usa o maior valor
            return node_type_specs['high']

    def calculate_total_network_power(self, network: Net2) -> float:
        """
        Calcula a potência total instantânea da rede (em Watts).

        Este método itera sobre todos os nós na rede, calcula a potência
        de cada um com base na sua carga de CPU atual e soma tudo.

        Args:
            network: Um objeto de rede que contém o atributo 'nodes'.

        Returns:
            A potência total consumida pela rede em Watts (Joules por segundo).
        """
        graph = network.graph
        total_power = 0.0
        for node in graph.nodes:
            power_for_node = self._get_power_for_node(graph,node)
            if node%1 == 0.1:
                power_for_node *= 1.2
            total_power += power_for_node
        return total_power


