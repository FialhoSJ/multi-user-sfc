import random
import math
from typing import List, Dict
from core.net_v2 import Net2

class Crasher:
    """
    Gerencia a simulação de falhas baseada em Níveis de Confiabilidade (Reliability).
    A lógica de cálculo de estresse e Tiers é delegada para a classe Net2.
    """

    def __init__(self, topology, args, interval=200, time=30):
        # Controle de Ativação (Macro)
        self.activated = (float(args.ava) != 1.0 or float(args.link_ava) != 1.0)
        
        # Obtém informações da topologia (assumindo que o objeto topology possui este método)
        self.ec_servers = topology.get_topology_info()['ec_servers']
        
        self.nodes_crashed = []
        self.links_crashed_history = []
        
        # Passo de simulação
        self.simulation_step = 1.0 

    def calculate_node_probabilities(self, network: Net2) -> Dict:
        """
        Calcula a probabilidade de falha consultando a confiabilidade dinâmica
        do nó na rede (Net2).
        P(Falha) = 1 - Confiabilidade
        """
        physical_servers = {}
        
        # 1. Agregação por Servidor Físico (Assume IDs de nó como '33' e '33.1')
        for node in network.graph.nodes():
            node_data = network.graph.nodes[node]
            # Considera apenas servidores (ignora switches/roteadores para este cálculo)
            if 'server' not in str(node_data.get('type', '')):
                continue

            node_str = str(node)
            base_id = node_str.split('.1')[0] # Agrupa GPU (.1) com CPU
            
            if base_id not in physical_servers:
                physical_servers[base_id] = []
            physical_servers[base_id].append(node)

        aggregated_probs = {}
        
        for base_id, members in physical_servers.items():
            # Consulta a confiabilidade diretamente da rede
            # O Net2 já sabe lidar com Tiers e Estresse
            reliability = network.get_node_reliability(base_id)
            
            # Probabilidade de falha é o inverso da confiabilidade
            # max(0, ...) garante que não haja valores negativos por erro de arredondamento
            prob_failure = max(0.0, 1.0 - reliability)
            
            aggregated_probs[base_id] = {
                'prob': prob_failure,
                'members': members
            }
            
        return aggregated_probs

    def activate_crasher(self, network, sfc_manager=None, alg_name=None) -> List:
        """Executa a roleta usando as probabilidades reais da rede."""
        if not self.activated:
            return []

        server_groups = self.calculate_node_probabilities(network)
        
        candidates = []
        weights = []

        for base_id, info in server_groups.items():
            # Filtra válidos (é servidor de borda?) e ativos (não caiu ainda?)
            is_valid = any(m in self.ec_servers for m in info['members'])
            
            # Verifica se o servidor físico (ou seus componentes) já está na lista de caídos
            is_active = base_id not in self.nodes_crashed and \
                        any(m not in self.nodes_crashed for m in info['members'])

            if is_valid and is_active:
                candidates.append(base_id)
                weights.append(info['prob'])

        nodes_affected = []
        
        # Gira a Roleta se houver candidatos e risco real > 0
        if candidates and sum(weights) > 0:
            chosen_key = random.choices(candidates, weights=weights, k=1)[0]
            nodes_affected = server_groups[chosen_key]['members']
            
            for node in nodes_affected:
                if node not in self.nodes_crashed:
                    self.nodes_crashed.append(node)
        
        return nodes_affected

    def recover_specific_node(self, network, node_id) -> bool:
        """Recupera um nó específico solicitado pelo Controller."""
        if node_id in self.nodes_crashed:
            network.restore_node(node_id)
            self.nodes_crashed.remove(node_id)
            return True
        return False

    def activate_link_crasher(self, network):
        # Implementação opcional de links (não solicitada alteração, retorna None para segurança)
        return None