import random
import math
from typing import List, Dict
from core.net_v2 import Net2

class Crasher:
    """
    Gerencia a simulação de falhas baseada em Níveis de Confiabilidade (Reliability).
    Matemática: Probabilidade de Falha = (1 - Confiabilidade Base) + (Stress * Stress_Factor)
    """

    def __init__(self, topology, args, interval=200, time=30):
        # Controle de Ativação (Macro)
        self.activated = (float(args.ava) != 1.0 or float(args.link_ava) != 1.0)
        
        # Leitura do Fator de Estresse (Micro)
        self.alpha_stress = args.stress_factor
        if self.alpha_stress < 0:
            print(f"[WARNING] Fator de estresse negativo ({self.alpha_stress}) inválido. Ajustando para 0.0.")
            self.alpha_stress = 0.0

        # Mapeamento de Tiers (Micro)
        self.tier_reliability = {
            'c': self._validate_reliability(args.rel_high, "High Level"),
            'b': self._validate_reliability(args.rel_normal, "Normal Level"),
            'a': self._validate_reliability(args.rel_low, "Low Level"),
            'default': self._validate_reliability(args.rel_normal, "Default")
        }

        self.ec_servers = topology.get_topology_info()['ec_servers']
        self.nodes_crashed = []
        self.links_crashed_history = []
        
        # Passo de simulação
        self.simulation_step = 1.0 

    def _validate_reliability(self, value: float, name: str) -> float:
        """Garante que a confiabilidade esteja entre 0.0 e 1.0"""
        if value < 0.0 or value > 1.0:
            print(f"[WARNING] Confiabilidade inválida para {name}: {value}. Ajustando para 0.99.")
            return 0.99
        return value

    def _get_base_reliability(self, level: str) -> float:
        return self.tier_reliability.get(str(level), self.tier_reliability['default'])

    def calculate_node_probabilities(self, network: Net2) -> Dict:
        """
        Calcula a probabilidade de falha baseada na lógica de confiabilidade do Net2.
        P(Falha) = 1 - Confiabilidade
        """
        physical_servers = {}
        
        # 1. Agregação por Servidor Físico (Assume IDs de nó como '33' e '33.1')
        for node in network.graph.nodes():
            node_data = network.graph.nodes[node]
            # Considera apenas servidores (ignora switches/roteadores para este cálculo se necessário)
            if 'server' not in str(node_data.get('type', '')):
                continue

            node_str = str(node)
            base_id = node_str.split('.1')[0] # Agrupa GPU (.1) com CPU
            
            if base_id not in physical_servers:
                physical_servers[base_id] = []
            physical_servers[base_id].append(node)

        aggregated_probs = {}
        
        for base_id, members in physical_servers.items():
            # Usa a confiabilidade do nó principal (geralmente o nó de CPU representa o chassi)
            # Se base_id for o próprio ID do nó, usa-o.
            main_node = base_id 
            
            reliability = network.get_node_reliability(int(main_node))
            prob_failure = 1.0 - reliability
            
            aggregated_probs[base_id] = {
                'prob': prob_failure,
                'members': members
            }
            
        return aggregated_probs

    def activate_crasher(self, network, sfc_manager=None, alg_name=None) -> List:
        """Executa a roleta usando as probabilidades calculadas."""
        if not self.activated:
            return []

        server_groups = self.calculate_node_probabilities(network)
        
        candidates = []
        weights = []

        for base_id, info in server_groups.items():
            # Filtra válidos (é servidor de borda?) e ativos (não caiu ainda?)
            is_valid = any(m in self.ec_servers for m in info['members'])
            is_active = any(m not in self.nodes_crashed for m in info['members'])

            if is_valid and is_active:
                candidates.append(base_id)
                weights.append(info['prob'])

        nodes_affected = []
        
        # Gira a Roleta
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