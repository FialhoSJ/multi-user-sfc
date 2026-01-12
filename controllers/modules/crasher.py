import random
import math
from typing import List, Dict

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

    def calculate_node_probabilities(self, network) -> Dict:
        """
        Calcula a probabilidade de falha (Peso da Roleta).
        Fórmula: P(Falha) = (1 - R_base) + (Stress * Stress_Factor)
        """
        physical_servers = {}
        
        # 1. Agregação (Nó Lógico -> Servidor Físico)
        for node in network.graph.nodes():
            node_data = network.graph.nodes[node]
            if 'server' not in str(node_data.get('type', '')):
                continue

            # Agrupa irmãos (Ex: 33 e 33.1)
            node_str = str(node)
            base_id = node_str.split('.1')[0] 
            
            level = node_data.get('level_server', 'default')
            cpu_used = network.get_node_cpu_used(node)
            
            # Tenta pegar capacidade original se estiver down
            cpu_cap = network.get_node_cpu_capacity(node)
            if cpu_cap <= 0:
                cpu_cap = node_data.get('original_cpu_capacity', 100.0)

            if base_id not in physical_servers:
                physical_servers[base_id] = {
                    'total_used': 0.0, 
                    'total_cap': 0.0, 
                    'level': level,
                    'members': []
                }
            
            physical_servers[base_id]['total_used'] += cpu_used
            physical_servers[base_id]['total_cap'] += cpu_cap
            physical_servers[base_id]['members'].append(node)

        # 2. Cálculo dos Pesos
        aggregated_probs = {}
        
        for base_id, stats in physical_servers.items():
            # A. Confiabilidade Base (Ex: 0.99)
            base_reliability = self._get_base_reliability(stats['level'])
            
            # B. Taxa de Uso (0.0 a 1.0)
            utilization_ratio = 0.0
            if stats['total_cap'] > 0:
                utilization_ratio = stats['total_used'] / stats['total_cap']
            
            # C. Penalidade por Estresse (Dinâmica via argumento)
            stress_penalty = utilization_ratio * self.alpha_stress
            
            # D. Confiabilidade Final (Ex: 0.99 - (1.0 * 0.05) = 0.94)
            final_reliability = base_reliability - stress_penalty
            if final_reliability < 0: final_reliability = 0.0

            # E. Probabilidade de Falha (Inverso) -> Peso da Roleta
            prob_failure = 1.0 - final_reliability
            
            aggregated_probs[base_id] = {
                'prob': prob_failure,
                'members': stats['members']
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