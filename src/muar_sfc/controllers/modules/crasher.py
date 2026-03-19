import random

from muar_sfc.core.net_v2 import Net2


class Crasher:
    """
    Gerencia a simulação de falhas.
    MODIFICAÇÃO: Uso de conjuntos (set) para complexidade O(1) e remoção de verificações redundantes.
    """

    def __init__(self, topology, args, interval=200, time=30):
        self.activated = float(args.ava) != 1.0 or float(args.link_ava) != 1.0
        self.ec_servers = topology.get_topology_info()["ec_servers"]

        # --- REFATORAÇÃO: Set para acesso e remoção instantânea O(1) ---
        self.nodes_crashed = set()
        self.links_crashed_history = set()

        self.simulation_step = 1.0
        self.fail_target = args.fail_target

    def calculate_node_probabilities(self, network: Net2) -> dict:
        physical_servers = {}

        for node in network.graph.nodes():
            node_data = network.graph.nodes[node]
            if "server" not in str(node_data.get("type", "")):
                continue

            node_str = str(node)
            base_id = node_str.split(".1")[0]

            if base_id not in physical_servers:
                physical_servers[base_id] = []
            physical_servers[base_id].append(node)

        aggregated_probs = {}

        for base_id, members in physical_servers.items():
            reliability = network.get_node_reliability(base_id)
            prob_failure = max(0.0, 1.0 - reliability)

            aggregated_probs[base_id] = {
                "prob": prob_failure,
                "reliability": reliability,
                "members": members,
            }

        return aggregated_probs

    def activate_crasher(self, network, sfc_manager=None, alg_name=None) -> list:
        if not self.activated:
            return []

        # REFATORAÇÃO: Acesso EAFP direto, sem 'getattr' LBYL defasado
        user_target = self.fail_target
        server_groups = self.calculate_node_probabilities(network)

        candidates = []
        weights = []

        for base_id, info in server_groups.items():
            is_valid = any(m in self.ec_servers for m in info["members"])

            # Validação instantânea O(1) graças ao uso do 'set'
            is_active = base_id not in self.nodes_crashed and all(
                m not in self.nodes_crashed for m in info["members"]
            )

            if is_valid and is_active:
                reliability = info["reliability"]
                should_include = False

                if user_target == "all" or user_target == "high_risk" and reliability < 0.8666 or user_target == "med_risk" and 0.8666 <= reliability <= 0.9333 or user_target == "low_risk" and reliability > 0.9333:
                    should_include = True

                if not should_include:
                    continue

                candidates.append(base_id)
                weights.append(info["prob"])

        nodes_affected = []

        if candidates and sum(weights) > 0:
            chosen_key = random.choices(candidates, weights=weights, k=1)[0]
            nodes_affected = server_groups[chosen_key]["members"]
            confiabilidade_atual = server_groups[chosen_key]["reliability"]

            for node in nodes_affected:
                if node not in self.nodes_crashed:
                    self.nodes_crashed.add(node) # REFATORAÇÃO: O(1) Adição limpa
                    print(f"Node {node} CRASHED! Reliability: {confiabilidade_atual:.3f}")

        return nodes_affected

    def recover_specific_node(self, network, node_id) -> bool:
        """Recupera um nó específico solicitado pelo Controller."""
        if node_id in self.nodes_crashed:
            network.restore_node(node_id)
            self.nodes_crashed.remove(node_id) # REFATORAÇÃO: Remoção O(1) instantânea
            return True
        return False

    def activate_link_crasher(self, network):
        return None, None
