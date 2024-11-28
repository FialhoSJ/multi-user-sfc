from controllers.modules.mobility_manager import MobilityManager
from controllers.modules.sfcs_manager import SFCManager


class ResourceManager:
    def __init__(self):
        self.mobility_manager = MobilityManager()
        self.sfc_manager = SFCManager()
        self.network_nodes = []
        self.network_edges = []
        self.node_resources = {}

    def initialize_network(self, nodes, edges):
        self.network_nodes = nodes
        self.network_edges = edges
        for node in nodes:
            self.node_resources[node] = {
                "cpu": 100,
                "memory": 100,
                "bandwidth": 100
            }

    def allocate_resources(self, sfc_id):
        sfc = self.sfc_manager.get_sfc_by_id(sfc_id)
        if sfc and sfc["status"] == "pending":
            # Simulação de alocação de recursos
            for node in self.network_nodes:
                if self.node_resources[node]["cpu"] > self.sfc_manager.cpu_threshold * 100:
                    sfc["assigned_node"] = node
                    self.sfc_manager.deploy_sfc(sfc_id)
                    return True
        return False

    def monitor_network(self):
        for node, resources in self.node_resources.items():
            print(f"Node {node}: {resources}")

    def handle_user_movement(self, user_id, new_position):
        self.mobility_manager.update_positions(user_id, new_position)
        if self.mobility_manager.check_relocation(user_id):
            print(f"User {user_id} requires resource reallocation due to movement.")
