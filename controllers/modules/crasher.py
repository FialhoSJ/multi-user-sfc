import re 
import time
class Crasher():
    """Simulates network node failures based on specified modes and probabilities."""
    
    def __init__(self,edge_servers,edges_vnf,time=30,crash_links=False):
        self.time = time
        self.ec_servers = edge_servers
        self.edges_vnf = edges_vnf
        self.crash_links = False
        self.trials = 0
        self.nodes_crashed = []
        #self.a_server_was_crashed = 0
        #self.server_to_reroute = None
        #self.links_to_crash = []
        
    def activate_crasher(self, network):
        """Activates the crasher on the given network, excluding specific nodes."""
        nodes_info = {}
        
        node_choose = None
        lowest_rel = 1
        for node in self.ec_servers:
            rel = network.get_node_reliability(node)
            nodes_info[node] = rel
            if rel < lowest_rel:
                lowest_rel = rel
                node_choose = node
        
        if self.crash_links:
            nodes_to_crash = [node_choose]
            
            # Coleta todas as edges (conexões) da rede
            edges = network.sfs_flux_info.keys()
            
            # Lista para armazenar as edges do servidor escolhido
            edges_to_crash = []
            
            for edge in edges:
                # Verifica se o node escolhido (node_choose) está na edge
                if node_choose in edge:
                    # Identifica o servidor parceiro na edge
                    server_par = edge[0] if node_choose != edge[0] else edge[1]
                    
                    # Se o servidor parceiro não estiver na lista de servidores de edges ou for um valor específico, adicione-o
                    if server_par not in self.ec_servers and server_par != 0 and server_par != 34:
                        nodes_to_crash.append(server_par)

                    edges_to_crash.append(edge)  # Armazena a edge na lista de edges a serem derrubadas
            # Remove duplicatas da lista de nós a serem derrubados
            nodes_to_crash = list(set(nodes_to_crash))
            self.nodes_crashed = nodes_to_crash
            return nodes_to_crash
        else:
            self.nodes_crashed = [node_choose]
            return [node_choose]

    def implement_crash(self,nodes_crashed,network):
        sfcs_crashed = {}
        if len(nodes_crashed) != 0: 
            print(f"Servidores Crashados: {nodes_crashed}")
            self.sfcs_crashed = {}
            sfc_ids = []
            sfcs_to_crash = []

            for server in nodes_crashed:
                server_info = network.get_node_sfc_vnf_list(server)
                filtered_edges = {key: value for key, value in self.edges_vnf.items() if server in key}

                network.set_node_cache_capacity(server, -0.0000001)
                network.set_node_cpu_capacity(server, -0.0000001)

                if self.crash_links:
                    for link, sfc_vnf in filtered_edges.items():
                        network.set_link_bandwidth_capacity(link[0], link[1], -0.0000001)
                        network.set_link_latency(link[0], link[1], 10000)
                    
                if server_info != []:
                    sfc_ids = list(set([sfc[0] for sfc in server_info]))
                    users_crashed = []
                    pattern = re.compile(r'p\d+_\d+')

                    for sfc_id in sfc_ids:
                        # Popula o dicionário sfcs_crashed
                        match = pattern.search(sfc_id)
                        if match:
                            users_crashed.append(match.group())
                    users_crashed = list(set(users_crashed))
                    #self.users_crashed = self.users_crashed + len(users_crashed)

                    for sfc_id in sfc_ids:
                        # if sfc_id in sfc_id_duration:
                        #     # TODO: fazer alguma verificação pra garantir que aquela sfc está rodando
                        #     pass
                        #try:
                        
                        if sfc_id not in network.sfc_dict:
                            continue
                        
                        sfc = network.get_sfc_by_id(sfc_id)
                        sfc_rf = network.sfc_route_info[sfc_id]

                        latency_sfc= sum((len(value) - 1) for key, value in sfc_rf.items() if key not in ('src', 'dst'))
                        #duration = self.sfc_id_duration[sfc_id]

                        sfcs_crashed[sfc_id] = {'fall_time':time.time(),'has_backup':False,'old_latency':latency_sfc}
        return sfcs_crashed