import math
import random

class Crasher():
    """Simulates network node failures based on specified modes and probabilities."""
    
    def __init__(self):
        """
        Initializes the NetworkCrasher with configuration parameters.
        
        :param crash_probability: Probability of a crash happening in mode 1.
        :param operation_mode: Mode of operation (1 or 2) dictating how crashes occur.
        :param crash_limit: Optional limit to the number of nodes that can be crashed.
        """
        # self.crash_probability = crash_probability
        # self.servers_to_crash =  servers_to_crash
        # self.operation_mode = operation_mode
        # self.crash_limit = crash_limit
        # self.time_interval = time_interval
        # self.processing_nodes = processing_nodes
        self.processing_node_crashed = -1
        self.trials = 0
        self.a_server_was_crashed = 0
        self.server_to_reroute = None
        self.links_to_crash = []
#     def activate_crasher(self, network,tracer_topology):
#         """Activates the crasher on the given network, excluding specific nodes."""
        
#         # Isso permite que no modo de operação 4 a queda seja feita somente na primeira tentativa
#         if self.operation_mode == 4 and self.trials >= 1:
#             return []
        
#         network_nodes = self.processing_nodes
#         nodes_resource = network._node

#         # Filtra os servidores válidos e coleta seu uso de CPU
#         servidores_validos = [
#             (server, info['cpu_used'])
#             for server, info in nodes_resource.items()
#             if server not in [0, 34] and server in network_nodes
#         ]

#         # Ordena os servidores pelo uso de CPU (em ordem decrescente)
#         servidores_validos.sort(key=lambda x: x[1], reverse=True)

#         # Seleciona os 5 servidores mais usados
#         top_servidores = [server for server, _ in servidores_validos[1:4]]
#         server_choice = servidores_validos[0][0] #random.choice(top_servidores) if top_servidores else random.choice(network_nodes)
#         edges = network.sfs_flux_info.keys()

#         servers_to_crash = [server_choice]
#         self.processing_node_crashed = server_choice
        
#         for edge in edges:
#             if server_choice in edge:
#                 server_par = edge[0] if server_choice != edge[0] else edge[1]
#                 if server_par not in network_nodes and server_par != 0 and server_par != 34:
#                     servers_to_crash.append(server_par)
#         self.servers_to_crash = list(set(servers_to_crash))
       
#         servers_to_reroute = []

#         excluded_servers = self.servers_to_crash
#         crashed_position = tracer_topology[self.processing_node_crashed]

#         # Dicionário para armazenar os 5 servidores mais próximos e suas distâncias
#         nearest_servers = []

#         # Calcula a distância de todos os servidores
#         for server_id, position in tracer_topology.items():
#             if (server_id not in excluded_servers and server_id in self.processing_nodes and server_id != 0 and server_id !=34):
#                 distance = math.sqrt((crashed_position[0] - position[0]) ** 2 + 
#                                     (crashed_position[1] - position[1]) ** 2)
#                 nearest_servers.append((server_id, distance))

#         # Ordena os servidores pela distância e pega os 5 mais próximos
#         nearest_servers.sort(key=lambda x: x[1])
#         nearest_servers = nearest_servers[:3]  # Pegue os 5 mais próximos

#         edges = list(network.sfs_flux_info.keys())

#         # Percorre os 5 servidores mais próximos
#         for nearest_server_id, _ in nearest_servers:
#             options = []
#             location = nearest_server_id
#             for edge in edges:
#                 # Verifica se o 'location' está na primeira ou segunda posição da tupla
#                 if location in edge:
#                     connected_server = edge[0] if edge[1] == location else edge[1]
#                     if connected_server not in self.processing_nodes and connected_server != 34:
#                         options.append(connected_server)
#             # Verifica se há opções de servidores edge conectados e pega o primeiro
#             if options:
#                 options = list(set(options))
#                 servers_to_reroute.append(options[0])
#         self.server_to_reroute = servers_to_reroute


#         server_to_check = server_choice  # Servidor cujo os links serão buscados
#         edges = list(network.sfs_flux_info.keys())  # Lista de todas as tuplas/links
#         links_to_crash = []

#         # Percorre a lista de edges para encontrar todos os links associados ao servidor
#         for edge in edges:
#             # Verifica se o servidor está em qualquer uma das posições da tupla
#             if server_to_check in edge:
#                 links_to_crash.append(edge)
#         self.links_to_crash = links_to_crash
#         # self.server_to_reroute = location


#         # Só faz sentido verificar a quantidade de nós crashados se for no modo 1    
#         # if len(self.crashed_nodes) >= self.crash_limit or (self.trials == 0 and self.operation_mode != 3):
#         #if len(self.crashed_nodes) >= self.crash_limit or (self.trials == 0):
#             # self.trials = self.trials + 1 
#             # return []

#         nodes_crashed = []
#         # Execute crash simulation based on the configured operation mode.
#         if self.operation_mode == 1:
#             nodes_crashed = self._run_mode_1(network_nodes)
#         elif self.operation_mode == 2:
#             nodes_crashed = self._run_mode_2(network_nodes)
#         elif self.operation_mode == 3:
#             nodes_crashed = self._run_mode_3(network_nodes)
#         elif self.operation_mode == 4:
#             nodes_crashed = self._run_definitive_mode(network_nodes)

#         self.trials = self.trials + 1
#         return nodes_crashed

#     def _run_mode_1(self, nodes):
#         """Simulates crashes in mode 1 based on the crash probability."""
#         nodes_crashed = []        

        
#         servers = list(nodes)
#         crashed_servers = self.get_crashed_nodes()  
#         # Servidores disponíveis
#         available_servers =  list(set(servers) - set(crashed_servers))

#         crashed_now = []  # Lista para armazenar servidores que falharam no intervalo atual

#         for server in available_servers:  # Itera sobre uma cópia da lista de servidores disponíveis
#             if random.random() < self.crash_probability:
#                 if server not in crashed_servers:
#                     crashed_now.append(server)
#                     self._crash_node(server)
#                     self.a_server_was_crashed = 1

#         return crashed_now
    
#     def _run_mode_2(self, nodes):
#         """Randomly selects and crashes a single node in mode 2."""
#         chosen_node = random.choice(nodes)
        
#         while chosen_node in self.servers_to_crash:
#             chosen_node = random.choice(nodes)
#             print("tentando outro derrubar outro servidor, pois esse ja foi pra vala") # Para debug
            
#         self._crash_node(chosen_node)
#         self.a_server_was_crashed = 1
#         return [chosen_node]
    
#     def _run_mode_3(self, nodes):
#         """Selects and crashes a single node in mode 3."""
#         #if len(self.crashed_nodes) >1:
#         for server in self.servers_to_crash:
#             self._crash_node(server)
#             self.a_server_was_crashed = 1
#         return self.servers_to_crash
    
#     def _run_definitive_mode(self,nodes):
#         """Selects and crashes a single node in mode 3."""
#         #if len(self.crashed_nodes) >1:
#         for server in self.servers_to_crash:
#             self._crash_node(server)
#             self.a_server_was_crashed = 1
#         return self.servers_to_crash
# #         chosen_node = -1
# #         for server in self.servers_to_crash:
# #             if server not in self.crashed_nodes:
# #                 chosen_node = server
# #                 break
        
# #         if chosen_node  != -1:
# #             self._crash_node(chosen_node)
# #             self.a_server_was_crashed = 1
# #             return [chosen_node]
# #         else:
# #             return []

#     def _crash_node(self, node):
#         """Marks a node as crashed if it hasn't already been marked."""
#         if node in self.servers_to_crash:
#             # self.crashed_nodes.append(node)
#             print(f"Node {node} has crashed.")
#             print()

#     def get_crashed_nodes(self):
#         """Returns a list of crashed nodes."""
#         return self.servers_to_crash