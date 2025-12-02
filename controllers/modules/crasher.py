import random
import math
import re 

from numpy import copy
class Crasher():
    """Simulates network node failures based on specified modes and probabilities."""
    
    def __init__(self,topology,args,interval=200,time=30):
        self.time = time
        self.activated = (float(args.ava) != 1.0)
        self.number_of_fails = 0 if not (float(args.ava) != 1.0) else int(args.number_of_fails)
        self.availability = float(args.ava)
        self.ec_servers = topology.get_topology_info()['ec_servers']
        self.edges_vnf = {key: [] for key in topology.get_topology_info()['edges']}
        self.crash_links = False
        self.trials = 0
        self.nodes_crashed = []
        self.cluster_to_crash = []
        self.fail_interval = interval
        
        
        #self.a_server_was_crashed = 0
        #self.server_to_reroute = None
        #self.links_to_crash = []

    def cluster_method(self,network):
        node_choose = None
        edges = network.sfs_flux_info.keys()
        highest_consume = 0
        for node in self.ec_servers:
            edges_partners = [node]
            for edge in edges:
                if node in edge:
                    # Identifica o servidor parceiro na edge
                    server_par = edge[0] if node_choose != edge[0] else edge[1]
                    if server_par in self.ec_servers and server_par != 0:
                        edges_partners.append(server_par)  # Armazena a edge na lista de edges a serem derrubadas
            
            # Remove duplicatas da lista de nós a serem derrubados
            edges_partners = list(set(edges_partners))
            consumo = 0
            for edge_server in edges_partners:
                # if node != edge_server:
                cpu_used = network.get_node_cpu_used(edge_server)
                consumo = consumo + cpu_used
            if  highest_consume < consumo:
                highest_consume = consumo
                node_choose     = node
        return node_choose

    def most_sf_type(self,network,sf_type='EC'):
        node_choose = None
        max_sfc = -1
        max_cpu_cache_usage = -1
        for node in self.ec_servers:
            unique_sfc_set = set()  # Usar um conjunto para rastrear SFCS únicas no nó
            for sfc_vnf in network.get_node_sfc_vnf_list(node):
                sfc_type = sfc_vnf[0].split("_")[1]
                vnf_id = sfc_vnf[1].id.split("_")[0]
                if (vnf_id == sf_type):
                    unique_sfc_set.add(sfc_vnf[0])  # Adiciona ao conjunto (evita duplicados)
            
            unique_sfc_count = len(unique_sfc_set)  # Conta as SFCs únicas
            cpu_used = network.get_node_cpu_used(node)  # Obtém o uso de CPU do nó
            cache_used = network.get_node_cache_used(node)  # Obtém o uso de cache do nó
            cpu_cache_usage = cpu_used + cache_used  # Soma para considerar o uso total de recursos
            
            # Atualiza o nó com mais SFCs únicas ou desempata com base no uso de CPU e cache
            if (unique_sfc_count > max_sfc): #or (unique_sfc_count == max_sfc and cpu_cache_usage > max_cpu_cache_usage):
                max_sfc = unique_sfc_count
                max_cpu_cache_usage = cpu_cache_usage
                node_choose = node
        return node_choose

    def highest_resource_consumer(self, network):
        """
        Seleciona o nó com o maior consumo de recursos (CPU e cache) na rede.
        
        Args:
            network: Objeto que representa a rede.
            
        Returns:
            node_choose: O nó que consome mais recursos.
        """
        node_choose = None
        max_resource_usage = -1  # Variável para rastrear o maior consumo de recursos
        
        for node in self.ec_servers:
            # Obtém o consumo de CPU e cache para o nó
            cpu_used   = network.get_node_cpu_used(node)  
            cache_used = network.get_node_cache_used(node) 
            
            # if cpu_used > 100 or cache_used > 100:
            #     continue
            
            total_resource_usage = cpu_used + cache_used  # Soma dos recursos usados
            
            # Atualiza o nó com maior consumo de recursos
            if total_resource_usage > max_resource_usage:
                max_resource_usage = total_resource_usage
                node_choose = node
        return node_choose
    
    def crash_cluster(self, network):
        node_choose = None
        edges = network.sfs_flux_info.keys()
        highest_consume = 0
        if self.cluster_to_crash == []:
            for node in self.ec_servers:
                edges_partners = [node]
                for edge in edges:
                    if node in edge:
                        # Identifica o servidor parceiro na edge
                        server_par = edge[0] if node_choose != edge[0] else edge[1]
                        if server_par in self.ec_servers and server_par != 0:
                            edges_partners.append(server_par)  # Armazena a edge na lista de edges a serem derrubadas

                # Remove duplicatas da lista de nós a serem derrubados
                edges_partners = list(set(edges_partners))
                consumo = 0
                for edge_server in edges_partners:
                    # if node != edge_server:
                    cpu_used = network.get_node_cpu_used(edge_server)
                    consumo = consumo + cpu_used
                if  highest_consume < consumo:
                    highest_consume = consumo
                    node_choose     = node

            edges_from_cluster = []
            for edge in edges:
                if node in edge:
                    # Identifica o servidor parceiro na edge
                    server_par = edge[0] if node_choose != edge[0] else edge[1]
                    if server_par in self.ec_servers and server_par != 0:
                        edges_from_cluster.append(server_par)  # Armazena a edge na lista de edges a serem derrubadas

            edges_from_cluster = list(set(edges_from_cluster))
            self.cluster_to_crash = edges_partners
        
        node_choose = None
        for node in self.cluster_to_crash:
            node_choose = self.cluster_to_crash.pop(0) 
        return node_choose


    def update_node_rel(self, network):
        nodes_rel = network.nodes_reliability
        updated_rel = {}

        for node, rel in nodes_rel.items():
            time = 0.01 # padrão por simplificação
            base_failure_rate = 1 - rel

            alpha_base = 1000
            alpha_cpu = 1
            alpha_mem = 1.5

            # TODO pode ser que diferenciar o uso de recurso de backup pra normal seja mais eficiente
            cpu_used =  network.graph.nodes[node]['cpu_used']
            cache_used =  network.graph.nodes[node]['cache_used']

            lambda_total = (alpha_base * base_failure_rate +
                            alpha_cpu * cpu_used +
                            alpha_mem * cache_used)
            
            reliability = math.exp(-lambda_total * time)
            p_falha = 1-reliability
            updated_rel[node] = p_falha
        return updated_rel

    def activate_crasher(self, network,sfc_manager,alg_name):
        """Activates the crasher on the given network, excluding specific nodes."""


        ###############################################################
        node_choose = None
        h_rel = 0
        
        nodes_rel = self.update_node_rel(network)
        for node,rel in nodes_rel.items():
            if rel > h_rel:
                h_rel = rel
                node_choose = node

        #reuse_quantity 
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
            nodes_to_crash = set()
            if node_choose is not None:
                all_network_nodes = set(network.graph.nodes())
                nodes_to_crash.add(node_choose)
                node_str = str(node_choose)

                try:
                    if '.1' in node_str:
                        # Nó de desempenho (ex: 33.1), encontrar o nó base (ex: 33)
                        base_node = int(node_str.split('.1')[0])
                        if base_node in all_network_nodes:
                            nodes_to_crash.add(base_node)
                    else:
                        # Nó base (ex: 33), encontrar o nó de desempenho (ex: 33.1)
                        counterpart_node_str = f"{node_str}.1"
                        counterpart_node = float(counterpart_node_str)
                        if counterpart_node in all_network_nodes:
                            nodes_to_crash.add(counterpart_node)
                except (ValueError, IndexError):
                    # Ignora se houver um erro de conversão com um nome de nó inesperado
                    pass
            
            final_nodes_list = list(nodes_to_crash)
            
            # Adiciona os nós à lista global de falhas, garantindo unicidade
            self.nodes_crashed.extend(final_nodes_list)
            self.nodes_crashed = list(set(self.nodes_crashed))

            return final_nodes_list

        ###############################################################
        # # #node_choose = 5
        # #     #print(rel)
        # node_choose = None
        # if alg_name not in ['vegeta','msf','g']:
        #     #node_choose = self.cluster_method(network=network)
        #     #node_choose = self.most_sf_type(network,sf_type='RE')
        #     node_choose = self.highest_resource_consumer(network)
        #     #node_choose = self.crash_cluster(network)

        # if alg_name in ['g']:
        #     node_choose = random.choice(self.ec_servers)
        #     while node_choose in [33,34,9]:
        #         node_choose = random.choice(self.ec_servers)

        # if alg_name in ['msf']:
        #     node_choose = None
        #     max_resource_usage = -1  # Variável para rastrear o maior consumo de recursos
            
        #     for node in [9,33,34]:
        #         # Obtém o consumo de CPU e cache para o nó
        #         cpu_used   = network.get_node_cpu_used(node)  
        #         total_resource_usage = cpu_used   # Soma dos recursos usados
        #         if total_resource_usage > max_resource_usage:
        #             max_resource_usage = total_resource_usage
        #             node_choose = node
        # Após o loop, `node_more_sfc` terá o nó com mais SFCs únicas
        # if node_choose == None:
        #     return False
        # if alg_name == 'ga':
                
        # sfcs_with_backup = list(sfc_manager.sfs_backup.keys())
        # server_backup_count = {}

        # # Itera sobre os servidores
        # node_more_cpu = None  # Nenhum servidor inicial selecionado
        # max_cpu_used = -1  # Valor inicial menor que qualquer possível uso de CPU

        # # Itera sobre os servidores
        # for node in self.ec_servers:
        #     cpu_used = network.get_node_cpu_used(node)
        #     # Verifica se este servidor tem mais CPU usada que o máximo atual
        #     if cpu_used > max_cpu_used:
        #         max_cpu_used = cpu_used
        #         node_more_cpu = node
        # # Define o servidor escolhido
        # node_choose = node_more_cpu

        # node_more_sfc = None  
        # max_sfc = -1  
        # for node in self.ec_servers:
        #     count_unique = []  # Lista para rastrear SFCS únicas no nó
        #     for sfc_vnf in network.get_node_sfc_vnf_list(node):
        #         sfc_type = sfc_vnf[0].split("_")[1]
        #         if sfc_type == 'unique':
        #             count_unique.append(sfc_vnf[0])  # Adiciona a SFC se não estiver na lista
            
            
        #sfcs_in_node = len(count_unique)  # Calcula o número de SFCS únicas no nó

        #     # Verifica se este nó tem mais SFCS únicas que o máximo atual
        #     if sfcs_in_node > max_sfc:
        #         max_sfc = sfcs_in_node
        #         node_more_sfc = node

        # Define o servidor escolhido
        # node_choose = node_more_sfc

    # def recover_from_crash(self, network):
    #     nodes_crashed = list(self.nodes_crashed)
    #     if len(self.nodes_crashed) != 0: 
    #         print(f"Recuperando servidor: {self.nodes_crashed}")
    #         for server in nodes_crashed:
    #             # Definir capacidades negativas para simular o crash
    #             network.set_node_cache_capacity(server, 100)
    #             network.set_node_cpu_capacity(server, 100)
    #     self.nodes_crashed = []

    def recover_from_crash(self, network):
        if self.nodes_crashed:  # Verifica se há servidores na lista
            server = self.nodes_crashed[0]  # Recupera o primeiro servidor
            print(f"Recuperando servidor: {server}")
            # Definir capacidades negativas para simular o crash
            # network.set_node_cache_capacity(server, 100)
            # network.set_node_cpu_capacity(server, 100)
            self.nodes_crashed.remove(server)  # Remove o servidor recuperado da lista
        return self.nodes_crashed
    
    
    
    
    def implement_crash(self, nodes_crashed, network,players_sfc_list):
        if len(nodes_crashed) != 0: 
            print(f"Servidores Crashados: {nodes_crashed}")
            sfc_ids = []

            for server in nodes_crashed:
                server_info = network.get_node_sfc_vnf_list(server)
                network.set_node_cache_capacity(server, -0.0000001)
                network.set_node_cpu_capacity(server, -0.0000001)
                
                if server_info != []:
                    # Extrai os sfc_ids
                    sfc_ids = list(set([sfc[0] for sfc in server_info]))
            sfcs_list = []
            for sfc_id in sfc_ids:
                new_sfc_list = self.find_sfc_pair_or_list(players_sfc_list,sfc_id)
                sfcs_list.append(new_sfc_list)
            unique_lists = [list(t) for t in set(tuple(sublist) for sublist in sfcs_list)]
            return unique_lists


    def find_sfc_pair_or_list(self,player_sfc_id_list, sfc_key):
        for sfc_list in player_sfc_id_list:
            if sfc_key in sfc_list:
                return sfc_list  # Retorna a lista onde a chave está presente
        return None  # Retorna None se a chave não for encontrada
