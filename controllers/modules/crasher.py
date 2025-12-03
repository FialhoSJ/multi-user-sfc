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
        """
        Calcula a probabilidade de falha AGRUPADA por servidor físico.
        Considera nós irmãos (ex: 33 e 33.1) como um único servidor.
        """
        # Dicionário para armazenar dados agregados: 
        # { 'base_id_str': {'cpu_used': 0, 'cpu_cap': 0, 'nodes': [], 'rel_base': 1.0} }
        physical_servers = {}
        
        # 1. Varre todos os nós para agrupar irmãos e somar recursos
        for node in network.graph.nodes():
            node_str = str(node)
            # Identifica o ID base (ex: '33.1' vira '33', '33' vira '33')
            base_id = node_str.split('.1')[0]
            
            # Coleta métricas do nó atual
            cpu_used = network.graph.nodes[node].get('cpu_used', 0)
            
            # Tenta pegar capacidade, default 100 se não existir
            try:
                cpu_cap = network.graph.nodes[node].get('cpu_capacity', 100)
                if cpu_cap <= 0: cpu_cap = 100
            except:
                cpu_cap = 100
                
            # Coleta confiabilidade base (network.nodes_reliability pode ter chave int ou str)
            # Assume que a confiabilidade base é igual para as partes, pega a do nó atual
            node_rel_base = network.nodes_reliability.get(node, 1.0)
            if node not in network.nodes_reliability:
                # Tenta converter para int se a chave for int no dicionário original
                try:
                    node_rel_base = network.nodes_reliability.get(int(node), 1.0)
                except:
                    pass

            # Inicializa o grupo se não existir
            if base_id not in physical_servers:
                physical_servers[base_id] = {
                    'cpu_used': 0, 
                    'cpu_cap': 0, 
                    'nodes': [], 
                    'base_failure_rate': (1 - node_rel_base) # Pega o inverso da confiabilidade (taxa de falha)
                }
            
            # Agrega os valores
            physical_servers[base_id]['cpu_used'] += cpu_used
            physical_servers[base_id]['cpu_cap'] += cpu_cap
            physical_servers[base_id]['nodes'].append(node)
            
            # Atualiza a taxa de falha base para ser a maior encontrada entre as partes (pior caso) ou média
            # Aqui mantemos a lógica simples: se uma parte tem rel ruim, o server tem rel ruim.
            current_rate = 1 - node_rel_base
            if current_rate > physical_servers[base_id]['base_failure_rate']:
                physical_servers[base_id]['base_failure_rate'] = current_rate

        # 2. Calcula a probabilidade final para cada grupo físico
        aggregated_probs = {} # { base_id: {'prob': 0.5, 'members': [33, 33.1]} }
        
        for base_id, stats in physical_servers.items():
            time = 0.01
            alpha_base = 1000
            alpha_cpu = 1 
            alpha_mem = 0 # Cache ignorado conforme pedido
            
            # Calcula a taxa de uso real do servidor físico (soma dos usos / soma das capacidades)
            total_used = stats['cpu_used']
            total_cap = stats['cpu_cap']
            
            cpu_stress_ratio = 0
            if total_cap > 0:
                cpu_stress_ratio = total_used / total_cap
            
            # Fórmula da taxa de falha (Lambda)
            lambda_total = (alpha_base * stats['base_failure_rate'] +
                            alpha_cpu * cpu_stress_ratio)
            
            # Converte taxa em probabilidade (Exponencial)
            p_falha = 1 - math.exp(-lambda_total * time)
            
            aggregated_probs[base_id] = {
                'prob': p_falha,
                'members': stats['nodes']
            }
            
        return aggregated_probs

    def activate_crasher(self, network, sfc_manager, alg_name):
        """
        Ativa falhas usando Seleção por Roleta em Servidores Físicos Agrupados.
        """
        node_choose_members = []
        
        # 1. Recebe as probabilidades já agrupadas por servidor físico
        # Retorno: { '33': {'prob': 0.X, 'members': [33, 33.1]}, ... }
        server_groups = self.update_node_rel(network)
        
        candidates_keys = []
        weights = []

        # 2. Filtra candidatos válidos (apenas servidores de borda)
        for base_id, info in server_groups.items():
            # Verifica se pelo menos uma parte do servidor está na lista de servidores permitidos (ec_servers)
            # A lista ec_servers geralmente tem inteiros, base_id é string aqui. Convertemos para verificar.
            is_valid_candidate = False
            for member in info['members']:
                if member in self.ec_servers:
                    is_valid_candidate = True
                    break
            
            if is_valid_candidate:
                candidates_keys.append(base_id)
                weights.append(info['prob'])

        # 3. Gira a Roleta
        if candidates_keys and sum(weights) > 0:
            # Sorteia uma chave (ex: '33') com base no peso (probabilidade de falha)
            chosen_key = random.choices(candidates_keys, weights=weights, k=1)[0]
            node_choose_members = server_groups[chosen_key]['members']
            
        elif candidates_keys:
            # Fallback aleatório uniforme se pesos forem zero
            chosen_key = random.choice(candidates_keys)
            node_choose_members = server_groups[chosen_key]['members']

        # 4. Efetiva a falha em todos os membros do grupo sorteado
        if self.crash_links:
            # (Mantendo lógica original de links se necessário, mas adaptada para usar o primeiro membro como referência)
             # Se crash_links for True, essa lógica precisaria de revisão profunda para suportar grupos. 
             # Vou manter o pass para não quebrar, assumindo crash_links=False conforme contexto usual.
            pass 
        else:
            self.nodes_crashed.extend(node_choose_members)
            self.nodes_crashed = list(set(self.nodes_crashed)) # Remove duplicatas por segurança

            return node_choose_members

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
