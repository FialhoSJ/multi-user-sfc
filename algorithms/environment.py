import copy
import gymnasium as gym
import numpy as np
from algorithms.networkUtils import get_shortest_path, pre_get_single_source_minimum_latency_path, calcular_latencia_total, get_available_shortest_path
from algorithms.networkUtils import calculate_computational_latency, calculate_latency_betwen_nodes
from gymnasium import spaces
import time

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

class NetworkEnv(gym.Env):
    def __init__(self, graph='', services=[0], service_requirements=1, latency_request=1, dst=1, valid_nodes=1, pesos=None, sfc=None):
        super().__init__()

        # Armazenando o grafo e outros parâmetros
        self.G = graph
        self.valid_nodes = valid_nodes

        # Parâmetros dos serviços
        self.service_requirements = service_requirements
        self.services = services
        self.service = services[0]
        self.dst_node = dst
        self.current_location = dst
        self.latency_request = latency_request
        self.latency_used = 0

        # Fatores de custo
        self.cpu_factor = pesos["cpu"]
        self.cache_factor = pesos["cache"]
        self.band_factor = pesos["band"]
        self.latency_factor = pesos["latency"]

        # Variáveis de controle
        self.is_training = True
        self.success = False
        self.reuse = False
        self.servers_used = []  # Usar set para verificação mais rápida
        self.allocation_results = {}

        # Inicialização dos objetos sfc e dst_vnf
        self.sfc = sfc  # Certifique-se de que o objeto sfc é passado para o ambiente       
        # Espaços de observação e ação
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.valid_nodes) * 9,),  # Pode ser melhor modularizado no futuro
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.valid_nodes))

        # Cache de paths para otimizar cálculos repetitivos
        self.cached_paths = {}

    # def set_dst_vnf(self,dst_vnf):
    #     self.dst_vnf = dst_vnf
    #     self.current_vnf = self.dst_vnf
    #     self.prev_vnf = self.current_vnf.get_previous_vnf()

    def reset(self, seed=None, options=None):
        """
        Reinicia o ambiente e reinicializa os parâmetros, restaurando o estado original do grafo.
        """
        # start_time = time.time()  # Início da medição de tempo
        # Recarrega os nós do grafo a partir do backup
        # print(f"Tempo para deep copy do grafo: {time.time() - start_time:.4f} segundos")  # Fim da medição de tempo
        self.G = copy.deepcopy(self.G_backup)
        # Reinicializa as variáveis de controle e custos
        self.latency_used = 0
        self.current_location = self.dst_node
        self.servers_used.clear()  # Limpa o set de servidores usados
        self.service = self.services[0]
        self.ac_total_cost = self.total_reward = self.reward = self.total_cost = 0
        self.success = False
        self.fail_reason = None
        self.allocation_results = {}
        self.path = None
        # self.server_resources = copy.deepcopy(self.server_resources_backup)
        # self.set_dst_vnf(self.dst_vnf)
        # self.bandwidth_required = self.sfc.get_link_bandwidth_required(self.prev_vnf.id, self.current_vnf.id)
        self.bandwidth_required = self.service_requirements[self.service]["in_bw"]

        return self.get_normalized_state(), {}

    def step(self, action):
        """
        Realiza uma ação no ambiente, avaliando o sucesso ou falha.
        """
        if not 0 <= action < len(self.valid_nodes):
            raise ValueError(f"Ação inválida: {action}.")
        done = False
        self.server = self.valid_nodes[int(action)]

        if type(self.server) == type("a"):
            aux=1

        # Verifica se o caminho já foi calculado e está em cache
        if (self.current_location, self.server) in self.cached_paths:
            self.path = self.cached_paths[(self.current_location, self.server)]
            if not self.is_path_bandwidth_sufficient(self.G, self.path, self.bandwidth_required):
                # Se a largura de banda não for suficiente, calcula o caminho com largura de banda disponível
                self.path = get_available_shortest_path(self.G, self.current_location, self.server, self.bandwidth_required)
                self.cached_paths[(self.current_location, self.server)] = self.path
        else:
            # Se não, calcula o menor caminho sem considerar a largura de banda
            self.path = get_available_shortest_path(self.G, self.current_location, self.server, self.bandwidth_required)
            # Armazena o caminho no cache
            self.cached_paths[(self.current_location, self.server)] = self.path

        if not self.path:
            aux=1
        # Verifica se há recursos suficientes para alocar o serviço
        if not self.allocate_resources_on_node(self.G, self.server,self.session_number, self.is_shareable):
            self.total_cost = self.calculate_total_cost(self.G)
            return self._fail_step('resource')

        # Se os recursos estão disponíveis, aloca-os
        

        # Atualiza a latência usada
        self.latency_used += calcular_latencia_total(self.path, self.G)

        # Verifica se os limites de latência ou largura de banda são atingidos
        if self.latency_used > self.latency_request:
            self.total_cost = self.calculate_total_cost(self.G)
            return self._fail_step('latency')
        elif not self.path and not self.allocate_bandwidth_along_path(self.G, self.path, self.bandwidth_required, self.service):
            self.total_cost = self.calculate_total_cost(self.G)
            return self._fail_step('bandwidth')
        
        self.total_cost = self.calculate_total_cost(self.G)
        self.ac_total_cost += self.total_cost
        self.reward = -self.total_cost
        self.total_reward += self.reward
        self.servers_used.append(self.server)  # Usar set para adicionar o servidor

        if not self.is_training:
            self.allocation_results[self.service] = {
                'allocated_server': self.server,
                'path': self.path,
                'cost': self.total_cost
            }

        done = self.service == self.services[-1]
        if not done:
            self.service = self.services[self.services.index(self.service) + 1]
            self.update_bandwidth_required()
        else:
            aux = 1
            self.success=True
        self.current_location = self.server

        return self.get_normalized_state(), self.reward, done, False, {}
    
    def set_graph(self,graph):
        self.G_backup = copy.deepcopy(graph)
        self.G = graph

    # def set_server_resources(self,server_resources):
    #     self.server_resources_backup = server_resources
    #     self.server_resources = copy.deepcopy(server_resources)

    def update_bandwidth_required(self):
        if not self.service_requirements or not self.service:
            raise ValueError("Variaveis não instanciadas")
        self.bandwidth_required = self.service_requirements[self.service]["in_bw"]

    def set_dst_node(self,dst_node):
        if not dst_node:
            raise ValueError("dst_node None")
        self.dst_node = dst_node
        self.current_location = dst_node
    
    def _has_resources(self,graph, node_id, vnf, session_id, is_shareable):
        """
        Verifica se o nó tem CPU e cache suficientes para alocar o VNF, 
        considerando reutilização se o serviço for compartilhável.
        
        Parâmetros:
        - graph: grafo com os nós contendo 'cpu_used', 'cpu_capacity', 'cache_used', 'cache_capacity'
        - node_id: identificador do nó
        - vnf: microsserviço com métodos get_cpu_request() e get_cache_request()
        - session_id: usada para formar a chave de instância
        - is_shareable: função que recebe service_id e retorna True se pode ser reutilizado

        Retorna:
        - True se houver recursos disponíveis suficientes (considerando reutilização)
        - False caso contrário
        """
        service_id = vnf.id
        service_key = (service_id, session_id)
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()
        node = graph.nodes[node_id]

        if node['type'] not in ['server', 'mobile_device']:
            return False

        # Se já existe e é reutilizável, não exige recursos adicionais
        if service_key in node['services']:
            if is_shareable(service_id):
                return True

        # Caso contrário, precisa de recursos suficientes
        cpu_available = node['cpu_capacity'] - node['cpu_used']
        cache_available = node['cache_capacity'] - node['cache_used']

        return cpu_available >= cpu_required and cache_available >= cache_required

    
    def _fail_step(self, reason):
        self.fail_reason = reason
        self.success = False
        state = self.get_normalized_state()
        self.total_cost = 2000
        self.reward = -self.total_cost
        # self.ac_total_cost += self.total_cost
        done = True

        # if done and  self.min_cost > self.ac_total_cost:
        #     self.min_cost = self.total_cost
        # self.print_info_of_allocation()
        return state, self.reward, done, False, {}


    
    def allocate_resources_on_node(self,graph, node_id,session_id, is_shareable):
        """
        Verifica e aloca recursos (CPU e cache) em um nó do grafo.

        Parâmetros:
        - graph: grafo com os nós e atributos de recursos
        - node_id: identificador do nó alvo
        - vnf: objeto do microsserviço com métodos get_cpu_request() e get_cache_request()
        - session_id: identificador da sessão (usado na chave do serviço)
        - is_shareable: função que recebe service_id e retorna True se o serviço é compartilhável

        Retorna:
        - latência computacional (float)
        """
        vnf = self.sfc.get_vnf_by_id(self.service)
        service_id = vnf.id
        service_key = (service_id, session_id)
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()
        node = graph.nodes[node_id]

        if node['type'] not in ['server', 'mobile_device']:
            return False
        if node['cpu_used'] + cpu_required > node['cpu_capacity']:
            return False
        if node['cache_used'] + cache_required > node['cache_capacity']:
            return False

        if service_key in node['services']:
            node['services'][service_key]['copys'] += 1
            if not is_shareable(service_id):
                node['cpu_used'] += cpu_required
                node['cache_used'] += cache_required
        else:
            node['services'][service_key] = {
                'cpu': cpu_required,
                'cache': cache_required,
                'copys': 1
            }
            node['cpu_used'] += cpu_required
            node['cache_used'] += cache_required
            if is_shareable(service_id):
                node['reuse'].append(vnf)

        return True
    
    def is_shareable(self,service_name):
        # TODO Mudar para a informação de compartilháveis estar em uma variável separável.
        #if self.shareable_node:
        if True:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False





    def allocate_bandwidth_along_path(self,graph, path, bandwidth_required, ms_name):
        """
        Verifica e aloca banda em todos os enlaces de um caminho.
        
        Parâmetros:
        - graph: grafo com os nós e enlaces (com atributos 'bandwidth_capacity' e 'bandwidth_used')
        - path: lista de nós representando o caminho
        - bandwidth_required: banda necessária para a transmissão
        - ms_name: nome do microsserviço ou identificador da transmissão
        
        Retorna:
        - soma das latências de comunicação entre os nós
        """
        total_latency = 0.0

        # Primeira verificação: checa se há banda em todos os enlaces
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                raise False

        # Segunda etapa: aloca efetivamente a banda e acumula latência
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edges[u, v]
            vnf = self.sfc.get_vnf_by_id(ms_name)
            latency = calculate_latency_betwen_nodes(graph, u, v, vnf)  # None se não houver VNF necessário
            total_latency += latency

            if ms_name in edge['services_in_transit']:
                edge['services_in_transit'][ms_name]['copys'] += 1
                edge['bandwidth_used'] += bandwidth_required
            else:
                edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bandwidth_required}
                edge['bandwidth_used'] += bandwidth_required

        return True
    def calculate_total_cost(self, graph):
        """
        Calcula o custo total considerando recursos, latência e largura de banda.
        Os valores de CPU, cache e tipo do nó são extraídos diretamente do grafo.
        """
        node = graph.nodes[self.server]
        vnf = self.sfc.get_vnf_by_id(self.service)  # Aqui, assumimos que self.service já é um objeto VNF com métodos abaixo

        cpu_req = vnf.get_cpu_request() if not self.reuse else 0
        cache_req = vnf.get_cache_request() if not self.reuse else 0

        cpu_avail = node["cpu_capacity"] - node["cpu_used"]
        cache_avail = node["cache_capacity"] - node["cache_used"]
        e = 1e-6  # Para evitar divisão por zero

        self.cpu_cost = ((cpu_req / (cpu_avail + e)) + 1) ** self.cpu_factor
        self.cache_cost = ((cache_req / (cache_avail + e)) + 1) ** self.cache_factor

        self.latency_cost = ((self.latency_used / self.latency_request) + 1) ** self.latency_factor
        self.bandwidth_cost = 0
        self.boot_cost = 0

        if self.server in self.servers_used:
            self.cpu_cost *= self.cpu_factor
            self.cache_cost *= self.cache_factor

        return sum([
            self.cpu_cost,
            self.cache_cost,
            self.latency_cost,
            self.bandwidth_cost,
            self.boot_cost
        ])

    def check_reuse(self, service, server):
        """
        Verifica se o serviço pode ser reutilizado no servidor.
        """
        for sf_shareable in SHAREABLE_PREFIXES:
            if service.startswith(sf_shareable):
                if verificar_chave(self.G.nodes[server]['services'], (service, self.session_number), sf_shareable):
                    return True
        return False

    def is_path_bandwidth_sufficient(self,graph, path, bandwidth_required):
        """
        Verifica se todos os enlaces de um caminho possuem banda disponível suficiente.

        Parâmetros:
        - graph: grafo contendo os enlaces com 'bandwidth_capacity' e 'bandwidth_used'
        - path: lista de nós representando o caminho (ex: [n1, n2, n3])
        - bandwidth_required: quantidade de banda necessária (float)

        Retorna:
        - True se todos os enlaces do caminho têm banda suficiente
        - False caso contrário
        """
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.edges[u, v]
            if edge['bandwidth_used'] + bandwidth_required > edge['bandwidth_capacity']:
                return False
        return True


    def get_normalized_state(self):
        """
        Retorna o estado normalizado do ambiente com uma única iteração sobre os nós válidos.
        """
        state = []

        for node_id in self.valid_nodes:
            node = self.G.nodes[node_id]
            cpu_free = node["cpu_capacity"] - node["cpu_used"]
            cache_free = node["cache_capacity"] - node["cache_used"]

            # Parte 1: estado de recursos e flags
            state.extend([
                cpu_free / 100,
                cache_free / 100,
                1 if self.check_reuse(self.service, node_id) else 0,
                1 if node_id in self.servers_used else 0
            ])

            # Parte 2: informações fixas do serviço (repetidas para cada nó)
            state.extend([
                self.service_requirements[self.service]["cpu"] / 100,
                self.service_requirements[self.service]["cache"] / 100,
                min(self.latency_used / self.latency_request, 1)
            ])

            # Parte 3: latência e viabilidade de alocação
            if (self.current_location, node_id) not in self.cached_paths:
                path = get_available_shortest_path(self.G, self.current_location, node_id, self.bandwidth_required)
                self.cached_paths[(self.current_location, node_id)] = path
            else:
                path = self.cached_paths[(self.current_location, node_id)]

            cost_latency = calcular_latencia_total(path, self.G) / self.latency_request if path else 1
            state.append(min(cost_latency, 1))

            # Parte 4: binário - se pode alocar com base em latência e recursos
            can_allocate = (
                self._has_resources(self.G, node_id, self.sfc.get_vnf_by_id(self.service), self.session_number, self.is_shareable)
                or self.check_reuse(self.service, node_id)
            ) and cost_latency < 1

            state.append(1 if can_allocate else 0)

        return np.array(state, dtype=np.float32)
    

    def is_shareable(self, service_id):
        for prefix in SHAREABLE_PREFIXES:
            if service_id.startswith(prefix):
                return True
        return False

def verificar_chave(dicionario, chave, prefixo):
    string1, string2 = chave  # Desempacotando a tupla (string1, string2)
    
    # Verificar se a chave existe no dicionário com as condições
    for (string3, string4), valor in dicionario.items():
        if string3.startswith(prefixo) and string4 == string2:
            # Se string3 tem o prefixo e string4 é igual a string2
            if string1.startswith(prefixo):
                return True  # Chave encontrada
    return False  # Chave não encontrada




