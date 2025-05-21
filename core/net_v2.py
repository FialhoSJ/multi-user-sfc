import networkx as nx
import numpy as np
import math
import random
SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

class Net2:
    def __init__(self):
        self.graph = nx.Graph()
        self.md_graph = nx.Graph()

        self.sfc_dict = {}
        self.sfc_route_info = {} # sfc_id, route_info
        self.nodes_reliability = {}

        self.total_cpu_used = 0.00
        self.total_cpu_capacity = 0.00

        self.total_cache_used = 0.00
        self.total_cache_capacity = 0.00

        self.mobile_cpu_used = 0.0
        self.mobile_cache_used = 0.0
        #self.mobile_energy_used = 0.0

        self.total_bandwidth_used = 0.00  
        self.total_bandwidth_capacity = 0.00

        self.single_source_minimum_latency_path = None 

        self.processing_delay_info = [] # stores processing delay information for each node in the topology
        self.shareable_sf_sfc = {} # stores sfc information for a shareable sf for 
        self.sf_route_info = {}
        self.sfs_flux_info = {}
        self.shared_sfs = {}
        self.shareable_band = False
        self.shareable_node = True
        self.verbose = False
    
    def add_node(self, node_id, node_type, cpu_capacity=0.00,cache_capacity=0.00, w_channel_capacity=0.0, position=(0,0),ips=0):
        if node_type  == 'server':
            self.graph.add_node(node_id,type=node_type,
                                cpu_capacity=cpu_capacity,
                                cache_capacity=cache_capacity,
                                cpu_used=0.00,
                                cache_used=0.00,
                                ips=ips*10e10,
                                position=position,
                                reuse=[],
                                services={})
        elif node_type == 'mobile_device': # Grafo separado
            self.md_graph.add_node(node_id,
                                type=node_type,
                                cpu_capacity=cpu_capacity,
                                cache_capacity=cache_capacity,
                                cpu_used=0.00,
                                cache_used=0.00,
                                ips=ips*10e10,
                                position=position,
                                reuse=[],
                                services={})
        elif node_type == 'router':
            self.graph.add_node(node_id,type=node_type,
                                cpu_capacity=cpu_capacity,
                                cache_capacity=cache_capacity,
                                cpu_used=0.00,
                                cache_used=0.00,
                                w_channel_capacity=w_channel_capacity,
                                w_channel_used=0.0,
                                position=position,
                                services={},
                                w_services={})            
        else:
            raise ValueError("Tipo de nó não reconhecido")

    def remove_node(self, node_id):
        # OBS: Por enquanto não removemos nós da rede principal
        self.md_graph.remove_node(node_id)

        # if node_id in self.graph.nodes:
        #     self.graph.remove_node(node_id)
        # # Tenta remover do grafo de dispositivos móveis
        # elif node_id in self.md_graph.nodes:
        #     self.md_graph.remove_node(node_id)
        # else:
        #     raise ValueError(f"Nó {node_id} não encontrado em nenhum dos grafos.")
        
    def add_edge(self, node1, node2, bandwidth_capacity=1000.00, latency=1):
        self.graph.add_edge(node1, node2,
                            bandwidth_capacity=bandwidth_capacity,
                            bandwidth_used=0.00,
                            latency=latency,
                            services_in_transit={})

    def deploy_sfc(self, sfc, route_info,flag_test=0):  
        sfc_id = sfc.id
        if sfc_id not in self.sfc_dict:
            self.sfc_dict[sfc_id] = sfc
        
        if sfc_id not in self.sfc_route_info:
            self.sfc_route_info[sfc_id] = route_info
        
        session = sfc_id.split("_")[-1]
        total_latency = 0
        #edge servers => [2, 5, 6, 8, 9, 14, 18, 23, 25, 28, 33, 34])
        #try:
        for ms_name, path in route_info.items():
            if ms_name in ['src','dst']:
                continue
            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            comp_latency = self.allocate_microservice(vnf, node_allocated, session)
            total_latency += comp_latency
            bw_req = vnf.get_outcome_interface_bandwidth()

            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    if isinstance(v, str):
                        comm_latency = self.allocate_wireless_bandwidth(u,v,bw_req,ms_name)
                    elif isinstance(u,str):
                        comm_latency = self.allocate_wireless_bandwidth(v,u,bw_req,ms_name)
                    else:
                        comm_latency = self.allocate_bandwidth(u, v, bw_req, ms_name)
                    total_latency += comm_latency
        # except:
        #     print("aaaaa")
        #if flag_test == 0:
        #     self.deploy_sfc(sfc,route_info,flag_test=1)
        #self.undeploy_sfc(sfc_id=sfc.id)
        # self.undeploy_sfc(sfc_id=sfc.id)
        return True

    def undeploy_sfc(self, sfc_id):
        if sfc_id not in self.sfc_dict:
            raise ValueError(f"SFC {sfc_id} não encontrada.")

        sfc = self.sfc_dict[sfc_id]
        session = sfc_id.split("_")[-1]
        route_info = self.sfc_route_info[sfc_id]

        for ms_name, path in route_info.items():
            if ms_name in ['src', 'dst']:
                continue

            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]
            self.deallocate_microservice(node_allocated, vnf, session)
            bw_req = vnf.get_outcome_interface_bandwidth()
            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    if isinstance(v, str):
                        self.release_wireless_bandwidth(u,v,ms_name)
                    elif isinstance(u,str):
                        self.release_wireless_bandwidth(v,u,ms_name)
                    else:
                        self.release_bandwidth(u, v, ms_name)

        # Remover registros da SFC
        del self.sfc_dict[sfc_id]
        del self.sfc_route_info[sfc_id]

        if self.total_cpu_used < 0  or self.total_cache_used < 0 or self.total_bandwidth_used < 0:
            raise ValueError(f"Recursos com valores Negativos")

    def allocate_microservice(self, vnf, node_id, session_id):
        # TODO melhorar essa verificação. Funciona por agora, mas caso o código mude talvez seja necessário mudar
        mobile = False
        if isinstance(node_id, str): # Se é um mobile device 
            ips_vm_capacity = 0.15*10e10
            node = self.md_graph.nodes[node_id]
            mobile = True
        else:
            ips_vm_capacity = 0.2*10e10
            node = self.graph.nodes[node_id]
        
        service_id = vnf.id
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()   
        # data_bits_per_frame * ciclos/bits * 1000 (ms) / vm's ips

        if node['type'] not in ['server', 'mobile_device']:
            raise ValueError(f"Serviços só podem ser alocados em servidores ou usuários, não em '{node['type']}'.")

        if node['cpu_used'] + cpu_required > node['cpu_capacity'] or node['cache_used'] + cache_required > node['cache_capacity']:
            raise ValueError(f"Sem capacidade suficiente no nó {node_id}.")
        
        def put_resource(cpu_required,cache_required,mobile):
            node['cpu_used'] = round(node['cpu_used'] + cpu_required,2)
            node['cache_used'] = round(node['cache_used'] + cache_required,2)
            if mobile:
                self.mobile_cpu_used = round(self.mobile_cpu_used + cpu_required,2)  
                self.mobile_cache_used = round(self.mobile_cache_used + cache_required,2)
            else:
                self.total_cpu_used = round(self.total_cpu_used + cpu_required,2)
                self.total_cache_used = round(self.total_cache_used + cache_required,2)

        service_key = (service_id,session_id)
        if service_key in node['services']:
            node['services'][service_key]['copys'] += 1 # Serviço já instanciado, então incrementa o número de cópias
            if not self.is_shareable(service_id): # Se não for compartilhável ou a sessão não for a mesma, aumenta os recursos usados
                put_resource(cpu_required,cache_required,mobile)
        else:
            node['services'][service_key] = {'cpu': cpu_required,'cache': cache_required,'copys': 1}
            put_resource(cpu_required,cache_required,mobile)
            if self.is_shareable(service_id):
                node['reuse'].append(vnf)
                #self.shared_sfs[node_id].append(vnf) # Deve ser retura
        return (vnf.get_income_interface_bandwidth()/60 * 10e6) * 10 *1000/ips_vm_capacity

    def deallocate_microservice(self, node_id, vnf, session_id):
        mobile = False
        if isinstance(node_id, str):
            mobile = True
            node = self.md_graph.nodes[node_id]
        else:
            node = self.graph.nodes[node_id]
        
        service_id = vnf.id

        service_key = (service_id, session_id)
        if service_key not in node['services']:
            raise ValueError(f"Serviço {service_id} não encontrado no nó {node_id}.")

        service_info = node['services'][service_key]
        service_info['copys'] -= 1

        def take_resource(cpu_required,cache_required):
            node['cpu_used'] = round(node['cpu_used'] - cpu_required,2)
            node['cache_used'] = round(node['cache_used'] - cache_required,2)
            if mobile:
                self.mobile_cpu_used = round(self.mobile_cpu_used - cpu_required,2)  
                self.mobile_cache_used = round(self.mobile_cache_used - cache_required,2)
            else:
                self.total_cpu_used = round(self.total_cpu_used - cpu_required,2)
                self.total_cache_used = round(self.total_cache_used - cache_required,2)
        
        if service_info['copys'] <= 0:
            del node['services'][service_key]
            take_resource(service_info['cpu'],service_info['cache'])
            
            if self.is_shareable(service_id):
                if vnf in node['reuse']:
                    node['reuse'].remove(vnf)
        else:
            # Se não é compartilhável, libera os recursos mesmo em cada cópia
            if not self.is_shareable(service_id):
                take_resource(service_info['cpu'],service_info['cache'])

    def allocate_bandwidth(self, node1, node2, bw_required, ms_name):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta entre {node1} e {node2} não existe.")

        edge = self.graph.edges[node1, node2]
        if ms_name in edge['services_in_transit']:
            edge['services_in_transit'][ms_name]['copys'] += 1
            edge['bandwidth_used'] += bw_required
            self.total_bandwidth_used += bw_required
        else:
            if edge['bandwidth_used'] + bw_required > edge['bandwidth_capacity']:
                raise ValueError(f"Banda insuficiente entre {node1} e {node2}.")

            edge['services_in_transit'][ms_name] = {'copys' : 1,'bw_used': bw_required}
            edge['bandwidth_used'] += bw_required
            self.total_bandwidth_used += bw_required
        comm_latency = self.get_link_latency(node1,node2)
        return comm_latency

    def release_bandwidth(self, node1, node2, ms_name):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta entre {node1} e {node2} não existe.")

        edge = self.graph.edges[node1, node2]
        services = edge.get('services_in_transit', {})

        if ms_name not in services:
            raise ValueError(f"Serviço {ms_name} não está em trânsito entre {node1} e {node2}.")

        services[ms_name]['copys'] -= 1

        bw_to_release = services[ms_name]['bw_used']
        edge['bandwidth_used'] = max(0, edge['bandwidth_used'] - bw_to_release)
        self.total_bandwidth_used = max(0, self.total_bandwidth_used - bw_to_release)

        if services[ms_name]['copys'] == 0:
            del services[ms_name]

    def allocate_wireless_bandwidth(self, node1, node2, bw_required, ms_name):
        router = self.graph.nodes[node1]
        if ms_name in router['w_services']:
            router['w_services'][ms_name]['copys'] += 1
            router['w_channel_used'] += bw_required
            self.total_bandwidth_used += bw_required
        else:
            if router['w_channel_used'] + bw_required > router['w_channel_capacity']:
                raise ValueError(f"Banda insuficiente entre {node1} e {node2}.")
            router['w_services'][ms_name] = {'copys' : 1,'bw_used': bw_required}
            router['w_channel_used'] += bw_required
            self.total_bandwidth_used += bw_required
        data_packet = bw_required*10e6/60
        latencia = self.calcular_latencia_5g(data=data_packet,distancia_m=500)
        return latencia

    def release_wireless_bandwidth(self, node1, node2, ms_name):
        router = self.graph.nodes[node1]
        services = router.get('w_services', {})

        if ms_name not in services:
            raise ValueError(f"Serviço {ms_name} não está em trânsito entre {node1} e {node2}.")

        services[ms_name]['copys'] -= 1
        
        bw_to_release = services[ms_name]['bw_used']

        router['w_channel_used'] = max(0.0, router['w_channel_used'] - bw_to_release)
        self.total_bandwidth_used = max(0.0, self.total_bandwidth_used - bw_to_release)

        if services[ms_name]['copys'] == 0:
            del services[ms_name]

    def calcular_latencia_5g(
        self,
        data,
        distancia_m=500,
        potencia_transmissao_dbm=30.0,
        largura_banda_hz=100e6,
        temperatura_kelvin=290,
        figura_ruido_db=10.0,
        eficiencia_codec=0.5,
        snr_minimo_db=0.0,
        freq_portadora_hz=3.5e9,
        sigma_shadowing_db=0.001,
    ):
        """
        Calcula latência (ms) para uma dada distância em 5G, considerando path loss com shadowing.

        Parâmetro:
        - distancia_m: distância em metros (float ou lista/tupla de floats)

        Retorna latência em ms (float ou lista de floats, conforme input)
        """
        BOLTZMANN = 1.380649e-23

        def path_loss_5g(distancia_m):
            pl_db = 28.0 + 22 * math.log10(distancia_m) + 20 * math.log10(freq_portadora_hz / 1e9) #+ random.gauss(0, sigma_shadowing_db)
            return 10 ** (-pl_db / 10)  # ganho linear

        def calcular_latencia_um_ponto(dado):
            ganho = path_loss_5g(distancia_m)
            potencia_w = 10 ** (potencia_transmissao_dbm / 10) / 1000
            ruido_w_hz = BOLTZMANN * temperatura_kelvin * (10 ** (figura_ruido_db / 10))
            snr_linear = (ganho * potencia_w) / (ruido_w_hz * largura_banda_hz)
            snr_linear = max(snr_linear, 10 ** (snr_minimo_db / 10))
            taxa_bps = largura_banda_hz * math.log2(1 + snr_linear)
            latencia_ms = (dado / taxa_bps) * 1000  
            return latencia_ms
        
        return calcular_latencia_um_ponto(data)
        # if isinstance(distancia_m, (list, tuple)):
        #     return [calcular_latencia_um_ponto(d) for d in distancia_m]
        # else:
        #    return calcular_latencia_um_ponto(distancia_m)

    def get_sfc_by_id(self, sfc_id):
        return self.sfc_dict[sfc_id]

    def update(self):
        pass

    def get_shareable_sfs(self):
        return self.shared_sfs

    def get_shortest_path_length(self, source, target):
        try:
            return nx.dijkstra_path_length(self.graph, source, target, weight='latency')
        except nx.NetworkXNoPath:

            return float('inf')

    def get_shortest_path(self, source, target):
        try:
            return nx.dijkstra_path(self.graph, source, target, weight='latency')
        except nx.NetworkXNoPath:
            return []
    
    def get_single_source_minimum_latency_path(self, src):
        return nx.single_source_dijkstra_path(self.graph, src, weight='latency')

    def pre_get_single_source_minimum_latency_path(self):
        # print "pre_get_single_source_minimum_latency_path"
        single_source_minimum_latency_path = {}
        for node in self.graph.nodes():
            single_source_minimum_latency_path[node] = \
            nx.single_source_dijkstra(self.graph, source=node, cutoff=None, weight='latency')
        self.single_source_minimum_latency_path = single_source_minimum_latency_path
        return single_source_minimum_latency_path

    def is_shareable(self,service_name):
        if self.shareable_node:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False

    def get_link_info(self, node1, node2):
        return self.graph.edges[node1, node2]
    
    def get_node_cpu_used(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")
        return self.graph.nodes[node_id]['cpu_used']

    def get_node_cpu_free(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")
        return self.graph.nodes[node_id]['cpu_capacity'] - self.graph.nodes[node_id]['cpu_used']

    def get_node_cpu_capacity(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")
        return self.graph.nodes[node_id]['cpu_capacity']

    def get_node_cache_used(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")
        return self.graph.nodes[node_id]['cache_used']

    def get_node_cache_free(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")
        return self.graph.nodes[node_id]['cache_capacity'] - self.graph.nodes[node_id]['cache_used']

    def get_node_cache_capacity(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")
        return self.graph.nodes[node_id]['cache_capacity']

    def get_link_bandwidth_used(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
        return self.graph.edges[node1, node2]['bandwidth_used']

    def get_link_bandwidth_free(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
        return self.graph.edges[node1, node2]['bandwidth_capacity'] - self.graph.edges[node1, node2]['bandwidth_used']

    def get_link_bandwidth_capacity(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
        return self.graph.edges[node1, node2]['bandwidth_capacity']

    def get_link_latency(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta entre {node1} e {node2} não existe.")
        return self.graph.edges[node1, node2]['latency']

    def get_cpu_utilization_rate(self):
        return self.total_cpu_used*1.0/self.total_cpu_capacity
    
    def get_cache_utilization_rate(self):
        return self.total_cache_used*1.0/self.total_cache_capacity

    def get_bandwidth_utilization_rate(self):
        self.update()
        return self.total_bandwidth_used*1.0/self.total_bandwidth_capacity

    def print_network(self):
        print("\n--- Nós ---")
        for node, data in self.graph.nodes(data=True):
            print(f"{node} -> {data}")
        print("\n--- Arestas ---")
        for u, v, data in self.graph.edges(data=True):
            print(f"{u} <-> {v} -> {data}")

    def print_out_nodes_information(self, failure_cpu=None, failure_cache=None):
        if failure_cpu is None:
            print("CPU       utilization: ", str(round(self.total_cpu_used*1.0/self.total_cpu_capacity*100,3)) +'%')
        else:
            print("CPU       utilization: ", str(round(self.total_cpu_used*1.0/self.total_cpu_capacity*100,3)) +'%', end=" ")
        if failure_cache is None:
            print("Cache     utilization: ", str(round(self.total_cache_used*1.0/self.total_cache_capacity*100,3)) +'%')
        else:
            print("Cache     utilization: ", str(round(self.total_cache_used*1.0/self.total_cache_capacity*100,3)) +'%', end=" ")

    def print_out_edges_information(self, failure_band=None):
        if failure_band is None:
            print("Bandwidth utilization: ", str(round(self.total_bandwidth_used*1.0/self.total_bandwidth_capacity*100,3))+'%')
        else:
            print("Bandwidth utilization: ", str(round(self.total_bandwidth_used*1.0/self.total_bandwidth_capacity*100,3))+'%', end=" ")
            print(f"     Failure for Band: {failure_band}%")

    def print_out_acceptance_information(self,success_arr):
        if len(success_arr) != 0: 
            print("Acceptance: ", str(round(np.mean(success_arr)*100,3))+'%', end=" ")

    def connect_mobile_user(self, user_id, router_id, cpu_capacity):
        # TODO A latência e banda dessa comunicação devem ser modelados 
        self.add_node(user_id, 'user', cpu_capacity)
        self.add_edge(user_id, router_id, bandwidth_capacity=100, latency=5)

# Teste das funcionalidades com banda e latência
if __name__ == '__main__':
    substrate_network = Net2()

    substrate_network.add_node("S1", "server", cpu_capacity=100)
    substrate_network.add_node("R1", "router")
    substrate_network.add_node("R2", "router")

    substrate_network.add_edge("S1", "R1", bandwidth_capacity=500, latency=2)
    substrate_network.add_edge("R1", "R2", bandwidth_capacity=300, latency=10)

    substrate_network.allocate_microservice("S1", "svc1", 40)

    # Conecta usuário móvel e aloca serviço
    substrate_network.connect_mobile_user("U1", "R2", cpu_capacity=20)
    substrate_network.allocate_microservice("U1", "svc2", 10)

    # Reserva banda entre S1 <-> R1
    substrate_network.allocate_bandwidth("S1", "R1", 100)
    substrate_network.allocate_bandwidth("R1", "R2", 50)

    # Mostra toda a estrutura da rede
    substrate_network.print_network()
