import networkx as nx
import numpy as np
import math
import random

SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')

class Net2:
    def __init__(self):
        # --- Inicialização de Grafos ---
        self.graph = nx.Graph()
        self.md_graph = nx.Graph()

        # --- Estruturas de Dados de Controle ---
        self.sfc_dict = {}
        self.sfc_route_info = {}  # sfc_id, route_info
        self.nodes_reliability = {}
        self.alpha_stress = 0.0
        self.tier_reliability = {'default': 1.0, 'a': 1.0, 'b': 1.0, 'c': 1.0}
        self.processing_delay_info = []
        self.shareable_sf_sfc = {}
        self.sf_route_info = {}
        self.sfs_flux_info = {}
        self.shared_sfs = {}
        self.single_source_minimum_latency_path = None

        # --- Flags de Configuração ---
        self.shareable_band = False
        self.shareable_node = True
        self.verbose = False

        # --- Métricas Globais (CPU) ---
        self.total_cpu_used = 0.00
        self.total_cpu_saved = 0
        self.total_cpu_capacity = 0.00
        self.total_cpu_requested = 0.00
        self.mobile_cpu_used = 0.0

        # --- Métricas Globais (GPU) ---
        self.total_gpu_used = 0.00
        self.total_gpu_saved = 0
        self.total_gpu_capacity = 0.00
        self.total_gpu_requested = 0.00
        self.mobile_gpu_used = 0.0

        # --- Métricas Globais (Cache) ---
        self.total_cache_used = 0.00
        self.total_cache_saved = 0
        self.total_cache_capacity = 0.00
        self.total_cache_requested = 0.00
        self.mobile_cache_used = 0.0
        self.shared_vnfs_count = 0

        # --- Métricas Globais (Banda) ---
        self.total_bandwidth_used = 0.00
        self.total_bandwidth_capacity = 0.00

    # =========================================================================
    # 1. GERENCIAMENTO DE TOPOLOGIA (NÓS E ARESTAS)
    # =========================================================================

    def add_node(self, node_id, node_type, cpu_capacity=0.00, cache_capacity=0.00, w_channel_capacity=0.0, position=(0, 0), ips=0):
        if 'server' in node_type:
            node_level = node_type.split("_")[-1]
            node_type_clean = node_type.split("_")[0]
            self.graph.add_node(node_id, type=node_type_clean,
                                cpu_capacity=cpu_capacity,
                                cache_capacity=cache_capacity,
                                cpu_used=0.00,
                                cache_used=0.00,
                                ips=ips * 10e10,
                                position=position,
                                reuse=[],
                                services={},
                                sfcs_list=[],
                                level_server=node_level,
                                is_active=True)
            
        elif node_type == 'mobile_device':
            # Simula variabilidade de dispositivos móveis
            sorteio = random.random()
            if sorteio <= 0.33:
                cpu_capacity *= 0.75
                cache_capacity *= 0.75
                ips *= 0.75
            elif sorteio > 0.67:
                cpu_capacity *= 1.25
                cache_capacity *= 1.25
                ips *= 1.25

            self.md_graph.add_node(node_id,
                                   type=node_type,
                                   cpu_capacity=cpu_capacity,
                                   cache_capacity=cache_capacity,
                                   cpu_used=0.00,
                                   cache_used=0.00,
                                   ips=ips * 10e10,
                                   position=position,
                                   reuse=[],
                                   services={},
                                   sfcs_list=[],
                                   is_active=True)
            
        elif node_type == 'router':
            self.graph.add_node(node_id, type=node_type,
                                cpu_capacity=cpu_capacity,
                                cache_capacity=cache_capacity,
                                cpu_used=0.00,
                                cache_used=0.00,
                                w_channel_capacity=w_channel_capacity,
                                w_channel_used=0.0,
                                position=position,
                                services={},
                                w_services={},
                                is_active=True)
        else:
            raise ValueError("Tipo de nó não reconhecido")

    def remove_node(self, node_id):
        # OBS: Por enquanto removemos apenas do grafo de dispositivos móveis
        self.md_graph.remove_node(node_id)

    def add_edge(self, node1, node2, bandwidth_capacity=1000.00, latency=1):
        self.graph.add_edge(node1, node2,
                            bandwidth_capacity=bandwidth_capacity,
                            bandwidth_used=0.00,
                            latency=latency,
                            services_in_transit={})

    def connect_mobile_user(self, user_id, router_id, cpu_capacity):
        # TODO: A latência e banda dessa comunicação devem ser modeladas melhor futuramente
        self.add_node(user_id, 'mobile_device', cpu_capacity=cpu_capacity)
        # Nota: Originalmente chamava 'user', mudei para 'mobile_device' para bater com add_node, 
        # mas mantendo lógica original do script se 'user' fosse tratado igual.
        # Assumindo que a lógica original usava mobile_device no add_node.
        self.add_edge(user_id, router_id, bandwidth_capacity=100, latency=5)

    # =========================================================================
    # 2. ALOCAÇÃO E CICLO DE VIDA DE SFC (Service Function Chaining)
    # =========================================================================

    def deploy_sfc(self, sfc, route_info, flag_test=0):
        sfc_id = sfc.id
        if sfc_id not in self.sfc_dict:
            self.sfc_dict[sfc_id] = sfc

        if sfc_id not in self.sfc_route_info:
            self.sfc_route_info[sfc_id] = route_info

        # Itera sobre os microserviços (VNFs) da SFC
        for ms_name, path in route_info.items():
            if ms_name in ['src', 'dst']:
                continue
            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            self.allocate_microservice(sfc, vnf, node_allocated)
            bw_req = vnf.get_outcome_interface_bandwidth()

            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    if isinstance(v, str):
                        self.allocate_wireless_bandwidth(u, v, bw_req, ms_name)
                    elif isinstance(u, str):
                        self.allocate_wireless_bandwidth(v, u, bw_req, ms_name)
                    else:
                        self.allocate_bandwidth(u, v, bw_req, ms_name)
        return True

    def undeploy_sfc(self, sfc_id):
        if sfc_id not in self.sfc_dict:
            raise ValueError(f"SFC {sfc_id} não encontrada.")

        sfc = self.sfc_dict[sfc_id]
        route_info = self.sfc_route_info[sfc_id]

        for ms_name, path in route_info.items():
            if ms_name in ['src', 'dst']:
                continue

            vnf = sfc.get_vnf_by_id(ms_name)
            node_allocated = path[0]

            self.deallocate_microservice(node_allocated, sfc_id, vnf)
            
            if len(path) > 1:
                for u, v in zip(path[:-1], path[1:]):
                    if isinstance(v, str):
                        self.release_wireless_bandwidth(u, v, ms_name)
                    elif isinstance(u, str):
                        self.release_wireless_bandwidth(v, u, ms_name)
                    else:
                        self.release_bandwidth(u, v, ms_name)

        # Remover registros da SFC
        del self.sfc_dict[sfc_id]
        del self.sfc_route_info[sfc_id]

        if self.total_cpu_used < 0 or self.total_cache_used < 0 or self.total_bandwidth_used < 0:
            raise ValueError(f"Recursos com valores Negativos após undeploy")

    # =========================================================================
    # 3. ALOCAÇÃO DE RECURSOS (COMPUTACIONAIS E REDE)
    # =========================================================================

    def allocate_microservice(self, sfc, vnf, node_id):
        sfc_id = sfc.id
        session = sfc_id.split("_")[-1]
        mobile = False

        # Verifica grafos
        if node_id in self.md_graph:
            node = self.md_graph.nodes[node_id]
            mobile = True
        elif node_id in self.graph:
            node = self.graph.nodes[node_id]
            mobile = False
        else:
            raise ValueError(f"Nó {node_id} não encontrado em nenhum dos grafos (allocate).")

        if not node.get('is_active', True):
            raise ValueError(f"Falha critica: tentativa de alocar no nó {node_id} que está inativo")

        service_id = vnf.id
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()

        # Lógica CPU vs GPU
        is_gpu = self._is_gpu_node(node_id)
        if is_gpu:
            self.total_gpu_requested = round(self.total_gpu_requested + cpu_required, 2)
        else:
            self.total_cpu_requested = round(self.total_cpu_requested + cpu_required, 2)

        self.total_cache_requested = round(self.total_cache_requested + cache_required, 2)

        if node['type'] not in ['server', 'mobile_device']:
            raise ValueError(f"Serviços só podem ser alocados em servidores ou usuários, não em '{node['type']}'.")

        # --- Helper Interno ---
        def put_resource(cpu_req, cache_req, is_mobile):
            node['cpu_used'] = round(node['cpu_used'] + cpu_req, 2)
            node['cache_used'] = round(node['cache_used'] + cache_req, 2)

            if is_gpu:
                if is_mobile:
                    self.mobile_gpu_used = round(self.mobile_gpu_used + cpu_req, 2)
                else:
                    self.total_gpu_used = round(self.total_gpu_used + cpu_req, 2)
            else:
                if is_mobile:
                    self.mobile_cpu_used = round(self.mobile_cpu_used + cpu_req, 2)
                else:
                    self.total_cpu_used = round(self.total_cpu_used + cpu_req, 2)
            
            if is_mobile:
                self.mobile_cache_used = round(self.mobile_cache_used + cache_req, 2)
            else:
                self.total_cache_used = round(self.total_cache_used + cache_req, 2)
        # ----------------------

        if sfc_id not in node['sfcs_list']:
            node['sfcs_list'].append(sfc_id)

        service_key = (service_id, session)

        # Lógica de Reuso
        if service_key in node['services']:
            node['services'][service_key]['copys'] += 1

            if not self.is_shareable(service_id):
                if node['cpu_used'] + cpu_required > node['cpu_capacity'] or \
                   node['cache_used'] + cache_required > node['cache_capacity']:
                    node['services'][service_key]['copys'] -= 1
                    if sfc_id in node['sfcs_list']: node['sfcs_list'].remove(sfc_id)
                    raise ValueError(f"Sem capacidade suficiente no nó {node_id} para instância não compartilhável.")
                put_resource(cpu_required, cache_required, mobile)
            else:
                # É compartilhável, apenas registra economia
                if is_gpu:
                    self.total_gpu_saved = round(self.total_gpu_saved + cpu_required, 2)
                else:
                    self.total_cpu_saved = round(self.total_cpu_saved + cpu_required, 2)
                self.total_cache_saved = round(self.total_cache_saved + cache_required, 2)
                self.shared_vnfs_count += 1
        else:
            # Serviço novo no nó
            if node['cpu_used'] + cpu_required > node['cpu_capacity'] or \
               node['cache_used'] + cache_required > node['cache_capacity']:
                if sfc_id in node['sfcs_list']: node['sfcs_list'].remove(sfc_id)
                raise ValueError(f"Sem capacidade suficiente no nó {node_id} para novo serviço.")

            node['services'][service_key] = {'cpu': cpu_required, 'cache': cache_required, 'copys': 1}
            put_resource(cpu_required, cache_required, mobile)

            if self.is_shareable(service_id):
                node['reuse'].append(vnf)

    def deallocate_microservice(self, node_id, sfc_id, vnf):
        mobile = False
        if node_id in self.md_graph:
            node = self.md_graph.nodes[node_id]
            mobile = True
        elif node_id in self.graph:
            node = self.graph.nodes[node_id]
            mobile = False
        else:
            raise ValueError(f"Nó {node_id} não encontrado em nenhum dos grafos (deallocate).")

        service_id = vnf.id
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()

        is_gpu = self._is_gpu_node(node_id)
        if is_gpu:
            self.total_gpu_requested = round(self.total_gpu_requested - cpu_required, 2)
        else:
            self.total_cpu_requested = round(self.total_cpu_requested - cpu_required, 2)
        self.total_cache_requested = round(self.total_cache_requested - cache_required, 2)

        session_id = sfc_id.split("_")[-1]
        service_key = (service_id, session_id)
        if service_key not in node['services']:
            raise ValueError(f"Serviço {service_id} não encontrado no nó {node_id}.")

        service_info = node['services'][service_key]
        cpu_to_handle = service_info['cpu']
        cache_to_handle = service_info['cache']
        service_info['copys'] -= 1

        # --- Helper Interno ---
        def take_resource(cpu_required, cache_required):
            node['cpu_used'] = round(node['cpu_used'] - cpu_required, 2)
            node['cache_used'] = round(node['cache_used'] - cache_required, 2)

            if is_gpu:
                if mobile:
                    self.mobile_gpu_used = round(self.mobile_gpu_used - cpu_required, 2)
                else:
                    self.total_gpu_used = round(self.total_gpu_used - cpu_required, 2)
            else:
                if mobile:
                    self.mobile_cpu_used = round(self.mobile_cpu_used - cpu_required, 2)
                else:
                    self.total_cpu_used = round(self.total_cpu_used - cpu_required, 2)
            
            if mobile:
                self.mobile_cache_used = round(self.mobile_cache_used - cache_required, 2)
            else:
                self.total_cache_used = round(self.total_cache_used - cache_required, 2)
        # ----------------------

        if sfc_id in node['sfcs_list']:
            node['sfcs_list'].remove(sfc_id)

        if service_info['copys'] <= 0:
            del node['services'][service_key]
            take_resource(service_info['cpu'], service_info['cache'])
            if self.is_shareable(service_id):
                if vnf in node['reuse']:
                    node['reuse'].remove(vnf)
        else:
            if not self.is_shareable(service_id):
                take_resource(service_info['cpu'], service_info['cache'])
            else:
                if is_gpu:
                    self.total_gpu_saved = round(self.total_gpu_saved - cpu_to_handle, 2)
                else:
                    self.total_cpu_saved = round(self.total_cpu_saved - cpu_to_handle, 2)
                
                self.total_cache_saved = round(self.total_cache_saved - cache_to_handle, 2)
                self.shared_vnfs_count -= 1
                self.shared_vnfs_count = max(0, self.shared_vnfs_count)

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

            edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bw_required}
            edge['bandwidth_used'] += bw_required
            self.total_bandwidth_used += bw_required
        comm_latency = self.get_link_latency(node1, node2)
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
            router['w_services'][ms_name] = {'copys': 1, 'bw_used': bw_required}
            router['w_channel_used'] += bw_required
            self.total_bandwidth_used += bw_required
        
        # Latência 5G placeholder
        latencia = 0 
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

    # =========================================================================
    # 4. CÁLCULO DE LATÊNCIA E MODELOS FÍSICOS
    # =========================================================================

    def calculate_5g_latency(self, graph, data, distancia_m=750, potencia_transmissao_dbm=30.0,
                             largura_banda_hz=100e6, temperatura_kelvin=290, figura_ruido_db=10.0,
                             eficiencia_codec=0.5, snr_minimo_db=0.0, freq_portadora_hz=3.5e9, sigma_shadowing_db=0):
        
        BOLTZMANN = 1.380649e-23

        def path_loss_5g(distancia_m):
            pl_db = 28.0 + 22 * math.log10(distancia_m) + 20 * math.log10(freq_portadora_hz / 1e9)
            pl_db += random.gauss(0, sigma_shadowing_db)
            return 10 ** (-pl_db / 10)

        def calcular_latencia_um_ponto(dado):
            ganho = path_loss_5g(distancia_m)
            potencia_w = 10 ** (potencia_transmissao_dbm / 10) / 1000
            ruido_w_hz = BOLTZMANN * temperatura_kelvin * (10 ** (figura_ruido_db / 10))
            snr_linear = (ganho * potencia_w) / (ruido_w_hz * largura_banda_hz)
            snr_linear = max(snr_linear, 10 ** (snr_minimo_db / 10))
            taxa_bps = largura_banda_hz * math.log2(1 + snr_linear) * eficiencia_codec
            latencia_ms = (dado / taxa_bps) * 1000
            return latencia_ms
        
        return calcular_latencia_um_ponto(data)

    def calculate_computational_latency(self, graph, node, vnf):
        ips = self.md_graph[node]['ips'] if self.is_mobile_node(node) else self.graph[node]['ips']
        packet = vnf.get_income_interface_bandwidth() / 60 * 1e6
        return packet * 10 * 1000 / ips

    def calculate_latency_betwen_nodes(self, graph, node1, node2, vnf):
        data_packet = (vnf.get_outcome_interface_bandwidth() / 60) * 1e6
        if self.is_mobile_node(node1):
            return self.calculate_5g_latency(data_packet, self.md_graph.nodes[node1]['position'])
        elif self.is_mobile_node(node2):
            return self.calculate_5g_latency(data_packet, self.md_graph.nodes[node2]['position'])
        else:
            return self.get_link_latency(node1, node2)

    def calculate_average_sfc_latency(self):
        if not self.sfc_dict:
            return 0.0

        total_latency_all_sfcs = 0.0
        for sfc_id, sfc in self.sfc_dict.items():
            current_sfc_latency = 0.0
            route_info = self.sfc_route_info[sfc_id]

            current_vnf = sfc.get_vnf_by_id('src')
            while current_vnf and current_vnf.id != 'dst':
                next_vnf = sfc.get_next_vnf(current_vnf)
                if not next_vnf or next_vnf.id == 'dst':
                    break

                # --- Latência Computacional ---
                allocation_path = route_info.get(next_vnf.id)
                if not allocation_path:
                    current_vnf = next_vnf
                    continue

                allocated_node = allocation_path[0]
                if self.is_mobile_node(allocated_node):
                    ips = self.md_graph.nodes[allocated_node]['ips']
                else:
                    ips = self.graph.nodes[allocated_node]['ips']
                
                packet = next_vnf.get_income_interface_bandwidth() / 60 * 1e6
                comp_latency = packet * 10 * 1000 / ips
                current_sfc_latency += comp_latency

                # --- Latência de Comunicação ---
                path_to_next_vnf = route_info.get(next_vnf.id, [])
                if len(path_to_next_vnf) > 1:
                    edge_latency = 0
                    for i in range(len(path_to_next_vnf) - 1):
                        u = path_to_next_vnf[i]
                        v = path_to_next_vnf[i + 1]
                        edge_latency += self.calculate_latency_betwen_nodes(self.graph, u, v, next_vnf)
                    current_sfc_latency += edge_latency

                current_vnf = next_vnf

            total_latency_all_sfcs += current_sfc_latency

        return total_latency_all_sfcs / len(self.sfc_dict)

    # =========================================================================
    # 5. SIMULAÇÃO DE FALHAS E RECUPERAÇÃO
    # =========================================================================

    def set_node_down(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")

        node = self.graph.nodes[node_id]
        node["is_active"] = False

        if 'original_cpu_capacity' not in node:
            node['original_cpu_capacity'] = node.get('cpu_capacity', 0)
            node['original_cache_capacity'] = node.get('cache_capacity', 0)

        node['cpu_capacity'] = 0
        node['cache_capacity'] = 0
        print(f"Debug: Nó {node_id} caiu! (Carga perdida: {node.get('cpu_used', 0)})")

    def restore_node(self, node_id, cpu_capacity=None, cache_capacity=None):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} não existe na topologia.")

        node = self.graph.nodes[node_id]
        node["is_active"] = True

        restored_cpu = cpu_capacity if cpu_capacity is not None else node.get('original_cpu_capacity', 100)
        restored_cache = cache_capacity if cache_capacity is not None else node.get('original_cache_capacity', 100)

        node['cpu_capacity'] = restored_cpu
        node['cache_capacity'] = restored_cache
        node.pop('original_cpu_capacity', None)
        node.pop('original_cache_capacity', None)

        print(f"David - Debug: Nó {node_id} recuperado e Ativo")

    def set_link_down(self, u, v):
        if not self.graph.has_edge(u, v):
            raise ValueError(f"Link {u}-{v} não encontrado.")

        edge = self.graph.edges[u, v]
        if 'original_bw' not in edge:
            edge['original_bw'] = edge.get('bandwidth_capacity', 1000.0)
            edge['original_lat'] = edge.get('latency', 1.0)

        edge['bandwidth_capacity'] = 0.0
        edge['latency'] = float('inf')

    def restore_link(self, u, v):
        if not self.graph.has_edge(u, v):
            return

        edge = self.graph.edges[u, v]
        if 'original_bw' in edge:
            edge['bandwidth_capacity'] = edge['original_bw']
            edge['latency'] = edge['original_lat']
            del edge['original_bw']
            del edge['original_lat']
            
    def activate_backup_path_bandwidth(self, path, bw_required, vnf_id_backup):
        """
        Ativa o consumo de banda em um caminho de backup que estava em standby (0 bw).
        """
        # Itera sobre os links do caminho
        for u, v in zip(path[:-1], path[1:]):
            if not self.graph.has_edge(u, v):
                continue
                
            edge = self.graph.edges[u, v]
            
            # 1. Verifica se há capacidade (Best Effort)
            # Se não houver banda agora, a ativação falha (risco do Cold Standby)
            if edge['bandwidth_used'] + bw_required > edge['bandwidth_capacity']:
                print(f"CRITICAL: Falha ao ativar banda de backup no link {u}-{v}. Congestionamento.")
                return False

            # 2. Atualiza o uso global do link
            edge['bandwidth_used'] += bw_required
            self.total_bandwidth_used += bw_required
            
            # 3. Atualiza o registro do serviço naquele link
            # O serviço já existe lá (com 0 bw), apenas atualizamos
            if vnf_id_backup in edge['services_in_transit']:
                edge['services_in_transit'][vnf_id_backup]['bw_used'] += bw_required
            else:
                # Caso raro onde o serviço não estava registrado, criamos
                edge['services_in_transit'][vnf_id_backup] = {'copys': 1, 'bw_used': bw_required}
                
        return True

    # =========================================================================
    # 5.1. GERENCIAMENTO DE CONFIABILIDADE (Novo)
    # =========================================================================

    def set_reliability_params(self, args):
        """Define os parâmetros de confiabilidade a partir dos argumentos."""
        # Garante que stress_factor não seja negativo
        self.alpha_stress = args.stress_factor if hasattr(args, 'stress_factor') and args.stress_factor >= 0 else 0.0
        
        # Valida e define confiabilidades por nível (tier)
        def validate(val): return val if 0.0 <= val <= 1.0 else 0.99
        
        # Usa getattr para evitar erro se o argumento não existir no objeto args
        self.tier_reliability = {
            'c': validate(getattr(args, 'rel_high', 0.99)),
            'b': validate(getattr(args, 'rel_normal', 0.95)),
            'a': validate(getattr(args, 'rel_low', 0.90)),
            'default': validate(getattr(args, 'rel_normal', 0.95))
        }

    def get_node_reliability(self, node_id):
        """
        Calcula a confiabilidade unificada do servidor físico.
        Considera a soma de carga da CPU (versão normal) e GPU (versão .1)
        para determinar o estresse total do hardware.
        """
        # 1. Normalização de IDs: Descobre quem é a CPU (base) e quem é a GPU (.1)
        node_id_str = str(node_id)
        if node_id_str.endswith(".1"):
            base_id = node_id_str.replace(".1", "")
            gpu_id = node_id_str
        else:
            base_id = node_id_str
            gpu_id = node_id_str + ".1"

        # 2. Variáveis acumuladoras
        total_used = 0.0
        total_capacity = 0.0
        server_level = 'default'
        is_active_physically = False
        
        # Função auxiliar para extrair dados se o nó existir no grafo
        def get_part_data(nid):
            if nid in self.graph:
                node = self.graph.nodes[nid]
                # Se qualquer parte (CPU ou GPU) estiver ativa, consideramos a máquina ligada
                if node.get('is_active', True):
                    # Pega a capacidade (tratando caso de falha momentânea onde cap=0)
                    cap = node.get('cpu_capacity', 0.0)
                    if cap <= 0: 
                        cap = node.get('original_cpu_capacity', 1.0) or 1.0
                    return node.get('cpu_used', 0.0), cap, node.get('level_server', 'default'), True
            return 0.0, 0.0, None, False

        # 3. Coleta dados das duas partes
        used_cpu, cap_cpu, lvl_cpu, active_cpu = get_part_data(int(base_id))
        used_gpu, cap_gpu, lvl_gpu, active_gpu = get_part_data(float(gpu_id))

        # Se nem CPU nem GPU existem ou estão ativas, confiabilidade é 0
        if not active_cpu and not active_gpu:
            return 0.0

        # Define o nível do servidor (prefere a info da base/CPU, mas aceita da GPU se necessário)
        server_level = lvl_cpu if lvl_cpu else (lvl_gpu if lvl_gpu else 'default')

        # 4. Consolidação
        total_used = used_cpu + used_gpu
        total_capacity = cap_cpu + cap_gpu

        # Evita divisão por zero
        if total_capacity <= 0:
            return 0.0

        # 5. Cálculo Unificado
        # R_base depende do nível da máquina física
        base_r = self.tier_reliability.get(str(server_level), self.tier_reliability['default'])

        # O estresse agora é a fração de uso TOTAL da caixa (CPU + GPU)
        global_utilization = total_used / total_capacity
        
        stress_penalty = global_utilization * self.alpha_stress
        final_reliability = base_r - stress_penalty
        
        return max(0.0, final_reliability)

    # =========================================================================
    # 6. ALGORITMOS DE CAMINHO MÍNIMO (PATHFINDING)
    # =========================================================================

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
        single_source_minimum_latency_path = {}
        for node in self.graph.nodes():
            single_source_minimum_latency_path[node] = \
                nx.single_source_dijkstra(self.graph, source=node, cutoff=None, weight='latency')
        self.single_source_minimum_latency_path = single_source_minimum_latency_path
        return single_source_minimum_latency_path

    # =========================================================================
    # 7. GETTERS E MÉTRICAS DE RECURSOS (NÓ E REDE)
    # =========================================================================

    # --- Informações de Nó Individual ---
    def get_node_cpu_used(self, node_id):
        if node_id not in self.graph: raise ValueError(f"Nó {node_id} não existe.")
        return self.graph.nodes[node_id]['cpu_used']

    def get_node_cpu_free(self, node_id):
        if node_id not in self.graph: raise ValueError(f"Nó {node_id} não existe.")
        return self.graph.nodes[node_id]['cpu_capacity'] - self.graph.nodes[node_id]['cpu_used']

    def get_node_cpu_capacity(self, node_id):
        if node_id not in self.graph: raise ValueError(f"Nó {node_id} não existe.")
        return self.graph.nodes[node_id]['cpu_capacity']

    def get_node_cache_used(self, node_id):
        if node_id not in self.graph: raise ValueError(f"Nó {node_id} não existe.")
        return self.graph.nodes[node_id]['cache_used']

    def get_node_cache_free(self, node_id):
        if node_id not in self.graph: raise ValueError(f"Nó {node_id} não existe.")
        return self.graph.nodes[node_id]['cache_capacity'] - self.graph.nodes[node_id]['cache_used']

    def get_node_cache_capacity(self, node_id):
        if node_id not in self.graph: raise ValueError(f"Nó {node_id} não existe.")
        return self.graph.nodes[node_id]['cache_capacity']

    def get_node_sfcs(self, node_id):
        return self.graph.nodes[node_id]["sfcs_list"]

    # --- Informações de Link Individual ---
    def get_link_bandwidth_used(self, node1, node2):
        if not self.graph.has_edge(node1, node2): raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['bandwidth_used']

    def get_link_bandwidth_free(self, node1, node2):
        if not self.graph.has_edge(node1, node2): raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['bandwidth_capacity'] - self.graph.edges[node1, node2]['bandwidth_used']

    def get_link_bandwidth_capacity(self, node1, node2):
        if not self.graph.has_edge(node1, node2): raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['bandwidth_capacity']

    def get_link_latency(self, node1, node2):
        if not self.graph.has_edge(node1, node2): raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['latency']
    
    def get_link_info(self, node1, node2):
        return self.graph.edges[node1, node2]

    # --- Totais Requisitados e Economizados ---
    def get_total_gpu_request(self): return self.total_gpu_requested
    def get_total_gpu_saved(self): return self.total_gpu_saved
    def get_total_cpu_request(self): return self.total_cpu_requested
    def get_total_cpu_saved(self): return self.total_cpu_saved
    def get_total_cache_request(self): return self.total_cache_requested
    def get_total_cache_saved(self): return self.total_cache_saved

    # --- Totais Usados ---
    def get_cpu_network_used(self): return self.total_cpu_used
    def get_gpu_network_used(self): return self.total_gpu_used
    def get_cpu_total_used(self): return self.total_cpu_used + self.mobile_cpu_used
    def get_gpu_total_used(self): return self.total_gpu_used + self.mobile_gpu_used
    def get_cache_total_used(self): return self.total_cache_used + self.mobile_cache_used
    def get_cache_used(self): return self.total_cache_used

    # --- Capacidades Totais do Sistema ---
    def get_total_system_cpu_capacity(self):
        total_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            if 'cpu_capacity' in node_data and not self._is_gpu_node(node_id):
                total_capacity += node_data['cpu_capacity']
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cpu_capacity' in node_data and not self._is_gpu_node(node_id):
                total_capacity += node_data['cpu_capacity']
        return total_capacity

    def get_total_system_gpu_capacity(self):
        total_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            if 'cpu_capacity' in node_data and self._is_gpu_node(node_id):
                total_capacity += node_data['cpu_capacity']
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cpu_capacity' in node_data and self._is_gpu_node(node_id):
                total_capacity += node_data['cpu_capacity']
        return total_capacity

    # --- Taxas de Utilização do Sistema ---
    def get_server_cpu_utilization_rate(self):
        if self.total_cpu_capacity == 0: return 0.0
        return self.total_cpu_used / self.total_cpu_capacity

    def get_total_system_utilization_cpu_rate(self):
        total_capacity = self.get_total_system_cpu_capacity()
        if total_capacity == 0: return 0.0
        total_used = self.total_cpu_used + self.mobile_cpu_used
        return total_used / total_capacity

    def get_total_system_utilization_gpu_rate(self):
        total_capacity = self.get_total_system_gpu_capacity()
        if total_capacity == 0: return 0.0
        total_used = self.total_gpu_used + self.mobile_gpu_used
        return total_used / total_capacity

    def get_total_system_utilization_cache_rate(self):
        if self.total_cache_capacity == 0: return 0.0
        total_used = self.total_cache_used + self.mobile_cache_used
        return total_used / self.total_cache_capacity

    def get_total_system_processing_utilization_rate(self):
        total_processing_used = self.get_cpu_total_used() + self.get_gpu_total_used()
        total_processing_capacity = self.get_total_system_cpu_capacity() + self.get_total_system_gpu_capacity()
        if total_processing_capacity == 0: return 0.0
        return total_processing_used / total_processing_capacity

    def get_cache_utilization_rate(self):
        return self.total_cache_used * 1.0 / self.total_cache_capacity

    def get_bandwidth_utilization_rate(self):
        self.update()
        return self.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity

    # --- Percentuais de Utilização Específicos ---
    def get_network_cpu_utilization_percentage(self):
        total_network_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            if 'cpu_capacity' in node_data and not self._is_gpu_node(node_id):
                total_network_capacity += node_data['cpu_capacity']
        if total_network_capacity == 0: return 0.0
        return (self.total_cpu_used / total_network_capacity) * 100

    def get_network_gpu_utilization_percentage(self):
        total_network_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            if 'cpu_capacity' in node_data and self._is_gpu_node(node_id):
                total_network_capacity += node_data['cpu_capacity']
        if total_network_capacity == 0: return 0.0
        return (self.total_gpu_used / total_network_capacity) * 100

    def get_network_cache_utilization_percentage(self):
        total_network_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            if 'cache_capacity' in node_data:
                total_network_capacity += node_data['cache_capacity']
        if total_network_capacity == 0: return 0.0
        return (self.total_cache_used / total_network_capacity) * 100

    def get_mobile_cpu_utilization_percentage(self):
        total_mobile_capacity = 0.0
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cpu_capacity' in node_data and not self._is_gpu_node(node_id):
                total_mobile_capacity += node_data['cpu_capacity']
        if total_mobile_capacity == 0: return 0.0
        return (self.mobile_cpu_used / total_mobile_capacity) * 100

    def get_mobile_gpu_utilization_percentage(self):
        total_mobile_capacity = 0.0
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cpu_capacity' in node_data and self._is_gpu_node(node_id):
                total_mobile_capacity += node_data['cpu_capacity']
        if total_mobile_capacity == 0: return 0.0
        return (self.mobile_gpu_used / total_mobile_capacity) * 100

    def get_mobile_cache_utilization_percentage(self):
        total_mobile_capacity = 0.0
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cache_capacity' in node_data:
                total_mobile_capacity += node_data['cache_capacity']
        if total_mobile_capacity == 0: return 0.0
        return (self.mobile_cache_used / total_mobile_capacity) * 100

    # =========================================================================
    # 8. JAIN'S FAIRNESS INDEX (JFI)
    # =========================================================================

    def calculate_jain_fairness(self, utilizations):
        if not utilizations: return 1.0
        n = len(utilizations)
        sum_of_values = sum(utilizations)
        sum_of_squares = sum(x * x for x in utilizations)
        if sum_of_squares == 0: return 1.0
        return (sum_of_values ** 2) / (n * sum_of_squares)

    def get_cpu_jain_fairness(self):
        cpu_utilizations = []
        for node_id, node_data in self.graph.nodes(data=True):
            if not self._is_gpu_node(node_id) and node_data.get('cpu_capacity', 0) > 0:
                cpu_utilizations.append(node_data['cpu_used'] / node_data['cpu_capacity'])
        for node_id, node_data in self.md_graph.nodes(data=True):
            if not self._is_gpu_node(node_id) and node_data.get('cpu_capacity', 0) > 0:
                cpu_utilizations.append(node_data['cpu_used'] / node_data['cpu_capacity'])
        return self.calculate_jain_fairness(cpu_utilizations)

    def get_gpu_jain_fairness(self):
        gpu_utilizations = []
        for node_id, node_data in self.graph.nodes(data=True):
            if self._is_gpu_node(node_id) and node_data.get('cpu_capacity', 0) > 0:
                gpu_utilizations.append(node_data['cpu_used'] / node_data['cpu_capacity'])
        for node_id, node_data in self.md_graph.nodes(data=True):
            if self._is_gpu_node(node_id) and node_data.get('cpu_capacity', 0) > 0:
                gpu_utilizations.append(node_data['cpu_used'] / node_data['cpu_capacity'])
        return self.calculate_jain_fairness(gpu_utilizations)

    def get_cache_jain_fairness(self):
        cache_utilizations = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('cache_capacity', 0) > 0:
                cache_utilizations.append(node_data['cache_used'] / node_data['cache_capacity'])
        for node_id, node_data in self.md_graph.nodes(data=True):
            if node_data.get('cache_capacity', 0) > 0:
                cache_utilizations.append(node_data['cache_used'] / node_data['cache_capacity'])
        return self.calculate_jain_fairness(cache_utilizations)

    def get_bandwidth_jain_fairness(self):
        bw_utilizations = []
        for u, v, edge_data in self.graph.edges(data=True):
            if edge_data.get('bandwidth_capacity', 0) > 0:
                bw_utilizations.append(edge_data['bandwidth_used'] / edge_data['bandwidth_capacity'])
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == 'router' and node_data.get('w_channel_capacity', 0) > 0:
                bw_utilizations.append(node_data['w_channel_used'] / node_data['w_channel_capacity'])
        return self.calculate_jain_fairness(bw_utilizations)

    # =========================================================================
    # 9. OUTPUT E DEBUG
    # =========================================================================

    def print_network(self):
        print("\n--- Nós ---")
        for node, data in self.graph.nodes(data=True):
            print(f"{node} -> {data}")
        print("\n--- Arestas ---")
        for u, v, data in self.graph.edges(data=True):
            print(f"{u} <-> {v} -> {data}")

    def print_out_nodes_information(self, failure_cpu=None, failure_cache=None):
        total_cpu_util = self.get_total_system_utilization_cpu_rate() * 100
        total_gpu_util = self.get_total_system_utilization_gpu_rate() * 100
        total_processing_util = self.get_total_system_processing_utilization_rate() * 100

        print(f"Total processing util.   : {total_processing_util:.3f}%")
        print(f"Total System CPU util.   : {total_cpu_util:.3f}%")
        print(f"Total System GPU util.   : {total_gpu_util:.3f}%")
        print(f"Network CPU util         : {self.get_network_cpu_utilization_percentage():.3f}%")
        print(f"Network GPU util         : {self.get_network_gpu_utilization_percentage():.3f}%")
        print(f"Mobile CPU util          : {self.get_mobile_cpu_utilization_percentage():.3f}%")

        if self.total_cpu_requested > 0:
            cpu_saving_rate = (self.total_cpu_saved / self.total_cpu_requested) * 100
            print(f"CPU Saving Rate          : {cpu_saving_rate:.3f}%")
        else:
            print("CPU Saving Rate          : N/A (No CPU requested)")

        if self.total_gpu_requested > 0:
            gpu_saving_rate = (self.total_gpu_saved / self.total_gpu_requested) * 100
            print(f"GPU Saving Rate          : {gpu_saving_rate:.3f}%")
        else:
            print("GPU Saving Rate          : N/A (No GPU requested)")

        cache_util = (self.total_cache_used + self.mobile_cache_used) / self.total_cache_capacity * 100
        print(f"Cache utilization        : {cache_util:.3f}%")

        if self.total_cache_requested > 0:
            cache_saving_rate = (self.total_cache_saved / self.total_cache_requested) * 100
            print(f"Cache Saving Rate        : {cache_saving_rate:.3f}%")
        else:
            print("Cache Saving Rate        : N/A")

    def print_out_edges_information(self, failure_band=None):
        bw_util = str(round(self.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity * 100, 3)) + '%'
        if failure_band is None:
            print("Bandwidth utilization: ", bw_util)
        else:
            print("Bandwidth utilization: ", bw_util, end=" ")
            print(f"     Failure for Band: {failure_band}%")

    def print_out_acceptance_information(self, success_arr):
        if len(success_arr) != 0:
            print("Acceptance: ", str(round(np.mean(success_arr) * 100, 3)) + '%', end=" ")

    # =========================================================================
    # 10. MÉTODOS AUXILIARES (HELPERS)
    # =========================================================================

    def update(self):
        pass

    def get_shareable_sfs(self):
        return self.shared_sfs

    def get_sfc_by_id(self, sfc_id):
        return self.sfc_dict[sfc_id]
    
    def get_number_actives_sfcs(self):
        return len(self.sfc_dict)

    def get_acceptance_rate(self, success_arr):
        if len(success_arr) != 0:
            media = np.mean(success_arr)
            media_porc = media * 100
            return media_porc

    def is_shareable(self, service_name):
        if self.shareable_node:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False

    def is_mobile_node(self, node):
        return node in self.md_graph

    def _is_gpu_node(self, node_id):
        """Verifica se um nó é uma GPU com base no seu ID (terminando em .1)."""
        id_str = str(node_id)
        return id_str.endswith(".1")

# =========================================================================
# TESTE PRINCIPAL
# =========================================================================
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