import math
import re
import random
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Any, Optional

import networkx as nx
import numpy as np

from core.vnf import VNF

# =========================================================================
# CLASSE AUXILIAR DE MÉTRICAS (Refatoração SRP)
# =========================================================================

@dataclass
class NetworkMetrics:
    """
    Centraliza o gerenciamento de métricas dinâmicas da rede.
    Substitui as variáveis soltas da classe Net2 para garantir consistência.
    """
    # --- CPU ---
    total_cpu_requested: float = 0.0
    total_cpu_used: float = 0.0
    total_cpu_saved: float = 0.0
    mobile_cpu_used: float = 0.0

    # --- GPU ---
    total_gpu_requested: float = 0.0
    total_gpu_used: float = 0.0
    total_gpu_saved: float = 0.0
    mobile_gpu_used: float = 0.0

    # --- Cache ---
    total_cache_requested: float = 0.0
    total_cache_used: float = 0.0
    total_cache_saved: float = 0.0
    mobile_cache_used: float = 0.0

    # --- Banda e Outros ---
    total_bandwidth_used: float = 0.0
    shared_vnfs_count: int = 0


# =========================================================================
# FUNÇÕES E CONSTANTES GLOBAIS
# =========================================================================

def extrair_sessao(s):
    match = re.search(r"p\d+_(\d+)(?:_|$)", s)
    return match.group(1) if match else None


SHAREABLE_PREFIXES = ('IA_DET_FT_', 'RE_region_', 'MA_region_')


# =========================================================================
# CLASSE PRINCIPAL NET2
# =========================================================================

class Net2:
    def __init__(self):
        # --- Inicialização de Grafos (Topologia) ---
        self.graph = nx.Graph()
        self.md_graph = nx.Graph()

        # --- Estruturas de Dados de Controle e Roteamento ---
        self.sfc_dict = {}
        self.sfc_route_info = {}        # Mapeamento: sfc_id -> route_info
        self.nodes_reliability = {}

        # Parâmetros de Confiabilidade e Estresse
        self.alpha_stress = {'default': 0.0, 'a': 0.0, 'b': 0.0, 'c': 0.0}
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

        # --- Gerenciamento Centralizado de Métricas (Refatorado) ---
        self.metrics = NetworkMetrics()

        # --- Capacidades Totais da Infraestrutura ---
        # Representam o "teto" físico da rede
        self.total_cpu_capacity = 0.00
        self.total_gpu_capacity = 0.00
        self.total_cache_capacity = 0.00
        self.total_bandwidth_capacity = 0.00

    # =========================================================================
    # 1. GERENCIAMENTO DE TOPOLOGIA (NÓS E ARESTAS)
    # =========================================================================

    def detach_vnf_from_route_record(self, sfc_id: str, vnf_name: str) -> None:
        if sfc_id in self.sfc_route_info:
            if vnf_name in self.sfc_route_info[sfc_id]:
                del self.sfc_route_info[sfc_id][vnf_name]

    def set_sharing_params(self, share_arg: str) -> None:
        """
        Injeta a configuração global de reuso (SF Sharing) definida no CLI.
        Substitui o comportamento hardcoded.
        """
        # Garante que qualquer variação de 'Y', 'y', etc., seja interpretada corretamente
        self.shareable_node = (str(share_arg).lower() == 'y')

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
        self.md_graph.remove_node(node_id)

    def add_edge(self, node1, node2, bandwidth_capacity=1000.00, latency=1):
        self.graph.add_edge(node1, node2,
                            bandwidth_capacity=bandwidth_capacity,
                            bandwidth_used=0.00,
                            bandwidth_reserved=0.00,
                            latency=latency,
                            services_in_transit={})

    def connect_mobile_user(self, user_id, router_id, cpu_capacity):
        self.add_node(user_id, 'mobile_device', cpu_capacity=cpu_capacity)
        self.add_edge(user_id, router_id, bandwidth_capacity=100, latency=5)

    # =========================================================================
    # 2. ALOCAÇÃO E CICLO DE VIDA DE SFC
    # =========================================================================

    def deploy_sfc(self, sfc, route_info):
        sfc_id = sfc.id
        is_backup_sfc = getattr(sfc, 'is_backup', False)

        if sfc_id not in self.sfc_dict:
            self.sfc_dict[sfc_id] = sfc

        if sfc_id not in self.sfc_route_info:
            self.sfc_route_info[sfc_id] = route_info

        for ms_name, path in route_info.items():
            if ms_name in ['src', 'dst'] or "virt" in ms_name:
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
                        self.allocate_bandwidth(u, v, bw_req, ms_name, is_backup=is_backup_sfc)
        return True

    def undeploy_sfc(self, sfc_id):
        """
        Remove a SFC da rede.

        Aplica o padrão EAFP e garante atomicidade na remoção lógica.
        Mesmo se falhar ao liberar recursos físicos (nó caiu, link sumiu),
        o registro lógico da SFC é removido para evitar vazamento de memória (Zumbis).
        """
        # Guard clause para evitar processamento desnecessário
        if sfc_id not in self.sfc_dict:
            return

        try:
            sfc = self.sfc_dict[sfc_id]
            route_info = self.sfc_route_info.get(sfc_id, {})

            # 1. Tentativa de desalocação física (Best Effort)
            # Iteramos sobre uma cópia da lista para evitar erros de modificação durante iteração
            for ms_name, path in list(route_info.items()):
                if ms_name in ['src', 'dst'] or "virt" in ms_name:
                    continue

                if not path:
                    continue

                node_allocated = path[0]

                # Verificação de existência (Look Before You Leap - necessário aqui para grafos dinâmicos)
                node_exists = (node_allocated in self.graph) or (node_allocated in self.md_graph)

                if node_exists:
                    try:
                        vnf = sfc.get_vnf_by_id(ms_name)
                        self.deallocate_microservice(node_allocated, sfc_id, vnf)
                    except Exception as e:
                        # Logamos o erro mas NÃO paramos o processo de undeploy
                        if self.verbose:
                            print(f"[Net2] Aviso: Falha parcial ao desalocar VNF {ms_name} de {sfc_id}: {e}")

                # Liberação de banda (encapsulada para não quebrar o loop)
                if len(path) > 1:
                    self._safe_release_bandwidth(path, ms_name)

        except Exception as critical_error:
            if self.verbose:
                print(f"[Net2] Erro crítico durante undeploy de {sfc_id}: {critical_error}")

        finally:
            # 2. Limpeza Lógica Garantida (Critical Section)
            # O bloco finally garante que isso execute independente de exceções acima.
            self._remove_sfc_from_registry(sfc_id)

    def _safe_release_bandwidth(self, path, ms_name):
        """Helper para liberar banda isolando falhas."""
        try:
            for u, v in zip(path[:-1], path[1:]):
                u_exists = (u in self.graph or u in self.md_graph)
                v_exists = (v in self.graph or v in self.md_graph)

                if u_exists and v_exists:
                    if isinstance(v, str):
                        self.release_wireless_bandwidth(u, v, ms_name)
                    elif isinstance(u, str):
                        self.release_wireless_bandwidth(v, u, ms_name)
                    else:
                        self.release_bandwidth(u, v, ms_name)
        except ValueError:
            pass  # Ignora erros de links que já não existem

    def _remove_sfc_from_registry(self, sfc_id):
        """Remove registros internos para manter consistência."""
        if sfc_id in self.sfc_dict:
            del self.sfc_dict[sfc_id]

        if sfc_id in self.sfc_route_info:
            del self.sfc_route_info[sfc_id]

        # Revalidação defensiva de métricas
        self.metrics.total_cpu_used = max(0.0, self.metrics.total_cpu_used)
        self.metrics.total_cache_used = max(0.0, self.metrics.total_cache_used)
        self.metrics.total_bandwidth_used = max(0.0, self.metrics.total_bandwidth_used)

    def undeploy_specific_vnf_context(self, sfc_id, vnf_id_to_remove):
        if sfc_id not in self.sfc_dict:
            return False

        sfc = self.sfc_dict[sfc_id]
        route_info = self.sfc_route_info.get(sfc_id)

        if not route_info:
            return False

        vnf_target = sfc.get_vnf_by_id(vnf_id_to_remove)
        if not vnf_target:
            return False

        vnf_prev = sfc.get_previous_vnf(vnf_target)

        if vnf_id_to_remove in route_info:
            path = route_info[vnf_id_to_remove]
            if path:
                node_allocated = path[0]
                try:
                    self.deallocate_microservice(node_allocated, sfc_id, vnf_target)
                except ValueError:
                    pass

                if len(path) > 1:
                    for u, v in zip(path[:-1], path[1:]):
                        try:
                            if isinstance(v, str):
                                self.release_wireless_bandwidth(u, v, vnf_id_to_remove)
                            elif isinstance(u, str):
                                self.release_wireless_bandwidth(v, u, vnf_id_to_remove)
                            else:
                                self.release_bandwidth(u, v, vnf_id_to_remove)
                        except ValueError:
                            pass
            del route_info[vnf_id_to_remove]

        if vnf_prev and vnf_prev.id != 'src':
            if vnf_prev.id in route_info:
                path_prev = route_info[vnf_prev.id]
                if len(path_prev) > 1:
                    for u, v in zip(path_prev[:-1], path_prev[1:]):
                        try:
                            if isinstance(v, str):
                                self.release_wireless_bandwidth(u, v, vnf_prev.id)
                            elif isinstance(u, str):
                                self.release_wireless_bandwidth(v, u, vnf_prev.id)
                            else:
                                self.release_bandwidth(u, v, vnf_prev.id)
                        except ValueError:
                            pass

                if path_prev:
                    route_info[vnf_prev.id] = [path_prev[0]]

        return True

    # =========================================================================
    # 3. ALOCAÇÃO DE RECURSOS (COMPUTACIONAIS E REDE)
    # =========================================================================

    def allocate_microservice(self, sfc, vnf, node_id):
        sfc_id = sfc.id

        if hasattr(sfc, 'session_id'):
            session = sfc.session_id
        else:
            parts = sfc_id.split("_")
            if "backup" in sfc_id:
                session = parts[3] if len(parts) > 3 else parts[-1]
            else:
                session = parts[-1]

        mobile = False
        if node_id in self.md_graph:
            node = self.md_graph.nodes[node_id]
            mobile = True
        elif node_id in self.graph:
            node = self.graph.nodes[node_id]
            mobile = False
        else:
            raise ValueError(f"Nó {node_id} não encontrado.")

        if not node.get('is_active', True):
            raise ValueError(f"Falha critica: nó {node_id} inativo")

        service_id = vnf.id
        cpu_required = vnf.get_cpu_request()
        cache_required = vnf.get_cache_request()

        is_gpu = self._is_gpu_node(node_id)

        # Atualização de Metrics (REQUESTED)
        if is_gpu:
            self.metrics.total_gpu_requested = round(self.metrics.total_gpu_requested + cpu_required, 2)
        else:
            self.metrics.total_cpu_requested = round(self.metrics.total_cpu_requested + cpu_required, 2)

        self.metrics.total_cache_requested = round(self.metrics.total_cache_requested + cache_required, 2)

        if node['type'] not in ['server', 'mobile_device']:
            raise ValueError(f"Tipo de nó inválido: {node['type']}")

        # --- Helper Interno com acesso correto a self.metrics ---
        def put_resource(cpu_req, cache_req, is_mobile):
            node['cpu_used'] = round(node['cpu_used'] + cpu_req, 2)
            node['cache_used'] = round(node['cache_used'] + cache_req, 2)

            if is_gpu:
                if is_mobile:
                    self.metrics.mobile_gpu_used = round(self.metrics.mobile_gpu_used + cpu_req, 2)
                else:
                    self.metrics.total_gpu_used = round(self.metrics.total_gpu_used + cpu_req, 2)
            else:
                if is_mobile:
                    self.metrics.mobile_cpu_used = round(self.metrics.mobile_cpu_used + cpu_req, 2)
                else:
                    self.metrics.total_cpu_used = round(self.metrics.total_cpu_used + cpu_req, 2)

            if is_mobile:
                self.metrics.mobile_cache_used = round(self.metrics.mobile_cache_used + cache_req, 2)
            else:
                self.metrics.total_cache_used = round(self.metrics.total_cache_used + cache_req, 2)
        # ----------------------

        clean_current_id = service_id.replace("_b", "")

        compatible_instance_found = False
        if self.is_shareable(service_id) or self.is_shareable(clean_current_id):
            for (existing_id, existing_session) in node['services'].keys():
                existing_clean = existing_id.replace("_b", "")
                if existing_clean == clean_current_id and existing_session == session:
                    compatible_instance_found = True
                    break

        if sfc_id not in node['sfcs_list']:
            node['sfcs_list'].append(sfc_id)

        service_key = (service_id, session)

        # CASO 1: Match Exato (Reuso)
        if service_key in node['services']:
            node['services'][service_key]['copys'] += 1

            if not self.is_shareable(service_id):
                if node['cpu_used'] + cpu_required > node['cpu_capacity'] or \
                   node['cache_used'] + cache_required > node['cache_capacity']:
                    node['services'][service_key]['copys'] -= 1
                    if sfc_id in node['sfcs_list']:
                        node['sfcs_list'].remove(sfc_id)
                    raise ValueError(f"Sem capacidade no nó {node_id} para instância não-shared.")
                put_resource(cpu_required, cache_required, mobile)
            else:
                # É compartilhável -> Saving
                if is_gpu:
                    self.metrics.total_gpu_saved = round(self.metrics.total_gpu_saved + cpu_required, 2)
                else:
                    self.metrics.total_cpu_saved = round(self.metrics.total_cpu_saved + cpu_required, 2)

                self.metrics.total_cache_saved = round(self.metrics.total_cache_saved + cache_required, 2)
                self.metrics.shared_vnfs_count += 1

        # CASO 2: Nova Instância
        else:
            cost_cpu = cpu_required
            cost_cache = cache_required

            if compatible_instance_found:
                # Reuso de binário/imagem
                cost_cpu = 0
                cost_cache = 0

                if is_gpu:
                    self.metrics.total_gpu_saved = round(self.metrics.total_gpu_saved + cpu_required, 2)
                else:
                    self.metrics.total_cpu_saved = round(self.metrics.total_cpu_saved + cpu_required, 2)

                self.metrics.total_cache_saved = round(self.metrics.total_cache_saved + cache_required, 2)
                self.metrics.shared_vnfs_count += 1

            if node['cpu_used'] + cost_cpu > node['cpu_capacity'] or \
               node['cache_used'] + cost_cache > node['cache_capacity']:
                if sfc_id in node['sfcs_list']:
                    node['sfcs_list'].remove(sfc_id)
                raise ValueError(f"Sem capacidade no nó {node_id} para novo serviço.")

            node['services'][service_key] = {'cpu': cost_cpu, 'cache': cost_cache, 'copys': 1}
            put_resource(cost_cpu, cost_cache, mobile)

            if self.is_shareable(service_id):
                node['reuse'].append(vnf)

    def deallocate_microservice(self, node_id: str, sfc_id: str, vnf: VNF) -> None:
        """
        Remove recursos alocados.
        Correção: EAFP - Se o serviço não existe, retorna silenciosamente.
        """
        if node_id in self.md_graph:
            node = self.md_graph.nodes[node_id]
            is_mobile = True
        elif node_id in self.graph:
            node = self.graph.nodes[node_id]
            is_mobile = False
        else:
            return  # Nó não existe mais

        service_id = vnf.id
        session_id = extrair_sessao(sfc_id)
        # Mantendo a sua estrutura de chave original
        service_key = (service_id, session_id)

        # --- CORREÇÃO EAFP ---
        if service_key not in node.get('services', {}):
            # O serviço já foi removido ou nunca esteve aqui. Tudo certo.
            return

        cpu_req = vnf.get_cpu_request()
        cache_req = vnf.get_cache_request()
        is_gpu_node = self._is_gpu_node(node_id)
        service_info = node['services'][service_key]

        service_info['copys'] -= 1
        remove_physical_instance = (service_info['copys'] <= 0)
        is_shareable_service = self.is_shareable(service_id)

        # Atualização Request Metrics
        if is_gpu_node:
            self.metrics.total_gpu_requested -= cpu_req
        else:
            self.metrics.total_cpu_requested -= cpu_req
        self.metrics.total_cache_requested -= cache_req

        # ==============================================================================
        # ATUALIZAÇÃO SAVINGS (CACHE CORRIGIDO)
        # ==============================================================================

        # Verifica se essa instância tinha custo reduzido/zero (subsidiada)
        # Se stored_cost (ex: 0) < request (ex: 10), ela gerou economia na entrada.
        stored_cache_cost = service_info.get('cache', 0.0)
        stored_cpu_cost = service_info.get('cpu', 0.0)

        # 2. Avalia se a instância foi subsidiada na entrada (custou menos que o requisitado)
        was_subsidized_cache = stored_cache_cost < cache_req
        was_subsidized_cpu = stored_cpu_cost < cpu_req

        # 3. Estorno de Cache
        if not remove_physical_instance or was_subsidized_cache:
            self.metrics.total_cache_saved = max(0.0, self.metrics.total_cache_saved - cache_req)

        # 4. Estorno de Processamento (CPU / GPU) e Contadores
        if not remove_physical_instance or was_subsidized_cpu:
            if is_gpu_node:
                self.metrics.total_gpu_saved = max(0.0, self.metrics.total_gpu_saved - cpu_req)
            else:
                self.metrics.total_cpu_saved = max(0.0, self.metrics.total_cpu_saved - cpu_req)

            # Só reduzimos o contador de VNFs compartilhadas se realmente for um serviço shareable
            if is_shareable_service:
                self.metrics.shared_vnfs_count = max(0, self.metrics.shared_vnfs_count - 1)

        # ==============================================================================

        # Atualização Listas e Remoção Física
        if sfc_id in node['sfcs_list']:
            node['sfcs_list'].remove(sfc_id)

        if remove_physical_instance:
            del node['services'][service_key]

            # Atualiza uso físico
            node['cpu_used'] = round(node['cpu_used'] - cpu_req, 2)
            node['cache_used'] = round(node['cache_used'] - cache_req, 2)

            if is_gpu_node:
                if is_mobile:
                    self.metrics.mobile_gpu_used -= cpu_req
                else:
                    self.metrics.total_gpu_used -= cpu_req
            else:
                if is_mobile:
                    self.metrics.mobile_cpu_used -= cpu_req
                else:
                    self.metrics.total_cpu_used -= cpu_req

            if is_mobile:
                self.metrics.mobile_cache_used -= cache_req
            else:
                self.metrics.total_cache_used -= cache_req

            if is_shareable_service and vnf in node.get('reuse', []):
                node['reuse'].remove(vnf)

    def allocate_bandwidth(self, node1, node2, bw_required, ms_name, is_backup=False):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta entre {node1} e {node2} não existe.")

        edge = self.graph.edges[node1, node2]
        current_reserved = edge.get('bandwidth_reserved', 0.0)
        total_committed = edge['bandwidth_used'] + current_reserved

        if ms_name in edge['services_in_transit']:
            existing_bw = edge['services_in_transit'][ms_name]['bw_used']
            total_committed -= existing_bw

        if total_committed + bw_required > edge['bandwidth_capacity']:
            raise ValueError(f"Banda insuficiente entre {node1} e {node2}.")

        if ms_name in edge['services_in_transit']:
            edge['services_in_transit'][ms_name]['copys'] += 1
            edge['services_in_transit'][ms_name]['bw_used'] += bw_required
        else:
            edge['services_in_transit'][ms_name] = {'copys': 1, 'bw_used': bw_required, 'is_backup': is_backup}

        if is_backup:
            edge['bandwidth_reserved'] = edge.get('bandwidth_reserved', 0.0) + bw_required
        else:
            edge['bandwidth_used'] += bw_required
            # Atualização de metrics
            self.metrics.total_bandwidth_used += bw_required

        return self.get_link_latency(node1, node2)

    def get_shortest_path_with_bw(self, source: str, target: str, required_bw: float) -> Optional[List[str]]:
        """
        Retorna o menor caminho considerando apenas links com banda suficiente.
        Utiliza Subgraph View para evitar acoplamento com a estrutura interna das arestas.
        """
        if source == target:
            return [source]

        # Função de filtro para o NetworkX (Isola a lógica de topologia)
        def filter_edge(u, v):
            # Verifica arestas no grafo principal
            if self.graph.has_edge(u, v):
                edge = self.graph[u][v]
                free_bw = edge['bandwidth_capacity'] - edge['bandwidth_used']
                return free_bw >= required_bw
            return False

        # Cria uma 'view' temporária do grafo apenas com links válidos
        # Isso é leve e não copia os dados
        valid_view = nx.subgraph_view(self.graph, filter_edge=filter_edge)

        try:
            # Busca caminho na visualização filtrada
            return nx.dijkstra_path(valid_view, source, target, weight='latency')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def release_bandwidth(self, node1, node2, ms_name):
        """
        Libera banda de forma segura e idempotente (EAFP).
        Não levanta erro se o serviço já não estiver no link.
        """
        if not self.graph.has_edge(node1, node2):
            return

        edge = self.graph.edges[node1, node2]
        services = edge.get('services_in_transit', {})

        # --- CORREÇÃO: EAFP (Silent Return) ---
        if ms_name not in services:
            return

        entry = services[ms_name]
        bw_to_release = entry['bw_used']
        is_backup_entry = entry.get('is_backup', False)

        entry['copys'] -= 1

        if is_backup_entry:
            edge['bandwidth_reserved'] = max(0.0, edge.get('bandwidth_reserved', 0.0) - bw_to_release)
        else:
            edge['bandwidth_used'] = max(0.0, edge['bandwidth_used'] - bw_to_release)
            self.metrics.total_bandwidth_used = max(0.0, self.metrics.total_bandwidth_used - bw_to_release)

        if entry['copys'] <= 0:
            del services[ms_name]

    def allocate_wireless_bandwidth(self, node1, node2, bw_required, ms_name):
        router = self.graph.nodes[node1]
        if ms_name in router['w_services']:
            router['w_services'][ms_name]['copys'] += 1
            router['w_channel_used'] += bw_required
            self.metrics.total_bandwidth_used += bw_required
        else:
            if router['w_channel_used'] + bw_required > router['w_channel_capacity']:
                raise ValueError(f"Banda insuficiente entre {node1} e {node2}.")
            router['w_services'][ms_name] = {'copys': 1, 'bw_used': bw_required}
            router['w_channel_used'] += bw_required
            self.metrics.total_bandwidth_used += bw_required

        return 0

    def release_wireless_bandwidth(self, node1, node2, ms_name):
        router = self.graph.nodes[node1]
        services = router.get('w_services', {})

        if ms_name not in services:
            raise ValueError(f"Serviço {ms_name} não está em trânsito.")

        services[ms_name]['copys'] -= 1
        bw_to_release = services[ms_name]['bw_used']
        router['w_channel_used'] = max(0.0, router['w_channel_used'] - bw_to_release)

        self.metrics.total_bandwidth_used = max(0.0, self.metrics.total_bandwidth_used - bw_to_release)

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
    
    def calculate_sfc_total_latency(self, sfc_id: str) -> float:
        """Calcula a latência total exata de uma SFC (Processamento + Rede)."""
        if sfc_id not in self.sfc_dict or sfc_id not in self.sfc_route_info:
            return 0.0

        sfc = self.sfc_dict[sfc_id]
        route_info = self.sfc_route_info[sfc_id]
        total_latency = 0.0

        current_vnf = sfc.get_vnf_by_id('src')
        while current_vnf and current_vnf.id != 'dst':
            next_vnf = sfc.get_next_vnf(current_vnf)
            if not next_vnf or next_vnf.id == 'dst':
                break

            allocation_path = route_info.get(next_vnf.id)
            if not allocation_path:
                current_vnf = next_vnf
                continue

            # 1. Latência Computacional
            allocated_node = allocation_path[0]
            if self.is_mobile_node(allocated_node):
                ips = self.md_graph.nodes[allocated_node]['ips']
            else:
                ips = self.graph.nodes[allocated_node]['ips']

            packet = next_vnf.get_income_interface_bandwidth() / 60 * 1e6
            comp_latency = packet * 10 * 1000 / ips
            total_latency += comp_latency

            # 2. Latência de Rede (Links fixos e 5G)
            if len(allocation_path) > 1:
                for i in range(len(allocation_path) - 1):
                    u = allocation_path[i]
                    v = allocation_path[i + 1]
                    total_latency += self.calculate_latency_betwen_nodes(self.graph, u, v, next_vnf)

            current_vnf = next_vnf

        return total_latency

    # =========================================================================
    # 5. SIMULAÇÃO DE FALHAS E RECUPERAÇÃO
    # =========================================================================

    def calculate_average_system_reliability(self, backups_dict: Dict[str, List[Dict]] = None) -> float:
        """
        Calcula a confiabilidade média de TODAS as SFCs primárias na rede.

        Args:
            backups_dict: Dicionário vindo do BackupManager.sfcs_backups_instatiated
                        Formato: {'sfc_id': [{'vnf_id': '...', 'route_info': ...}, ...]}

        Returns:
            float: Média de confiabilidade do sistema (0.0 a 1.0).
        """
        if not self.sfc_dict:
            return 0.0

        total_reliability_sum = 0.0
        active_primary_sfcs_count = 0

        # Se não foi passado dicionário de backups, assume vazio (calcula apenas confiabilidade física)

        if backups_dict == 0 or {}:
            backups_dict = {}
        else:
            backups_dict = backups_dict or {}

        for sfc_id, sfc in self.sfc_dict.items():
            # 1. FILTRO: Ignora SFCs que são puramente backups
            if "backup" in sfc_id or getattr(sfc, 'is_backup', False):
                continue

            # Fail-fast: Se a SFC não tem rota definida, confiabilidade é 0 ou ignora
            if sfc_id not in self.sfc_route_info:
                continue

            route_info = self.sfc_route_info[sfc_id]

            # ---------------------------------------------------------
            # 2. MAPEAMENTO: Quais VNFs desta SFC possuem backup ativo?
            # ---------------------------------------------------------
            # Mapa: VNF_ID -> Confiabilidade do Nó de Backup
            vnf_backup_reliability_map = {}

            if sfc_id in backups_dict:
                for backup_entry in backups_dict[sfc_id]:
                    vnf_target_id = backup_entry.get('vnf_id')
                    bk_route = backup_entry.get('route_info', {})

                    # Encontra o nó físico onde o backup reside (busca chave terminada em '_b')
                    # Exemplo de chave na rota: 'firewall_b' -> ['node_10']
                    bk_node_list = next((v for k, v in bk_route.items() if k.endswith('_b') and v), None)

                    if vnf_target_id and bk_node_list:
                        bk_node_id = bk_node_list[0]
                        # Obtém a confiabilidade real do nó de backup
                        vnf_backup_reliability_map[vnf_target_id] = self.get_node_reliability(bk_node_id)

            # ---------------------------------------------------------
            # 3. AGRUPAMENTO POR DOMÍNIO DE FALHA (NÓ FÍSICO)
            # Se um nó cai, todas as VNFs dele caem.
            # ---------------------------------------------------------
            node_groups = defaultdict(list)

            for vnf_id, path in route_info.items():
                if vnf_id in ['src', 'dst'] or not path:
                    continue
                # path[0] é o nó físico primário
                primary_node = path[0]
                node_groups[primary_node].append(vnf_id)

            # ---------------------------------------------------------
            # 4. CÁLCULO DE CONFIABILIDADE DA SFC (Série-Paralelo)
            # ---------------------------------------------------------
            sfc_reliability = 1.0

            for primary_node, vnfs_list in node_groups.items():
                # Confiabilidade do nó primário
                try:
                    r_primary = self.get_node_reliability(primary_node)
                except (KeyError, AttributeError):
                    r_primary = 1.0  # Fallback seguro

                # Verifica se o GRUPO INTEIRO está protegido
                # Para o serviço sobreviver à queda do nó, TODAS as VNFs alocadas nele
                # precisam ter um backup operante em outro lugar.
                all_vnfs_protected = True

                # Probabilidade de TODOS os backups falharem simultaneamente
                # Inicializa com 1.0 (neutro para multiplicação)
                prob_backups_fail_combined = 1.0

                for vnf_id in vnfs_list:
                    r_backup = vnf_backup_reliability_map.get(vnf_id, 0.0)

                    if r_backup > 0.0:
                        # Se tem backup, acumula a chance de falha dele
                        # P(Falha Backup) = 1 - R_backup
                        prob_backups_fail_combined *= (1.0 - r_backup)
                    else:
                        # Se uma única VNF do nó não tem backup, o grupo não sobrevive à queda do nó
                        all_vnfs_protected = False
                        # Não precisamos verificar o resto das VNFs deste nó para fins de lógica Série
                        # mas continuamos para consistência se necessário.

                # APLICAÇÃO DA FÓRMULA RBD
                if all_vnfs_protected:
                    # Sistema Paralelo (Redundância):
                    # O estágio falha apenas se (Primário falhar) E (Backups falharem)
                    # P(Estágio Falhar) = P(Primário Falhar) * P(Backups Falharem)
                    prob_primary_fail = 1.0 - r_primary
                    prob_stage_fail = prob_primary_fail * prob_backups_fail_combined

                    stage_reliability = 1.0 - prob_stage_fail
                else:
                    # Sistema em Série (Elo mais fraco):
                    # Se o nó primário cair, o serviço para (pois falta backup para algo)
                    stage_reliability = r_primary

                # A confiabilidade total da SFC é o produto da confiabilidade de cada estágio (nó físico)
                sfc_reliability *= stage_reliability

            # Acumula para a média
            total_reliability_sum += sfc_reliability
            active_primary_sfcs_count += 1

        # ---------------------------------------------------------
        # 5. RETORNO DA MÉDIA
        # ---------------------------------------------------------
        if active_primary_sfcs_count == 0:
            return 0.0

        return total_reliability_sum / active_primary_sfcs_count

    def set_node_down(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        node = self.graph.nodes[node_id]
        node["is_active"] = False

        if 'original_cpu_capacity' not in node:
            node['original_cpu_capacity'] = node.get('cpu_capacity', 0)
            node['original_cache_capacity'] = node.get('cache_capacity', 0)

        node['cpu_capacity'] = 0
        node['cache_capacity'] = 0
        print(f"Debug: Nó {node_id} caiu!")

    def restore_node(self, node_id, cpu_capacity=None, cache_capacity=None):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        node = self.graph.nodes[node_id]
        node["is_active"] = True

        restored_cpu = cpu_capacity if cpu_capacity is not None else node.get('original_cpu_capacity', 100)
        restored_cache = cache_capacity if cache_capacity is not None else node.get('original_cache_capacity', 100)

        node['cpu_capacity'] = restored_cpu
        node['cache_capacity'] = restored_cache
        node.pop('original_cpu_capacity', None)
        node.pop('original_cache_capacity', None)

    def set_link_down(self, u, v):
        if not self.graph.has_edge(u, v):
            raise ValueError(f"Link {u}-{v} inexistente.")
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
        for u, v in zip(path[:-1], path[1:]):
            if not self.graph.has_edge(u, v):
                continue
            edge = self.graph.edges[u, v]

            if edge['bandwidth_used'] + bw_required > edge['bandwidth_capacity']:
                print(f"CRITICAL: Falha ao ativar banda backup {u}-{v}.")
                return False

            edge['bandwidth_used'] += bw_required
            self.metrics.total_bandwidth_used += bw_required

            if vnf_id_backup in edge['services_in_transit']:
                edge['services_in_transit'][vnf_id_backup]['bw_used'] += bw_required
            else:
                edge['services_in_transit'][vnf_id_backup] = {'copys': 1, 'bw_used': bw_required}
        return True

    def set_reliability_params(self, args):
        stress_c = getattr(args, 'stress_high', 0.0009)
        stress_b = getattr(args, 'stress_normal', 0.04)
        stress_a = getattr(args, 'stress_low', 0.15)

        self.alpha_stress = {
            'c': max(0.0, stress_c),
            'b': max(0.0, stress_b),
            'a': max(0.0, stress_a),
            'default': max(0.0, stress_b)
        }

        def validate(val): return val if 0.0 <= val <= 1.0 else 0.99
        self.tier_reliability = {
            'c': validate(getattr(args, 'rel_high', 0.9999)),
            'b': validate(getattr(args, 'rel_normal', 0.99)),
            'a': validate(getattr(args, 'rel_low', 0.95)),
            'default': validate(getattr(args, 'rel_normal', 0.99))
        }

    def get_node_reliability(self, node_id):
        node_id_str = str(node_id)
        if node_id_str.endswith(".1"):
            base_id = node_id_str.replace(".1", "")
            gpu_id = node_id_str
        else:
            base_id = node_id_str
            gpu_id = node_id_str + ".1"

        def get_part_data(nid):
            if nid in self.graph:
                node = self.graph.nodes[nid]
                if node.get('is_active', True):
                    cap = node.get('cpu_capacity', 0.0)
                    if cap <= 0:
                        cap = node.get('original_cpu_capacity', 1.0) or 1.0
                    return node.get('cpu_used', 0.0), cap, node.get('level_server', 'default'), True
            return 0.0, 0.0, None, False

        used_cpu, cap_cpu, lvl_cpu, active_cpu = get_part_data(int(base_id))
        used_gpu, cap_gpu, lvl_gpu, active_gpu = get_part_data(float(gpu_id))

        if not active_cpu and not active_gpu:
            return 0.0

        server_level = lvl_cpu if lvl_cpu else (lvl_gpu if lvl_gpu else 'default')
        total_used = used_cpu + used_gpu
        total_capacity = cap_cpu + cap_gpu

        if total_capacity <= 0:
            return 0.0

        level_key = str(server_level).lower()
        base_r = self.tier_reliability.get(level_key, self.tier_reliability['default'])
        alpha = self.alpha_stress.get(level_key, self.alpha_stress['default'])
        global_utilization = total_used / total_capacity
        stress_penalty = global_utilization * alpha

        return max(0.0, base_r - stress_penalty)

    # =========================================================================
    # 6. ALGORITMOS DE CAMINHO MÍNIMO
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
    # 7. GETTERS E MÉTRICAS DE RECURSOS (CORRIGIDOS COM self.metrics)
    # =========================================================================

    def get_node_sfc_vnf_list(self, node_id):
        """
        Retorna uma lista de tuplas (sfc_id, vnf_obj) para todas as VNFs
        alocadas no nó especificado. Usado por estratégias de backup.
        """
        if node_id not in self.graph:
            return []

        # Recupera a lista de IDs de SFCs que passam por este nó
        # (Essa lista é mantida pelo método allocate_microservice)
        node_sfcs = self.graph.nodes[node_id].get('sfcs_list', [])
        result = []

        for sfc_id in node_sfcs:
            # Segurança: verifica se a SFC ainda existe logicamente
            if sfc_id not in self.sfc_dict:
                continue

            sfc = self.sfc_dict[sfc_id]

            # Consulta o roteamento para confirmar quais VNFs específicas
            # desta SFC estão neste nó
            if sfc_id in self.sfc_route_info:
                route_info = self.sfc_route_info[sfc_id]

                for vnf_id, path in route_info.items():
                    # Ignora nós virtuais ou caminhos vazios
                    if vnf_id in ['src', 'dst'] or not path:
                        continue

                    # path[0] é o nó onde a VNF está processando
                    if path[0] == node_id:
                        vnf = sfc.get_vnf_by_id(vnf_id)
                        if vnf:
                            result.append((sfc_id, vnf))

        return result

    def get_node_cpu_used(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        return self.graph.nodes[node_id]['cpu_used']

    def get_node_cpu_free(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        return self.graph.nodes[node_id]['cpu_capacity'] - self.graph.nodes[node_id]['cpu_used']

    def get_node_cpu_capacity(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        return self.graph.nodes[node_id]['cpu_capacity']

    def get_node_cache_used(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        return self.graph.nodes[node_id]['cache_used']

    def get_node_cache_free(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        return self.graph.nodes[node_id]['cache_capacity'] - self.graph.nodes[node_id]['cache_used']
    
    def get_node_is_active(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        return self.graph.nodes[node_id].get('is_active', True)

    def get_node_cache_capacity(self, node_id):
        if node_id not in self.graph:
            raise ValueError(f"Nó {node_id} inexistente.")
        return self.graph.nodes[node_id]['cache_capacity']

    def get_node_sfcs(self, node_id):
        return self.graph.nodes[node_id]["sfcs_list"]

    def get_link_bandwidth_used(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['bandwidth_used']

    def get_link_bandwidth_free(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['bandwidth_capacity'] - self.graph.edges[node1, node2]['bandwidth_used']

    def get_link_bandwidth_capacity(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['bandwidth_capacity']

    def get_link_latency(self, node1, node2):
        if not self.graph.has_edge(node1, node2):
            raise ValueError(f"Aresta inexistente.")
        return self.graph.edges[node1, node2]['latency']

    def get_link_info(self, node1, node2):
        return self.graph.edges[node1, node2]

    # --- Totais Requisitados e Economizados (Corrigido) ---
    def get_total_gpu_request(self): return self.metrics.total_gpu_requested
    def get_total_gpu_saved(self): return self.metrics.total_gpu_saved
    def get_total_cpu_request(self): return self.metrics.total_cpu_requested
    def get_total_cpu_saved(self): return self.metrics.total_cpu_saved
    def get_total_cache_request(self): return self.metrics.total_cache_requested
    def get_total_cache_saved(self): return self.metrics.total_cache_saved

    # --- Totais Usados (Corrigido) ---
    def get_cpu_network_used(self): return self.metrics.total_cpu_used
    def get_gpu_network_used(self): return self.metrics.total_gpu_used

    def get_cpu_total_used(self):
        return self.metrics.total_cpu_used + self.metrics.mobile_cpu_used

    def get_network_only_processing_utilization(self):
        """
        Calcula a utilização combinada (CPU + GPU) APENAS da infraestrutura de servidores.
        Ignora dispositivos móveis.
        Retorna float entre 0.0 e >1.0 (se houver sobrecarga).
        """
        # 1. Numerador: Uso apenas da Rede (já excluindo mobile nas métricas)
        # As métricas 'total_*' em Net2 referem-se estritamente à rede fixa/servidores
        network_used_cpu = self.metrics.total_cpu_used
        network_used_gpu = self.metrics.total_gpu_used
        total_load = network_used_cpu + network_used_gpu

        # 2. Denominador: Capacidade Atual dos Servidores (Iterando apenas self.graph)
        total_capacity = 0.0

        for node_id, node_data in self.graph.nodes(data=True):
            # Filtro de segurança: Garante que é um servidor (ignora roteadores se tiverem cap 0)
            if node_data.get('type') == 'server':
                # Se o nó estiver caído (is_active=False), a cpu_capacity será 0.
                # Isso é CORRETO para Admission Control: capacidade cai -> utilização sobe -> bloqueia backups.
                total_capacity += node_data.get('cpu_capacity', 0.0)

        # Fail-safe: Se a capacidade for 0 (colapso total), retorna saturação máxima (1.0 ou mais)
        if total_capacity == 0:
            return 1.0 if total_load == 0 else 999.0

        return total_load / total_capacity

    def get_gpu_total_used(self):
        return self.metrics.total_gpu_used + self.metrics.mobile_gpu_used

    def get_cache_total_used(self):
        return self.metrics.total_cache_used + self.metrics.mobile_cache_used

    def get_cache_used(self):
        return self.metrics.total_cache_used

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

    # --- Taxas de Utilização do Sistema (Corrigidas) ---
    def get_server_cpu_utilization_rate(self):
        if self.total_cpu_capacity == 0:
            return 0.0
        return self.metrics.total_cpu_used / self.total_cpu_capacity

    def get_total_system_utilization_cpu_rate(self):
        total_capacity = self.get_total_system_cpu_capacity()
        if total_capacity == 0:
            return 0.0
        total_used = self.metrics.total_cpu_used + self.metrics.mobile_cpu_used
        return total_used / total_capacity

    def get_total_system_utilization_gpu_rate(self):
        total_capacity = self.get_total_system_gpu_capacity()
        if total_capacity == 0:
            return 0.0
        total_used = self.metrics.total_gpu_used + self.metrics.mobile_gpu_used
        return total_used / total_capacity

    def get_total_system_utilization_cache_rate(self):
        if self.total_cache_capacity == 0:
            return 0.0
        total_used = self.metrics.total_cache_used + self.metrics.mobile_cache_used
        return total_used / self.total_cache_capacity

    def get_total_system_processing_utilization_rate(self):
        total_processing_used = self.get_cpu_total_used() + self.get_gpu_total_used()
        total_processing_capacity = self.get_total_system_cpu_capacity() + self.get_total_system_gpu_capacity()
        if total_processing_capacity == 0:
            return 0.0
        return total_processing_used / total_processing_capacity

    def get_cache_utilization_rate(self):
        return self.metrics.total_cache_used * 1.0 / self.total_cache_capacity

    def get_bandwidth_utilization_rate(self):
        self.update()
        return self.metrics.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity

    # --- Percentuais de Utilização Específicos (Corrigidos) ---
    def get_network_cpu_utilization_percentage(self):
        total_network_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            # Filtra apenas servidores, ignorando GPUs
            if not self._is_gpu_node(node_id):
                # CORREÇÃO: Se o nó caiu, usa a capacidade original
                current_cap = node_data.get('cpu_capacity', 0.0)
                if not node_data.get('is_active', True):
                    current_cap = node_data.get('original_cpu_capacity', current_cap)

                total_network_capacity += current_cap

        if total_network_capacity == 0:
            return 0.0

        # O metrics.total_cpu_used continua contendo a carga dos nós caídos
        # até que o Controller faça o undeploy.
        return (self.metrics.total_cpu_used / total_network_capacity) * 100

    def get_network_gpu_utilization_percentage(self):
        total_network_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            # Filtra apenas nós de GPU
            if self._is_gpu_node(node_id):
                # CORREÇÃO: Usa capacidade original em caso de falha
                current_cap = node_data.get('cpu_capacity', 0.0)
                if not node_data.get('is_active', True):
                    current_cap = node_data.get('original_cpu_capacity', current_cap)

                total_network_capacity += current_cap

        if total_network_capacity == 0:
            return 0.0
        return (self.metrics.total_gpu_used / total_network_capacity) * 100

    def get_processing_network_used_precise(self):
        total_used = 0.0
        total_capacity = 0.0
        for node_id, node in self.graph.nodes(data=True):
            if node.get('type') == 'server' and node.get('is_active', True):
                total_used += node.get('cpu_used', 0.0)
                total_capacity += node.get('cpu_capacity', 0.0)
        if total_capacity == 0:
            return 0.0
        return total_used / total_capacity

    def get_network_cache_utilization_percentage(self):
        total_network_capacity = 0.0
        for node_id, node_data in self.graph.nodes(data=True):
            if 'cache_capacity' in node_data:
                total_network_capacity += node_data['cache_capacity']
        if total_network_capacity == 0:
            return 0.0
        return (self.metrics.total_cache_used / total_network_capacity) * 100

    def get_mobile_cpu_utilization_percentage(self):
        total_mobile_capacity = 0.0
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cpu_capacity' in node_data and not self._is_gpu_node(node_id):
                total_mobile_capacity += node_data['cpu_capacity']
        if total_mobile_capacity == 0:
            return 0.0
        return (self.metrics.mobile_cpu_used / total_mobile_capacity) * 100

    def get_mobile_gpu_utilization_percentage(self):
        total_mobile_capacity = 0.0
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cpu_capacity' in node_data and self._is_gpu_node(node_id):
                total_mobile_capacity += node_data['cpu_capacity']
        if total_mobile_capacity == 0:
            return 0.0
        return (self.metrics.mobile_gpu_used / total_mobile_capacity) * 100

    def get_mobile_cache_utilization_percentage(self):
        total_mobile_capacity = 0.0
        for node_id, node_data in self.md_graph.nodes(data=True):
            if 'cache_capacity' in node_data:
                total_mobile_capacity += node_data['cache_capacity']
        if total_mobile_capacity == 0:
            return 0.0
        return (self.metrics.mobile_cache_used / total_mobile_capacity) * 100

    # =========================================================================
    # 8. JAIN'S FAIRNESS INDEX (JFI)
    # =========================================================================

    def calculate_jain_fairness(self, utilizations):
        if not utilizations:
            return 1.0
        n = len(utilizations)
        sum_of_values = sum(utilizations)
        sum_of_squares = sum(x * x for x in utilizations)
        if sum_of_squares == 0:
            return 1.0
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

        if self.metrics.total_cpu_requested > 0:
            cpu_saving_rate = (self.metrics.total_cpu_saved / self.metrics.total_cpu_requested) * 100
            print(f"CPU Saving Rate          : {cpu_saving_rate:.3f}%")
        else:
            print("CPU Saving Rate          : N/A (No CPU requested)")

        if self.metrics.total_gpu_requested > 0:
            gpu_saving_rate = (self.metrics.total_gpu_saved / self.metrics.total_gpu_requested) * 100
            print(f"GPU Saving Rate          : {gpu_saving_rate:.3f}%")
        else:
            print("GPU Saving Rate          : N/A (No GPU requested)")

        cache_util = (self.metrics.total_cache_used + self.metrics.mobile_cache_used) / self.total_cache_capacity * 100
        print(f"Cache utilization        : {cache_util:.3f}%")

        if self.metrics.total_cache_requested > 0:
            cache_saving_rate = (self.metrics.total_cache_saved / self.metrics.total_cache_requested) * 100
            print(f"Cache Saving Rate        : {cache_saving_rate:.3f}%")
        else:
            print("Cache Saving Rate        : N/A")

    def print_out_edges_information(self, failure_band=None):
        # Corrigido: Usa self.metrics para bandwidth usada
        bw_util = str(round(self.metrics.total_bandwidth_used * 1.0 / self.total_bandwidth_capacity * 100, 3)) + '%'
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

    def get_number_active_primary_sfcs(self):
        """
        Retorna o número de SFCs ativas EXCLUINDO as de backup.
        Útil para métricas de 'running_sfcs' e cálculos de média por fluxo.
        """
        count = 0
        for sfc_id, sfc_obj in self.sfc_dict.items():
            # Verifica se é backup pelo ID (padrão de string) E pelo atributo do objeto (se existir)
            is_backup_id = "backup" in sfc_id
            is_backup_attr = getattr(sfc_obj, 'is_backup', False)

            if not is_backup_id and not is_backup_attr:
                count += 1
        return count

    def get_acceptance_rate(self, success_arr):
        if len(success_arr) != 0:
            media = np.mean(success_arr)
            media_porc = media * 100
            return media_porc
        
    def get_servers_reliability_dict(self):
        """
        Retorna um dicionário {node_id: reliability} contendo todos os servidores.
        O dicionário mantém a ordem de inserção da MENOR para a MAIOR confiabilidade.
        
        Regra: Se a capacidade total do servidor for 0, a confiabilidade é 0.0.
        """
        # Lista temporária para permitir a ordenação
        temp_list = []

        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'server':
                
                # Regra: Capacidade 0 -> Confiabilidade 0
                if data.get('cpu_capacity', 0) == 0:
                    reliability = 0.0
                else:
                    reliability = self.get_node_reliability(node_id)
                
                temp_list.append((node_id, reliability))

        # Ordena a lista temporária pela confiabilidade (menor -> maior)
        temp_list.sort(key=lambda x: x[1])

        # Converte para dicionário (preservando a ordem de inserção)
        return {node_id: rel for node_id, rel in temp_list}

    def is_shareable(self, service_name):
        if self.shareable_node:
            return service_name.startswith(SHAREABLE_PREFIXES)
        else:
            return False

    def is_mobile_node(self, node):
        return node in self.md_graph

    def _is_gpu_node(self, node_id):
        id_str = str(node_id)
        return id_str.endswith(".1")


# =========================================================================
# TESTE BÁSICO (Para validação rápida)
# =========================================================================
if __name__ == '__main__':
    # Simples mock para garantir que o script roda sem erros de sintaxe
    # e que a classe NetworkMetrics está funcional.
    try:
        substrate_network = Net2()
        print("Net2 inicializado com sucesso.")
        print(f"Metrics inicializadas: {substrate_network.metrics}")
    except Exception as e:
        print(f"Erro na inicialização: {e}")