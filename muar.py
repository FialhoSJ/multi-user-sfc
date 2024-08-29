import argparse
import math
import re
import os
import random
import numpy as np
import ast

from controllers.substrate_network_controller import SubstrateNetworkController
from controllers.crasher import Crasher
from datetime import datetime as dt
from controllers.substrate_network_controller_resilience import ResilientSubstrateNetworkController
from core.poisson_emitter import PoissonEmitter
from controllers.sfc_queue import SFCQueue
from controllers.sfc_generator import SFCGenerator

from algorithms.instantiator import AlgorithmInstantiator
from sumo.models.user_manager import UserManager
from sumo.tracer_instantiator import TracerInstantiator
from sumo.luxembourg.config_routes import topology_tracer_positions,positions
from topology.instantiator import TopologyInstantiator
from utils.directory_manager import setup_directories_and_files, setup_sbn_controller_directory_and_file
from utils.resource_output_utils import OutputWritter
seed = 42
random.seed(seed)
np.random.seed(seed)

# command line arguments
parser = argparse.ArgumentParser(description='Select MUAR arguments') 
parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
parser.add_argument('--alg',   type=str, help='(str) algorithm name', default='goku')
parser.add_argument('--n_players', type=int, help='(int) number of players', default=4)
parser.add_argument('--sfc',   type=str, help='(str) on or off', default='on')
parser.add_argument('--topology', type=str, help='(str) wich topology ex: luxembourg,small luxembourg ,paloalto', default='luxembourg')

#on: quebrar mais em funçoes
#off: monolítico
parser.add_argument('--share',  type=str, help='(str) whether to share sfs or not', default='y')
parser.add_argument('--time',  type=int, help='(int) the total time for the simulation in seconds', default=120)
parser.add_argument('--mobility',  type=str, help='(str) mobility', default='y')

parser.add_argument('--shareband',  type=str, help='(str) whether to share sfs or not', default='y')
parser.add_argument('--allow_delay', type=str, help='(str) whether to allow delay or not', default='y')

parser.add_argument('--allow_crasher', type=str, help='(str) whether to allow delay or not', default='y')

parser.add_argument('--reliability', type=str, help='(str) whether to allow delay or not', default=0.95) #0.95,0.975,0.99
parser.add_argument('--servers_to_crash', type=str, help='(str) whether to allow delay or not', default=1) #0.95,0.975,0.99

parser.add_argument('--costs_parameter',   type=str, help='cpu,cache,bandwidht,boot Ex: 1111', default='[1,1,1,1]')
parser.add_argument('--verbose',   type=str, help='verbose log', default='n')

#Coleta dos parâmetros da simulação
args = parser.parse_args()

alg_name = args.alg
top_name = args.topology
n_sessions = int(args.n_sessions) * 2 if alg_name == 'goku_backup' else int(args.n_sessions)
n_players = int(args.n_players)
mobility_activated = True if args.mobility == 'y' else False 
verbose = True if args.verbose == 'y' else False 
crasher_activated = 1 if args.allow_crasher == 'y' else 0
allow_delay =  True if args.share == 'y' else False
shareable = (args.share == 'y')
shareable_band = (args.shareband == 'y')
share_str = "sharing_y" if shareable else "sharing_n"
costs_parameters = ast.literal_eval(args.costs_parameter)
servers_to_crash = int(args.servers_to_crash)
reliability = float(args.reliability)
alg_instantiator = AlgorithmInstantiator()
topology_instantiator = TopologyInstantiator()
tracer_instantiator = TracerInstantiator()
user_manager = UserManager(int(n_sessions), n_players)

SELECTED_ALG = alg_instantiator.instantiate_algorithm(alg_name)
topology = topology_instantiator.instantiate_topology(top_name)

substrate_network = topology.generate_substrate_network()
number_of_nodes = substrate_network.number_of_nodes()
edges = substrate_network.edges
quantity_of_nodes_arranged = np.arange(1, number_of_nodes)
edges_vnf = {key: [] for key in edges}

# Crasher instantiator, must dismiss node close to the cloud(34)
processing_nodes = list(topology.processing_nodes)
processing_nodes.remove(34)

servers_that_will_crash =  random.sample(processing_nodes, servers_to_crash)
crasher_instance = Crasher(servers_to_crash=[28],operation_mode = 4, processing_nodes=processing_nodes)

AVERAGE_TIME_SESSION_ARRIVAL = 20
sfc_poisson_emitter = PoissonEmitter(AVERAGE_TIME_SESSION_ARRIVAL)

player_counter = 0
#sfc_queue = []
sfc_queue = SFCQueue()

SRC_NODE = 0
max_duration = 120

latency_interval = [6,10]
sfcs_latency = latency_interval[1] if args.allow_delay == 'y' else latency_interval[0]

fator = 0.4 # 1 players consumes fator*100 percentage of resources of an Edge Server
#fator = 0.1 # 1 players consumes fator*100 percentage of resources of an Edge Server
edges = substrate_network.edges
#https://ieeexplore.ieee.org/document/9417376
#10 cycles per Mbit
cpb = 10e6
#print(cpb, "cycles per Mbit")
IA_bw = 150# * n_players
IA = IA_bw * cpb
IA = 0
# 400x400 pixels RGB with about 0.48 MB per frame
DET_bw = 0.230 # 480 KB * 8 * 60 fps
DET = DET_bw * cpb
# 4 to 12 feature representations of VO, each have 25KB [100,300] KB per frame
FT_bw = 0.144 # 300 KB * 8 * 60  [48,144] 
FT = FT_bw * cpb
IA_DET_FT_bw = int(IA_bw + DET_bw + FT_bw)
# each AR VO have 2500 KB in average, 2500 * 12
CA_size = 240 # 2500 KB * 8 * 12 objects 240 for 12 VOs

chr = 1/3 # cache hit ratio
MA_bw = int(CA_size * chr)
MA = MA_bw * cpb
# not in cache
UNI_bw = int(CA_size * (1-chr))
UNI = UNI_bw * cpb
# matched and non-matched objects
RE_bw = MA_bw + UNI_bw
RE = int(RE_bw * cpb)
#https://ieeexplore.ieee.org/document/9316983
# final out put 
EC_TC_bw = int(IA_bw*0.9*0.9*0.8) 
EC_TC = EC_TC_bw * cpb 

total = IA+DET+FT+MA+UNI+RE+EC_TC
IA = int(IA/total*fator*100)
DET = DET/total*fator*100
FT = FT/total*fator*100
IA_DET_FT = int(IA + DET + FT)
MA = int(MA/total*fator*100)
UNI = int(UNI/total*fator*100)
RE = int(RE/total*fator*100)
EC_TC = int(EC_TC/total*fator*100)
MONO = int(DET+FT+MA+UNI+RE+EC_TC)
CA_size = CA_size/CA_size*fator*100

def previous_sfc_setup (backup=False):
    # Função para calcular a distância entre dois nós
    def calculate_distance(node1, node2, topology_tracer_positions):
        x1, y1 = topology_tracer_positions[node1]
        x2, y2 = topology_tracer_positions[node2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def find_valid_route(node, topology_tracer_positions, routes, visited_nodes):
        possible_destinations = {n: calculate_distance(node, n, topology_tracer_positions) 
                                for n in topology_tracer_positions if n != node and n not in visited_nodes}
        
        # Se não houver nós disponíveis, retorna None
        if len(possible_destinations) < 2:
            return None

        # Ordenar os destinos possíveis pela distância
        sorted_destinations = sorted(possible_destinations, key=possible_destinations.get)
        
        # Selecionar o segundo nó mais próximo
        destination = sorted_destinations[2] if len(sorted_destinations) > 2 else sorted_destinations[0]
        
        return destination


    # Definindo o número de nós, sessões e jogadores
    number_of_nodes = len(topology_tracer_positions)
    number_of_sessions = n_sessions
    number_of_players = n_players

    # Inicializando o dicionário para armazenar os destinos e rotas das SFCs
    # Inicializando o dicionário para armazenar os destinos e rotas das SFCs
    sfc_destinations_and_routes = {}

    # Iterando sobre sessões e jogadores para definir destinos e rotas
    for session in range(1, number_of_sessions + 1):
        # Definindo o nó de origem base para a sessão
        session_dst = random.randint(1, number_of_nodes)
        current_node = session_dst
        routes = {}
        for i in range(n_players):
            current_node = session_dst
            visited_nodes = {current_node}  # Conjunto de nós visitados
            # Inicializando a lista de rotas para o jogador `i+1`
            routes[i + 1] = []
            for _ in range(10):  
                next_node = find_valid_route(current_node, topology_tracer_positions, routes[i + 1], visited_nodes)
                if next_node is None:
                    break  # Se não há mais destinos válidos, interrompe o loop
                # Adicionando a tupla (current_node, next_node) na lista correspondente ao jogador
                routes[i + 1].append((current_node, next_node))
                visited_nodes.add(next_node)  # Adiciona o nó visitado ao conjunto
                current_node = next_node  # Atualiza o nó atual para o próximo

            key_session = f'{session}'

        locations = {i: session_dst for i in range(1, n_players+1)}

        if session % 2 == 1:  # Sessões ímpares (SFCs normais)
            # Definindo o destino da rota para a SFC normal
            # route_dst = find_valid_route(session_dst, topology_tracer_positions)    
            sfc_destinations_and_routes[session] = {
                "current_dst": n_players*[session_dst],
                "route": routes,
                "duration": np.random.poisson(max_duration),
                "locations": locations}
            
        else:  # Sessões pares (SFCs de backup)
            if backup:
                last_normal_dst = [x[1][0][1] for x in sfc_destinations_and_routes[session-1]['route'].items()]
                routes = sfc_destinations_and_routes[session-1]['route']
                # Novo dicionário a ser criado
                new_routes = {}

                for key, value in routes.items():
                    new_routes[key] = {}
                    for i in range(len(value)):
                        # Se não for o último elemento
                        if i < len(value) - 1:
                            new_routes[key][value[i][1]] = value[i+1][1]
                        else:
                            # Para o último elemento, ele se conecta a ele mesmo
                            new_routes[key][value[i][1]] = value[i][1]

                sfc_destinations_and_routes[session] = {
                    "current_dst": last_normal_dst,
                    "route": new_routes,
                    "duration": sfc_destinations_and_routes[session-1]['duration'],
                    "locations": n_players*[last_normal_dst]}
            else:
                sfc_destinations_and_routes[session] = {
                "current_dst": n_players*[session_dst],
                "route": routes,
                "duration": np.random.poisson(max_duration),
                "locations": locations}

    return sfc_destinations_and_routes

if alg_name == "goku_backup":
    sfc_destinations_and_routes = previous_sfc_setup(backup=True)
else: 
    sfc_destinations_and_routes = previous_sfc_setup()
tracer = tracer_instantiator.instantiate_tracer(top_name,user_manager,initial_routes=sfc_destinations_and_routes) if mobility_activated else 0

session_counter = 0
def generate_sfc_session(parameter) -> None:
    global n_players
    global session_counter
    session_counter = session_counter + 1
    counter = str(session_counter)
    print("Total Number of MUAR SFCs in session: ", counter)
    #dst_node = random.randint(0, number_of_nodes -1 )
    #dst_node = sfc_destinations_and_routes[session_counter]['current_dst'][0]
    # while(dst_node == SRC_NODE):
    #     dst_node = random.randint(0, number_of_nodes - 1)
        
    players_cache_sf_list = []
    players_unique_sf_list = []
    for i in range(1,n_players+1):
        dst_node = sfc_destinations_and_routes[session_counter]['current_dst'][i-1]
        caching_sf_list = []
        caching_sf_list.append({"type": 2, "name":"IA_DET_FT_" + counter, 
            "CPU": IA_DET_FT, "cache": 0, "in_bw": IA_bw, "out_bw": IA_DET_FT_bw})
        caching_sf_list.append({"type": 2, "name":"MA_region_" + str(dst_node),# + "_" + counter, 
            "CPU": MA, "cache": CA_size, "in_bw": IA_DET_FT_bw, "out_bw": MA_bw})
        caching_sf_list.append({"type": 2, "name":"RE_region_" + str(dst_node),# + "_" + counter, 
            "CPU": RE*chr, "cache": 0, "in_bw": MA_bw, "out_bw": RE_bw*chr})
        caching_sf_list.append({"type": 2, "name":"EC_TC_p" + str(i) + "_" + counter, 
            "CPU": EC_TC*chr, "cache": 0, "in_bw": RE_bw*chr, "out_bw": EC_TC_bw*chr})
        players_cache_sf_list.append(caching_sf_list)
        unique_sf_list = []
        unique_sf_list.append({"type": 2, "name":"IA_DET_FT_" + counter, 
            "CPU": IA_DET_FT, "cache": 0, "in_bw": IA_bw, "out_bw": IA_DET_FT_bw})
        unique_sf_list.append({"type": 2, "name":"UNI_p" + str(i) + "_" + counter, 
            "CPU": UNI, "cache": 0, "in_bw": IA_DET_FT_bw, "out_bw": UNI_bw})
        unique_sf_list.append({"type": 2, "name":"RE_p" + str(i) + "_"    + counter, 
            "CPU": RE*(1-chr), "cache": 0, "in_bw": UNI_bw, "out_bw": RE_bw*(1-chr)})
        unique_sf_list.append({"type": 2, "name":"EC_TC_p" + str(i) + "_" + counter, 
            "CPU": EC_TC*(1-chr), "cache": 0, "in_bw": RE_bw*(1-chr), "out_bw": EC_TC_bw*(1-chr)})
        players_unique_sf_list.append(unique_sf_list)
    #lifetime = np.random.poisson(max_duration)
    #lifetime = int(round(np.random.exponential(max_duration)))
    #duration = lifetime
    duration = sfc_destinations_and_routes[session_counter]['duration']
    players_sfc_cache_dict_list = []
    players_sfc_unique_dict_list = []
    
    for i in range(1,n_players+1):
        dst_node = sfc_destinations_and_routes[session_counter]['current_dst'][i-1]
        player_cache_dict = {}
        player_cache_dict['name'] = 'sfc_cache_p' + str(i) + '_' + counter
        player_cache_dict["vnf_list"] = players_cache_sf_list[i-1]
        player_cache_dict["bandwidth"] = EC_TC_bw
        player_cache_dict["src_node"] = SRC_NODE
        player_cache_dict["dst_node"] = dst_node
        player_cache_dict["duration"] = duration
        player_cache_dict["latency"] = sfcs_latency
        player_cache_dict["time"] = sfcs_latency
        players_sfc_cache_dict_list.append(player_cache_dict)
        player_unique_dict = {}
        player_unique_dict['name'] = 'sfc_unique_p' + str(i) + '_' + counter
        player_unique_dict["vnf_list"] = players_unique_sf_list[i-1]
        player_unique_dict["bandwidth"] = EC_TC_bw
        player_unique_dict["src_node"] = SRC_NODE
        player_unique_dict["dst_node"] = dst_node
        player_unique_dict["duration"] = duration
        player_unique_dict["latency"] = sfcs_latency
        players_sfc_unique_dict_list.append(player_unique_dict)
    
    players_sfc_list = []
    for i in range(1,n_players+1):
        players_sfc_list.append([SFCGenerator(players_sfc_cache_dict_list[i-1]).generate(),
                                  SFCGenerator(players_sfc_unique_dict_list[i-1]).generate()])
        
        sfc_queue.put_sfc(players_sfc_list[i-1])
        #heapq.heappush(sfc_queue, (1, counter, i, players_sfc_list[i-1]))
    if session_counter >= n_sessions:
        #print("SFC MUAR Session ## poisson stop  ##")
        sfc_poisson_emitter.stop()

def generate_mono_session(parameter):
    global n_players
    global session_counter
    session_counter = session_counter + 1
    counter = str(session_counter)
    print("Total Number of Monolithic MUAR in session: ", counter)
    dst_node = random.randint(0, number_of_nodes -1 )
    while(dst_node == SRC_NODE):
        dst_node = random.randint(0, number_of_nodes - 1)
    players_mono_sf_list = []
    for i in range(1,n_players+1):
        mono_sf_list = []
        mono_sf_list.append({"type": 2, "name":"IA_" + counter, 
            "CPU": IA, "cache": 0, "in_bw": IA_bw, "out_bw": IA_bw})
        mono_sf_list.append({"type": 2, "name":"MONO_p" + str(i) + "_"    + counter, 
            "CPU": MONO, "cache": CA_size, "in_bw": IA_bw, "out_bw": EC_TC_bw})
        players_mono_sf_list.append(mono_sf_list)
    lifetime = np.random.poisson(max_duration)
    #lifetime = int(round(np.random.exponential(max_duration)))
    duration = lifetime
    players_mono_dict_list = []
    for i in range(1,n_players+1):
        player_mono_dict = {}
        player_mono_dict['name'] = 'sfc_mono_p' + str(i) + '_' + counter
        player_mono_dict["vnf_list"] = players_mono_sf_list[i-1]
        player_mono_dict["bandwidth"] = EC_TC_bw
        player_mono_dict["src_node"] = SRC_NODE
        player_mono_dict["dst_node"] = dst_node
        player_mono_dict["duration"] = duration
        player_mono_dict["latency"] = sfcs_latency
        players_mono_dict_list.append(player_mono_dict)
    players_sfc_list = []
    for i in range(1,n_players+1):
        players_sfc_list.append([SFCGenerator(players_mono_dict_list[i-1]).generate()])
        #heapq.heappush(sfc_queue, (1, players_sfc_list[i-1]))
        sfc_queue.put_sfc(players_sfc_list[i-1])
    if session_counter >= n_sessions:
        print("MONO MUAR Session ## poisson stop  ##")
        sfc_poisson_emitter.stop()

if args.sfc == 'off':
    print('sfc off')
    sfc_poisson_emitter.start(generate_mono_session, (None))
if args.sfc == 'on':
    print('sfc on')
    sfc_poisson_emitter.start(generate_sfc_session, (None))

substrate_network.set_verbose(verbose=verbose)


resilient_algs = ['goku_backup']
backup_activated = True if alg_name in resilient_algs else False
if backup_activated:
    n_sessions_folder = int(int(n_sessions)/2)
else:
    n_sessions_folder = n_sessions
timestamp,file_paths = setup_directories_and_files(n_sessions_folder,n_players, args, quantity_of_nodes_arranged, edges)

substrate_network.shareable_band = shareable_band
substrate_network.shareable_node = shareable

sbn_controller = ResilientSubstrateNetworkController(substrate_network) if alg_name in resilient_algs else SubstrateNetworkController(substrate_network)

sbn_controller.number_of_nodes = number_of_nodes
sbn_controller.sfc_queue = sfc_queue
sbn_controller.sfc = args.sfc
sbn_controller.shareable = shareable
sbn_controller.alg = SELECTED_ALG
sbn_controller.alg_name = args.alg
sbn_controller.user_manager = user_manager
sbn_controller.crasher = crasher_instance
sbn_controller.crasher_activate = crasher_activated
sbn_controller.servers_to_crash = servers_to_crash

sbn_controller.verbose = verbose

sbn_controller.nodes = quantity_of_nodes_arranged
sbn_controller.edges = edges
sbn_controller.edges_vnf = edges_vnf
sbn_controller.tracer = tracer
sbn_controller.mobility_activated = mobility_activated
sbn_controller.allow_temporary_high_latency =  allow_delay

sbn_controller.latency_interval = latency_interval

sbn_controller.output_writter = OutputWritter(quantity_of_nodes_arranged, edges, file_paths['cpu'], file_paths['cache'], file_paths['bandwidth'], file_paths['sf'],setup_sbn_controller_directory_and_file(n_sessions_folder,n_players,alg_name, number_of_nodes, args, timestamp),backup_activated=backup_activated)
sbn_controller.shareable_band = shareable_band
sbn_controller.costs_parameters = costs_parameters
sbn_controller.flows = n_sessions
sbn_controller.players = n_players
sbn_controller.start()
