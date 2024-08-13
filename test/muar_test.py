from asyncio import base_subprocess
from curses import beep
from turtle import begin_poly
from controllers.substrate_network_controller import SubstrateNetworkController
from topology.simple_substrate_network import simple_six_node_topology
from algorithms.random_algorithm import RandomAlgorithm
from algorithms.greedy_algorithm import GreedyAlgorithm
from algorithms.dynamic_programming_algorithm import DynamicProgrammingAlgorithm
from algorithms.k_shortest_paths_algorithm import KShortestPathsAlgorithm
from algorithms.betweenness_centrality_algorithm import BetweennessCentralityAlgorithm
from algorithms.musfico import Musfico  
from algorithms.dp_test import DynamicProgrammingAlgorithm_test
import argparse
from test.paloalto_test import generate_substrate_network
import re
import os
import random
import numpy as np
from datetime import datetime as dt
from core.poisson_emitter import PoissonEmitter
from controllers.sfc_queue import SFCQueue
from controllers.sfc_generator import SFCGenerator
from controllers.sfc_controller import SFCController

# command line arguments
parser = argparse.ArgumentParser(description='Select MUAR arguments')
parser.add_argument('--n_sessions', type=int, help='(int) number of sessions', default=50)
parser.add_argument('--alg',   type=str, help='(str) algorithm name', default='dp')
parser.add_argument('--n_players', type=int, help='(int) number of players', default=4)
parser.add_argument('--sfc',   type=str, help='(str) on or off', default='on')
parser.add_argument('--prob',  type=str, help='(str) probability', default=0)
args = parser.parse_args()

alg_name = args.alg
if alg_name == 'musfico':
    SELECTED_ALG = Musfico()
if alg_name == 'dp':
    SELECTED_ALG = DynamicProgrammingAlgorithm()
if alg_name == 'g':
    SELECTED_ALG = GreedyAlgorithm()  
if alg_name == 'k':
    SELECTED_ALG = KShortestPathsAlgorithm(5)
if alg_name == 'b':
    SELECTED_ALG = BetweennessCentralityAlgorithm()
if alg_name == 'dp_new':
    SELECTED_ALG = DynamicProgrammingAlgorithm_test()

substrate_network = generate_substrate_network()
AVERAGE_TIME_SESSION_ARRIVAL = 10
sfc_poisson_emitter = PoissonEmitter(AVERAGE_TIME_SESSION_ARRIVAL)
n_sessions = args.n_sessions
player_counter = 0
sfc_queue = SFCQueue()
number_of_nodes = 36
SRC_NODE = 0
max_duration = 120
latency = 6
fator = 0.25 # 1 players consumes fator*100 percentage of resources of an Edge Server

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

#n_players = np.random.choice([4, 8, 12, 16])
n_players = 4

session_counter = 0
def generate_sfc_session(parameter):
    global n_players
    global session_counter
    session_counter = session_counter + 1
    counter = str(session_counter)
    print("Total Number of MUAR SFCs in session: ", counter)
    dst_node = random.randint(0, number_of_nodes -1 )
    while(dst_node == SRC_NODE):
        dst_node = random.randint(0, number_of_nodes - 1)
    players_cache_sf_list = []
    players_unique_sf_list = []
    for i in range(1,n_players+1):
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
    lifetime = np.random.poisson(max_duration)
    #lifetime = int(round(np.random.exponential(max_duration)))
    duration = lifetime
    players_sfc_cache_dict_list = []
    players_sfc_unique_dict_list = []
    
    for i in range(1,n_players+1):
        player_cache_dict = {}
        player_cache_dict['name'] = 'sfc_cache_p' + str(i) + '_' + counter
        player_cache_dict["vnf_list"] = players_cache_sf_list[i-1]
        player_cache_dict["bandwidth"] = EC_TC_bw
        player_cache_dict["src_node"] = SRC_NODE
        player_cache_dict["dst_node"] = dst_node
        player_cache_dict["duration"] = duration
        player_cache_dict["latency"] = latency
        players_sfc_cache_dict_list.append(player_cache_dict)
        player_unique_dict = {}
        player_unique_dict['name'] = 'sfc_unique_p' + str(i) + '_' + counter
        player_unique_dict["vnf_list"] = players_unique_sf_list[i-1]
        player_unique_dict["bandwidth"] = EC_TC_bw
        player_unique_dict["src_node"] = SRC_NODE
        player_unique_dict["dst_node"] = dst_node
        player_unique_dict["duration"] = duration
        player_unique_dict["latency"] = latency
        players_sfc_unique_dict_list.append(player_unique_dict)
    
    players_sfc_list = []
    for i in range(1,n_players+1):
        players_sfc_list.append([SFCGenerator(players_sfc_cache_dict_list[i-1]).generate(), SFCGenerator(players_sfc_unique_dict_list[i-1]).generate()])
        sfc_queue.put_sfc(players_sfc_list[i-1])
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
        player_mono_dict["latency"] = latency
        players_mono_dict_list.append(player_mono_dict)
    players_sfc_list = []
    for i in range(1,n_players+1):
        players_sfc_list.append([SFCGenerator(players_mono_dict_list[i-1]).generate()])
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

processing_nodes = np.arange(1, 37)

nodes_to_string = np.array2string(processing_nodes, suppress_small=True,
            precision=3, separator=',')

nodes_to_string = re.sub(' ', '', nodes_to_string)

nodes_to_string = re.sub('\n', '', nodes_to_string)

timestamp = dt.now().strftime('%Y%m%d%H%M%S%f')

cache_utilization_dir = os.path.dirname(f'./results_cache_utilization/sfc_{args.sfc}_alg_{args.alg}_prob_{round(float(args.prob), 2)}_{args.n_sessions}')
cpu_utilization_dir = os.path.dirname(f'./results_cpu_utilization/sfc_{args.sfc}_alg_{args.alg}_prob_{round(float(args.prob), 2)}_{args.n_sessions}')

path_to_save_cache = os.path.join(cache_utilization_dir,timestamp + '.csv')
path_to_save_cpu = os.path.join(cpu_utilization_dir,timestamp + '.csv')

if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('results_cpu_utilization'):
    os.mkdir('results_cpu_utilization')
if not os.path.exists('results_cache_utilization'):
    os.mkdir('results_cache_utilization')

with open(path_to_save_cache, "a") as file:
    file.write(nodes_to_string[1:-1] + '\n')
with open(path_to_save_cpu, "a") as file:
    file.write(nodes_to_string[1:-1] + '\n')

sbn_controller = SubstrateNetworkController(substrate_network)
sbn_controller.number_of_nodes = number_of_nodes
sbn_controller.sfc_queue = sfc_queue
sbn_controller.sfc = args.sfc
sbn_controller.alg = SELECTED_ALG
sbn_controller.alg_name = args.alg
sbn_controller.prob = args.prob
sbn_controller.processing_nodes = processing_nodes
sbn_controller.cpu_utilization_file = path_to_save_cpu
sbn_controller.cache_utilization_file = path_to_save_cache
sbn_controller.file_name = f'./results_flows_prob_{round(float(args.prob), 2)}_' + str(n_sessions) + '/' + alg_name + '_' + str(number_of_nodes) + '_sfc_'+ args.sfc + '_prob_' + str(round(float(args.prob),2)) + '/' + timestamp + '.csv'
sbn_controller.flows = n_sessions
directory = os.path.dirname(f'./results_flows_prob_{round(float(args.prob),2)}_' + str(n_sessions) + '/' + alg_name + '_' + str(number_of_nodes) + '_sfc_'+ args.sfc + '_prob_' + str(round(float(args.prob),2)) + '/' )
if not os.path.exists(directory):
    os.makedirs(directory)
#if not os.path.exists(os.path.dirname(f'./results_flows_prob_{round(float(args.prob),2)}_' + str(n_sessions))):
#    os.makedirs(directory)
#sbn_controller.prob_array = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

#for prob in sbn_controller.prob_array:
#    sbn_controller.file_name_list.append(f'./results_flows_prob_{str(prob)}_' + str(n_sessions) + '/' + alg_name + '_' + str(number_of_nodes) + '_sfc_'+ args.sfc + '_prob_' + str(prob) + '/' + dt.now().strftime('%Y%m%d%H%M%S%f') + '.csv')
#    directory = os.path.dirname(f'./results_flows_prob_{str(prob)}_' + str(n_sessions) + '/' + alg_name + '_' + str(number_of_nodes) + '_sfc_'+ args.sfc + '_prob_' + str(prob) + '/' )
#    if not os.path.exists(directory):
#        os.makedirs(directory)

with open(sbn_controller.file_name, "a") as f:
            f.write("No.,timestamp,number_of_sfc,cpu_utilization,bandwidth_utilization,cache_utilization,latency,duration,success,arrival_time,depart_time,sfc_id" + "\n")
sbn_controller.start()
