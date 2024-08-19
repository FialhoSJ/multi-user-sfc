import os
import numpy as np
import re
from datetime import datetime
import random
import random2

def format_nodes_to_string(nodes):
    nodes_to_string = np.array2string(nodes, suppress_small=True, precision=3, separator=',')
    return re.sub('[ \n]', '', nodes_to_string)

def format_edges_to_string(edges):
    edges_to_string = ';'.join(map(str, edges))
    return re.sub('[ \n]', '', edges_to_string)

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_directories_and_files(n_sessions,n_players, args, quantity_of_nodes_arranged, edges):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f') + str(random.randint(0, 10000))
    base_dir = 'results/'
    #paths = ['cache', 'cpu', 'bandwidth', 'edges_vnf', 'sf']
    paths = ['cache', 'cpu', 'bandwidth', 'sf']

    directories = {}

    for path in paths:
        dir_path = os.path.join(base_dir, f'results_{path}')
        create_directory_if_not_exists(dir_path)
        alg_path = os.path.join(dir_path, f'alg_{args.alg}_s_{n_sessions}_p_{n_players}_rel_{args.servers_to_crash}')
        create_directory_if_not_exists(alg_path)
        directories[path] = alg_path

    # Prepare file paths
    file_paths = {path: os.path.join(directories[path], timestamp + '.csv') for path in paths}
    
    nodes_string = format_nodes_to_string(quantity_of_nodes_arranged)
    edges_string = format_edges_to_string(edges)
    
    # Initialize files
    with open(file_paths['cache'], "a") as file:
        file.write(f'timestamp,{nodes_string[1:-1]}\n')
    with open(file_paths['cpu'], "a") as file:
        file.write(f'timestamp,{nodes_string[1:-1]}\n')
    with open(file_paths['sf'], "a") as file:
        file.write(f'timestamp,{nodes_string[1:-1]}\n')
    with open(file_paths['bandwidth'], "a") as file:
        file.write(f'timestamp;{edges_string}\n')
    # with open(file_paths['edges_vnf'], "a") as file:
    #     file.write(f'timestamp;{edges_string}\n')
    
    return timestamp,file_paths

def setup_sbn_controller_directory_and_file(n_sessions,n_players, alg_name, number_of_nodes, args, timestamp):
    dir = 'results/results_flows'
    directory_path = os.path.join(dir, f'{alg_name}_s_{n_sessions}_p_{n_players}_sfc_{args.sfc}_rel_{args.servers_to_crash}')
    create_directory_if_not_exists(directory_path)

    file_name = os.path.join(directory_path, f'{timestamp}.csv')
    with open(file_name, "a") as f:
        header = "No.,time,timestamp,tempo_norm,number_of_sfc,cpu_utilization,cpu_active_servers,bandwidth_utilization,bw_active_links,cache_utilization,cache_active_servers,cpu_resilient,cache_resilient,bw_resilient,latency,latency_diff,duration,success,arrival_time,depart_time,sfc_id,crash_moment,recovery_time,sfc_recovered,cpu_saved,cache_saved,shared_vnfs,running_sfcs,running_players,running_sessions,server_crashed,trascode_bw,users_crashed\n"
        f.write(header)
    return file_name
