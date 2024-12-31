import os
import numpy as np
import re
from datetime import datetime
import random
import time
from typing import Dict

def format_nodes_to_string(nodes):
    nodes_to_string = np.array2string(nodes, suppress_small=True, precision=3, separator=',')
    return re.sub('[ \n]', '', nodes_to_string)

def format_edges_to_string(edges):
    edges_to_string = ';'.join(map(str, edges))
    return re.sub('[ \n]', '', edges_to_string)

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_output_dir(args,topology):
    ec_servers = topology.get_topology_info()['ec_servers']
    edges = topology.get_topology_info()['edges']

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f') + str(random.randint(0, 10000))
    base_dir = 'results/'
    #paths = ['cache', 'cpu', 'bandwidth', 'edges_vnf', 'sf']
    paths = ['cache', 'cpu', 'bandwidth', 'sf']

    directories = {}

    for path in paths:
        dir_path = os.path.join(base_dir, f'results_{path}')
        create_directory_if_not_exists(dir_path)
        alg_path = os.path.join(dir_path, f'alg_{args.alg}_s_{args.n_sessions}_p_{args.n_sessions}_sfc_{args.sfc}')
        create_directory_if_not_exists(alg_path)
        directories[path] = alg_path

    # Prepare file paths
    file_paths = {path: os.path.join(directories[path], timestamp + '.csv') for path in paths}
    
    nodes_string = format_nodes_to_string(np.array(sorted(ec_servers)))
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
    
    dir = 'results/results_flows'
    res_dir = 'results/results_resilient'
    directory_path = os.path.join(dir, f'{args.alg}_s_{args.n_sessions}_p_{args.n_sessions}_sfc_{args.sfc}')
    res_directory_path = os.path.join(res_dir, f'{args.alg}_s_{args.n_sessions}_p_{args.n_sessions}_sfc_{args.sfc}')
    create_directory_if_not_exists(directory_path)
    create_directory_if_not_exists(res_directory_path)

    flows_path = os.path.join(directory_path, f'{timestamp}.csv')
    res_path = os.path.join(res_directory_path, f'{timestamp}.csv')

    # Define o header como uma lista para facilitar alterações
    header_fields = [
        "No.",
        "timestamp",
        "time_seconds",
        "users",
        "cpu_utilization",
        "bandwidth_utilization",
        "cache_utilization",
        "cpu_resilient",
        "cache_resilient",
        "bw_resilient",
        "latency",
        "latency_diff",
        "wait_time",
        "decision_time_ms",
        "success",
        "backup_success",
        "arrival_time",
        "sfc_id",
        "recovery_time",
        "sfc_recovered",
        "cpu_saved",
        "cache_saved",
        "number_of_sfc",
        "shared_vnfs",
        "running_sfcs",
        "running_players",
        "running_sessions",
        "trascode_bw",
    ]

    res_fields = ["sfc_id","recover_success","backup_success","latency_diff","time_to_recover"]
            
    header = ",".join(header_fields) + "\n"
    res_header = ",".join(res_fields) + "\n"

    with open(flows_path, "a") as f:
        f.write(header)

    with open(res_path, "a") as f:
        f.write(res_header)
    return timestamp,file_paths,flows_path,res_path

class OutputWritter:
    def __init__(self, nodes,processing_nodes, edges, cpu_utilization_file, cache_utilization_file, bw_utilization_file, sf_utilization_file,flows_file,res_file):
        self.nodes = nodes
        self.edges = edges
        self.processing_nodes = processing_nodes
        self.cpu_utilization_file = cpu_utilization_file
        self.cache_utilization_file = cache_utilization_file
        self.bw_utilization_file = bw_utilization_file
        self.sf_utilization_file = sf_utilization_file
        self.flows_file = flows_file
        self.resilient_file = res_file
        self.first_time = 0
        self.sfcs_latency_dict = {}
        self.counter_users = 0

    def resilient_output(self,sfc_id,info):
        is_success = info["recover_success"]
        backup_success = info["backup_success"]
        latency_diff = info["latency_diff"]
        time_to_recover = info["time_to_recover"]

        with open(self.resilient_file, "a") as file:
            line = str(sfc_id) + ',' + \
                str(is_success) + ',' + \
                str(backup_success) + ',' + \
                str(latency_diff) + ',' + \
                str(time_to_recover) + "\n"
            file.write(line)

    def output_flows(self,substrate_network,wait_time,running_players_sessions,counter,remaining_time,current_time, sfc, latency, run_duration, is_success,backup_sfc,bw_transcode, latency_diff=None):
        cpu_utilization = round(substrate_network.get_cpu_utilization_rate(), 4)
        cache_utilization = round(substrate_network.get_cache_utilization_rate(), 4)
        bw_utilization = round(substrate_network.get_bandwidth_utilization_rate(), 4)

        cpu_resilient = round(substrate_network.get_resilient_cpu_utilization(), 4)
        cache_resilient = round(substrate_network.get_resilient_cache_utilization(), 4)
        bw_resilient = round(substrate_network.get_resilient_bandwidth_utilization(), 4)

        # active_servers_cpu = round(substrate_network.get_active_servers_cpu_rate(), 4)
        # active_servers_cache = round(substrate_network.get_active_servers_cache_rate(), 4)
        # active_links_bw = round(substrate_network.get_active_links_bw_rate(), 4)

        running_sfcs, running_players, running_sessions = running_players_sessions

        cpu_saved = substrate_network.cpu_saved
        cache_saved = substrate_network.cache_saved
        shared_vnfs_count = substrate_network.shared_vnfs_count
        
        sfc_recovery_time = 0
        sfc_recovered = None
        
        # if sfc.id in sfcs_crashed:
        #     sfc_recovered =  0   
        #     sfc_recovery_time = None
        crashed_sfcs = []

        sfc_id = sfc.id
        self.update_user_count(sfc_id)
        # if sfc.id in sfcs_crashed and is_success == 1:
        #     sfc_recovery_time = time.time() - sfcs_crashed[sfc.id]
        #     sfc_recovered = 1

        first_loop = (self.first_time == 0)
        time_value = 0  

        if first_loop: 
            self.first_time = current_time
            time_value = 0
        else:        
            time_value = round(current_time - self.first_time,1)

        backup_success = None
        if backup_sfc:
            backup_success = is_success
            is_success = None
        
        with open(self.flows_file, "a") as file:
            line = str(counter) + ',' + \
                   str(current_time) + ',' + \
                   str(time_value) + ',' + \
                   str(self.counter_users) + ',' + \
                   str(cpu_utilization) + ',' + \
                   str(bw_utilization) + ',' + \
                   str(cache_utilization) + ',' + \
                   str(cpu_resilient) + ',' + \
                   str(cache_resilient) + ',' + \
                   str(bw_resilient) + ',' + \
                   str(latency) + ',' + \
                   str(latency_diff) + ',' + \
                   str(wait_time) + ',' + \
                   str(round(run_duration * 1000, 3)) + ',' + \
                   str(is_success) + ',' + \
                   str(backup_success) + ',' + \
                   str(sfc.arrival_time) + ',' + \
                   str(sfc.id) + "," + \
                   str(sfc_recovery_time) + "," + \
                   str(sfc_recovered) + "," + \
                   str(cpu_saved) + "," + \
                   str(cache_saved) + "," + \
                   str(sfc.number_of_vnfs) + ',' + \
                   str(shared_vnfs_count) + "," + \
                   str(running_sfcs) + "," + \
                   str(running_players) + "," + \
                   str(running_sessions) + "," + \
                   str(bw_transcode) + "\n"
            file.write(line)

    def update_user_count(self, sfc_id):
        # Extrair o número do player e da sessão
        player = int(sfc_id.split("_")[-2][1])
        session = int(sfc_id.split("_")[-1])
        
        # Verificar se a sessão atual é um backup (sessões pares são backups)
        users = self.counter_users

        users = (session - 1) * 5 + player
        
        if users > self.counter_users:
            self.counter_users = users


    def output_cpu_utilization(self, substrate_network, deploy_time, crashed_nodes=[]) -> None:
        """
        Outputs the CPU utilization of nodes to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
            crashed_nodes (list): List of nodes that are crashed and should not report CPU usage.
        """
        processing_nodes = sorted(self.processing_nodes)
        cpu_nodes_util = [
            None if node in crashed_nodes else round(substrate_network.get_node_cpu_used(node), 2)
            for node in processing_nodes
        ]

        # Format the array to a string
        string_cpu_nodes_util = ','.join(['None' if value is None else f"{value:.2f}" for value in cpu_nodes_util])

        # Write the result to the file
        with open(self.cpu_utilization_file, "a") as file:
            file.write(f"{deploy_time},{string_cpu_nodes_util}\n")


    def output_cache_utilization(self, substrate_network, deploy_time, crashed_nodes=[]) -> None:
        """
        Outputs the cache utilization of nodes to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
            crashed_nodes (list): List of nodes that are crashed and should not report cache usage.
        """
        processing_nodes = sorted(self.processing_nodes)

        cache_nodes_util = [
            None if node in crashed_nodes else round(substrate_network.get_node_cache_used(node), 2)
            for node in processing_nodes
        ]

        # Format the array to a string
        string_cache_nodes_util = ','.join(['None' if value is None else f"{value:.2f}" for value in cache_nodes_util])

        # Write the result to the file
        with open(self.cache_utilization_file, "a") as file:
            file.write(f"{deploy_time},{string_cache_nodes_util}\n")


    def output_bandwidth_utilization(self, substrate_network, deploy_time: float) -> None:
        """
        Outputs the bandwidth utilization of network edges to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
        """
        nodes_one = [node_one for node_one, node_two in self.edges]
        nodes_two = [node_two for node_one, node_two in self.edges]

        bw_edges_util = np.array(list(map(substrate_network.get_link_bandwidth_used, nodes_one, nodes_two)))

        string_bw_edges_util = np.array2string(bw_edges_util, suppress_small=True,
                                               precision=3, separator=';', 
                                               formatter={'float_kind': lambda x: "%.2f" % x})

        string_bw_edges_util = re.sub(' ', '', string_bw_edges_util)
        string_bw_edges_util = re.sub('\n', '', string_bw_edges_util)

        with open(self.bw_utilization_file, "a") as file:
            file.write(str(deploy_time) + ';' + string_bw_edges_util[1:-1] + '\n')

    def output_nodes_sf_utilization(self, substrate_network, deploy_time: float) -> None:
        """
        Outputs the service function (SF) utilization of nodes to a specified file.

        This function calculates the number of service functions running on each node in the network.
        The results are formatted as a string and appended to the SF utilization file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
            substrate_network (object): The substrate network object containing node information.

        Returns:
            None
        """
        sf_nodes_util = np.array([len(substrate_network.get_node_sfc_vnf_list(node)) for node in self.nodes])

        string_sf_nodes_util = np.array2string(sf_nodes_util, suppress_small=True,
                                               precision=3, separator=',', 
                                               formatter={'float_kind': lambda x: "%.2f" % x})

        string_sf_nodes_util = re.sub(' ', '', string_sf_nodes_util)
        string_sf_nodes_util = re.sub('\n', '', string_sf_nodes_util)

        with open(self.sf_utilization_file, "a") as file:
            file.write(str(deploy_time) + ',' + string_sf_nodes_util[1:-1] + '\n')

    def output_nodes_information(self, substrate_network,*args) -> None:
        substrate_network.print_out_nodes_information(args[0], args[1])

    def output_edges_information(self,substrate_network,*args) -> None:
        substrate_network.print_out_edges_information(args[0])
    
    def output_acceptance_information(self,substrate_network,success) -> None:
        substrate_network.print_out_acceptance_information(success)

    def print_output_info(self, substrate_network,success) -> None:
        """
        Outputs information about nodes and edges' 
        resource utilization. 
        """
        self.output_nodes_information(substrate_network,None, None)
        self.output_edges_information(substrate_network,None)
        self.output_acceptance_information(substrate_network,success)