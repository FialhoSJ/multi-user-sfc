import numpy as np
import re
import time
from typing import Dict

class OutputWritter:
    def __init__(self, nodes, edges, cpu_utilization_file, cache_utilization_file, bw_utilization_file, sf_utilization_file,flows_file):
        self.nodes = nodes
        self.edges = edges
        self.cpu_utilization_file = cpu_utilization_file
        self.cache_utilization_file = cache_utilization_file
        self.bw_utilization_file = bw_utilization_file
        self.sf_utilization_file = sf_utilization_file
        self.flows_file = flows_file
        self.first_time = 0

    def output_flows(self,substrate_network,running_players_sessions,counter,remaining_time,current_time, sfc, latency, run_duration, is_success, s2, a_server_was_crashed,bw_transcode,users_crashed,sfcs_crashed,crash_moment):
        """
        Outputs the flow information including various network utilization metrics.

        Args:
            current_time (float): The current simulation time.
            sfc (object): The service function chain (SFC) object containing the SF details.
            latency (float): The latency of the current flow.
            run_duration (float): The duration of the flow run.
            is_success (bool): Whether the flow was successful or not.
            s2 (float): Additional parameter related to the flow.

        Returns:
            None
        """
        cpu_utilization = round(substrate_network.get_cpu_utilization_rate(), 4)
        cache_utilization = round(substrate_network.get_cache_utilization_rate(), 4)
        bw_utilization = round(substrate_network.get_bandwidth_utilization_rate(), 4)

        cpu_resilient = round(substrate_network.get_resilient_cpu_utilization(), 4)
        cache_resilient = round(substrate_network.get_resilient_cache_utilization(), 4)
        bw_resilient = round(substrate_network.get_resilient_bandwidth_utilization(), 4)

        active_servers_cpu = round(substrate_network.get_active_servers_cpu_rate(), 4)
        active_servers_cache = round(substrate_network.get_active_servers_cache_rate(), 4)
        active_links_bw = round(substrate_network.get_active_links_bw_rate(), 4)

        running_sfcs, running_players, running_sessions = running_players_sessions

        cpu_saved = substrate_network.cpu_saved
        cache_saved = substrate_network.cache_saved
        shared_vnfs_count = substrate_network.shared_vnfs_count
        
        sfc_recovery_time = 0
        sfc_recovered = None
        
        if sfc.id in sfcs_crashed:
            sfc_recovered =  0   

        if sfc.id in sfcs_crashed and is_success == 1:
            sfc_recovery_time = time.time() - sfcs_crashed[sfc.id]
            sfc_recovered = 1
        first_loop = (self.first_time == 0)
        time_value = 0  


        if first_loop: 
            self.first_time = current_time
            time_value = 0
        else:        
            time_value = round(current_time - self.first_time,1)
        
        with open(self.flows_file, "a") as file:
            line = str(counter) + ',' + \
                   str(remaining_time) + ',' + \
                   str(current_time) + ',' + \
                   str(time_value) + ',' + \
                   str(sfc.number_of_vnfs) + ',' + \
                   str(cpu_utilization) + ',' + \
                   str(active_servers_cpu) + ',' + \
                   str(bw_utilization) + ',' + \
                   str(active_links_bw) + ',' + \
                   str(cache_utilization) + ',' + \
                   str(active_servers_cache) + ',' + \
                   str(cpu_resilient) + ',' + \
                   str(cache_resilient) + ',' + \
                   str(bw_resilient) + ',' + \
                   str(latency) + ',' + \
                   str(run_duration) + ',' + \
                   str(is_success) + ',' + \
                   str(sfc.arrival_time) + ',' + \
                   str(s2) + "," + \
                   str(sfc.id) + "," + \
                   str(crash_moment) + "," + \
                   str(sfc_recovery_time) + "," + \
                   str(sfc_recovered) + "," + \
                   str(cpu_saved) + "," + \
                   str(cache_saved) + "," + \
                   str(shared_vnfs_count) + "," + \
                   str(running_sfcs) + "," + \
                   str(running_players) + "," + \
                   str(running_sessions) + "," + \
                   str(a_server_was_crashed) + "," + \
                   str(bw_transcode) + "," + \
                   str(users_crashed) + "\n"
            file.write(line)


    def output_cpu_utilization(self, substrate_network, crashed_nodes, deploy_time: float) -> None:
        """
        Outputs the CPU utilization of nodes to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
        """
        cpu_nodes_util = np.array([np.nan if node in crashed_nodes else round(substrate_network.get_node_cpu_used(node), 2)
                                   for node in self.nodes])

        string_cpu_nodes_util = np.array2string(cpu_nodes_util, suppress_small=True,
                                                precision=3, separator=',', formatter={'float_kind': lambda x: "%.2f" % x})

        string_cpu_nodes_util = re.sub(' ', '', string_cpu_nodes_util)
        string_cpu_nodes_util = re.sub('\n', '', string_cpu_nodes_util)

        with open(self.cpu_utilization_file, "a") as file:
            file.write(str(deploy_time) + ',' + string_cpu_nodes_util[1:-1] + '\n')

    def output_cache_utilization(self, substrate_network, crashed_nodes, deploy_time: float) -> None:
        """
        Outputs the cache utilization of nodes to a specified file.

        Args:
            deploy_time (float): The deployment time to record with the utilization data.
        """
        cache_nodes_util = np.array([np.nan if node in crashed_nodes else round(substrate_network.get_node_cache_used(node), 2)
                                     for node in self.nodes])

        string_cache_nodes_util = np.array2string(cache_nodes_util, suppress_small=True,
                                                  precision=3, separator=',', formatter={'float_kind': lambda x: "%.2f" % x})

        string_cache_nodes_util = re.sub(' ', '', string_cache_nodes_util)
        string_cache_nodes_util = re.sub('\n', '', string_cache_nodes_util)

        with open(self.cache_utilization_file, "a") as file:
            file.write(str(deploy_time) + ',' + string_cache_nodes_util[1:-1] + '\n')

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

    # def output_edges_sf_utilization(self, edges_vnf, existing_vnf, sfc_list, deploy_time: float, 
    #                                 route_info: Dict, sfc: object) -> None:
    #     """
    #     Outputs the utilization of edges by service functions (SF) to track network performance.

    #     Args:
    #         deploy_time (float): The deployment time to record with the utilization data.
    #         route_info (Dict): Dictionary containing the route information for SF placement.
    #         sfc (object): The service function chain (SFC) object containing the SF details.
    #     """
    #     if isinstance(route_info, dict):
    #         for key, value in route_info.items():
    #             if key != 'src' and len(value) > 1:
    #                 for i in range(1, len(value)):
    #                     j = i - 1
    #                     trial_edge = (value[i], value[j])
    #                     edge = trial_edge if trial_edge in edges_vnf else (value[j], value[i])
    #                     if sfc.id in existing_vnf:
    #                         existing_vnf[sfc.id].append(key)
    #                     else:
    #                         existing_vnf[sfc.id] = [key]

    #                     edges_vnf[edge].append((sfc.id, key))

    #     # Check if any SFC has been undeployed
    #     for key, value in existing_vnf.items():
    #         if key not in sfc_list:
    #             to_remove = [(key, x) for x in value]

    #             # Iterate over each key in edges_vnf and filter the lists of tuples
    #             for edge in list(edges_vnf.keys()):
    #                 filtered = [tup for tup in edges_vnf[edge] if tup not in to_remove]
    #                 edges_vnf[edge] = filtered

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