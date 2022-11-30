from optparse import check_choice
import sched, time
import re
from threading import Timer
import logging
import threading
import _thread
import copy
from controllers.sfc_generator import SFCGenerator
from controllers.sfc_queue import SFCQueue
import numpy as np
from utils.k_shortest_paths import k_shortest_paths
from generate_trace import generate_trace
from scipy.spatial import KDTree

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# ch = logging.StreamHandler()
ch = logging.FileHandler('./logs/substrate_network_controller.log')
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)
# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')

class SubstrateNetworkController():
    def __init__(self, nw):
        self.processing_nodes = None
        self.cpu_utilization_file = None
        self.cache_utilization_file = None
        self.substrate_network = nw
        self.node_info = {}
        self.sfc_list = []
        self.sfcs_total_latency = {}
        self.sfcs_routing_info = {}
        self.is_stopped = True
        self.clocation = None
        self.update_interval = 1
        self.vehicles_trace = None
        self.number_of_nodes = None
        self.cpu_threshold = 0.8
        self.over_threshold_nodes_list = []
        self.timer = None
        self.sfc_queue = None
        self.temp_sfc_queue = SFCQueue()
        self.prob = None
        self.sfc = None
        self.sfc_list_player = []
        self.sfc_id_duration = {}
        self.bw_utilization = []
        self.cpu_utilization = []
        self.success_flag = []
        self.deploy_failure = 0
        self.flows = 0
        self.counter = 0
        self.lock = True
        self.alg = None
        self.alg_name = None
        self.file_name = None
        self.next_index = None
        self.players_sfc_list = []
        self.prob_array = None
        self.file_name_list = []
        #self.file_name = './results/' + dt.now().strftime('%Y%m%d%H%M%S') + '.csv'
        #with open(self.file_name, "a") as f:
        #    f.write("No., timestamp, number of sfc, CPU utilization, bandwidth utilization, latency, duration, success, arrival time, depart time" + "\n")

    def start(self):
        #self.vehicles_trace = generate_trace()
        if not self.is_stopped:
            self.is_stopped = True
            time.sleep(2*self.update_interval)
        self.is_stopped = False
        self.update()
        self.check_sfc_duration()
        _thread.start_new_thread(self.run, ())
        
    def output_nodes_cpu_cache_utilization(self):
        cpu_nodes_util = np.array(list(map(self.substrate_network.get_node_cpu_used, \
            self.processing_nodes)))
        cache_nodes_util = np.array(list(map(self.substrate_network.get_node_cache_used, \
        self.processing_nodes)))
        string_cpu_nodes_util = np.array2string(cpu_nodes_util, suppress_small=True,
            precision=3, separator=',', formatter={'float_kind':lambda x: "%.2f" % x})
        string_cache_nodes_util = np.array2string(cache_nodes_util, suppress_small=True,
            precision=3, separator=',', formatter={'float_kind':lambda x: "%.2f" % x})

        string_cpu_nodes_util = re.sub(' ', '', string_cpu_nodes_util)
        string_cache_nodes_util = re.sub(' ', '', string_cache_nodes_util)

        string_cpu_nodes_util = re.sub('\n', '', string_cpu_nodes_util)
        string_cache_nodes_util = re.sub('\n', '', string_cache_nodes_util)

        with open(self.cpu_utilization_file, "a") as file:
            file.write(string_cpu_nodes_util[1:-1] + '\n')
        with open(self.cache_utilization_file, "a") as file:
            file.write(string_cache_nodes_util[1:-1] + '\n')

    def output_nodes_information(self):
        self.substrate_network.print_out_nodes_information()

    def output_edges_information(self):
        self.substrate_network.print_out_edges_information()

    def output_info(self):
        self.output_nodes_information()
        self.output_edges_information()

    def get_nodes_information(self):
        for node in self.substrate_network.nodes():
            self.node_info[node] = self.get_node_information(node)

    def update(self):
        self.substrate_network.update()
        self.check_cpu_threshold()
        #logger.warn(self.over_threshold_nodes_list)
        #logger.warn(("sfc_list "+str(self.sfc_list)))
        # self.check_sfc_duration()
        # if not self.is_stopped:
        #     self.timer = Timer(self.update_interval, self.check_sfc_duration, ()).start()

    def _change_user_location(self, prob, clocation, src_location):
        """
            Updates user location for mobility
            simulation.
        """
        #nodes_array = np.array(list(self.substrate_network.get_neighbours(clocation)))
        nodes_array = np.arange(self.number_of_nodes)
        index_src = np.where(nodes_array == src_location)
        nodes_array = np.delete(nodes_array, index_src)
        choosen_node = np.random.choice(nodes_array)
        user_new_location = np.random.choice([clocation, choosen_node], p=[1  - float(prob), float(prob)])
        return user_new_location
    def check_cpu_threshold(self):
        self.over_threshold_nodes_list = []
        for node in self.substrate_network.nodes():
            self.check_node_cpu_threshold(node)
    def stop(self):
        self.is_stopped = True
        if self.timer:
            self.timer.cancel()
    def deploy_sfc(self, sfc):
        if sfc.id in self.sfc_list:
            return
        alg = self.alg
        alg.clear_all()
        alg.install_substrate_network(self.substrate_network)
        alg.install_SFC(sfc)
        s = time.time()
        alg.start_algorithm()
        s2 = time.time()

        route_info = alg.get_route_info()
        latency = alg.get_latency()
        # if routing info already in the dictionary, there's a redeploy.
        if self.alg_name == 'musfico' and sfc.id in self.sfcs_routing_info.keys():
            #print(f'dp,{route_info},{latency}', file=open('test_latency.txt', 'a'))
            route_info = {}
            # get stored route info 
            route_info = copy.deepcopy(self.sfcs_routing_info[sfc.id])
            # gets dst vnf 
            dst_vnf = sfc.get_dst_vnf()
            previous_vnf = sfc.get_previous_vnf(dst_vnf)
            dst_substrate_node = sfc.get_substrate_node(dst_vnf)
            prev_vnf_node = route_info[previous_vnf.id][0]
            # applies k shortest to link the last vnf with the previous one
            shortest_path = k_shortest_paths(self.substrate_network, prev_vnf_node, 
                dst_substrate_node, k=1, weight='latency')
            # get the shortest among the k shortes paths
            route_info[previous_vnf.id] = shortest_path[0]
            latency = 0
            for vnf_id in route_info.keys():
                if vnf_id == 'src':
                    continue
                path = route_info[vnf_id]
                for i in range(len(path) - 1):
                    edge_latency = self.substrate_network.get_link_latency(
                        path[i], path[i + 1])
                    latency += edge_latency
            #print(f'musfico,{route_info},{latency}', file=open('test_latency.txt', 'a'))
            if latency > sfc.get_latency_request():
                route_info = False
            if latency == 0:
                route_info = False
                latency = None
        if route_info:
            if len(route_info.keys()) != 6 and self.sfc == 'on' or len(route_info.keys()) != 4 and self.sfc == 'off':
                latency = None
                route_info = False

        #else:
        #    print("CTRL:",route_info)
        number_of_vnf = sfc.number_of_vnfs
        cpu_utilization = 0
        bw_utilization = 0
        cache_utilization = 0
        is_success = 0
        current_time = s2
        run_duration = s2 - s

        arrival_time = sfc.arrival_time
        sfc.depart_time = s2


        if route_info:
            self.substrate_network.deploy_sfc(sfc, route_info)
            self.sfc_list.append(sfc.id)
            self.sfc_id_duration[sfc.id] = sfc.duration

            # latency = node_info[sfc.get_dst_vnf().get_substrate_node()]['dst']["latency"]
            is_success = 1
            self.deploy_success(sfc)
            self.temp_sfc_queue.put_sfc(sfc)
            if sfc not in self.sfcs_routing_info.keys():
                self.sfcs_routing_info[sfc.id] = copy.deepcopy(route_info)
            if self.alg_name == 'musfico':
                self.sfcs_total_latency[sfc.id] = alg.latency_minus_dst
        else:
            self.deploy_failure = 1
            self.deploy_failed(sfc)
        # self.substrate_network.update()
        self.output_nodes_cpu_cache_utilization()
        self.update()

        cpu_utilization = self.substrate_network.get_cpu_utilization_rate()
        #cpu_utilization = self.substrate_network.get_cpu_overloaded_utilization_rate()
        bw_utilization = self.substrate_network.get_bandwidth_utilization_rate()
        #bw_utilization = self.substrate_network.get_network_utility()
        cache_utilization = self.substrate_network.get_bandwidth_utilization_rate()
        if is_success == 1:
            if cpu_utilization > 1 or cache_utilization > 1 or bw_utilization > 1:
                self.undeploy_sfc(sfc.id)
                is_success = 0
        self.counter += 1
        with open(self.file_name, "a") as f:
            # f.writelines("timestamp, number of sfc, CPU utilization, bandwidth utilization, latency, duration, success")
            line = str(self.counter) + ',' + \
                   str(current_time) + ',' + \
                   str(number_of_vnf) + ',' + \
                   str(cpu_utilization) + ',' + \
                   str(bw_utilization) + ',' + \
                   str(cache_utilization) + ',' + \
                   str(latency) + ',' + \
                   str(run_duration) + ',' + \
                   str(is_success) + ',' + \
                   str(arrival_time) + ',' +\
                   str(s2) + "," + \
                   str(sfc.id) + "\n"
            f.write(line)
        """
        prefix = sfc.id.rsplit("_",1)[0]
        print(prefix)
        session_id = int(sfc.id.rsplit("_",1)[1])
        #self.prob_array = [0,0.1,0.2,0.3,0.4,0.5]
        #mob_prob = session_id % len(self.prob_array) -1
        c_index = int(session_id / len(self.prob_array))
        c_sfc_id =  c_index + 1
        with open(self.file_name_list[self.get_session_mob_probability_index(sfc.id)], "a") as f:
            # f.writelines("timestamp, number of sfc, CPU utilization, bandwidth utilization, latency, duration, success")
            line = str(self.counter) + ',' + \
                   str(current_time) + ',' + \
                   str(number_of_vnf) + ',' + \
                   str(cpu_utilization) + ',' + \
                   str(bw_utilization) + ',' + \
                   str(cache_utilization) + ',' + \
                   str(latency) + ',' + \
                   str(run_duration) + ',' + \
                   str(is_success) + ',' + \
                   str(arrival_time) + ',' +\
                   str(s2) + "," + \
                   prefix + "_" + str(c_sfc_id) + "\n"
            f.write(line)
        """
        print("__________________________________________")
        self.output_info()
        print("")
        if is_success == 1:
            return True
        return False
    def deploy_success(self, sfc):
        print("deploy succeed, sfc: ", sfc.id)

    def deploy_failed(self, sfc):
        print(" deploy FAILED, sfc: ", sfc.id)

    def get_route_info(self):
        return self.substrate_network.sfc_route_info

    def undeploy_sfc(self, sfc_id):
        if sfc_id not in self.sfc_list:
            print(sfc_id, "not on the substrate network")
            return -1
        # self.stop()
        # time.sleep(self.update_interval*2)
        self.substrate_network.undeploy_sfc(sfc_id)
        self.sfc_list.remove(sfc_id)
        del self.sfc_id_duration[sfc_id]
        self.update()
        #if len(self.sfc_list) == 0:
        #    print('subs net control stop 334')
        #    self.stop()
        #    return 0
        # self.start()
        
    def handle_cpu_over_threshold(self, alg):
        """Seems use to for test. 
        """
        ## undeploy the sfc, redeploy sfc by disable the over threshold cpu
        self.stop()
        for node in self.over_threshold_nodes_list:
            # (sfc_id, vnf) = sn.get_node_sfc_vnf_list(node)
            sfc_vnf_list = self.substrate_network.get_node_sfc_vnf_list(node)
            for (sfc_id, vnf) in sfc_vnf_list:
                ## todo: here we should consider which sfc need to be undployed. May according to priority or some history data or SLA. or cost...
                sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                self.undeploy_sfc(sfc_id)
                sn = copy.deepcopy(self.substrate_network)
                sn.set_node_cpu_capacity(node, 0)
                sn.set_node_cpu_free(node, 0)
                alg.install_substrate_network(sn)
                alg.install_SFC(sfc)
                alg.start_algorithm()
                route_info = alg.get_route_info()
                if sfc.id in self.sfc_list:
                    print("sfc has been deployed")
                    return
                self.substrate_network.deploy_sfc(sfc, route_info)
                self.sfc_list.append(sfc.id)
                self.substrate_network.update()
        self.start()

    def run_old(self):
        while not self.is_stopped:
            sfc = self.sfc_queue.peek_sfc()# this is blocking
            print("Substrate network gets a new sfc: ", sfc.id)
            #print(str(sfc))
            print("queue_size: " + str(self.sfc_queue.qsize()))
            s = time.time()
            self.deploy_sfc(sfc)
            self.update()
            s2 = time.time()
            print("algorithm take time: ", s2 - s)
            if sfc.id == 'sfc_fi_p4_'+str(self.flows):
                print('run stopping muvr sfc')
                self.stop()
                exit(1)
            if sfc.id == 'sfc_p4_' + str(self.flows):
                print('run stopping muvr')
                self.stop()
                exit(1)

    def run(self):
        while not self.is_stopped:
            self.submit_sfcs() 

    def output_nodes_information(self):
        self.substrate_network.print_out_nodes_information()

    def output_edges_information(self):
        self.substrate_network.print_out_edges_information()

    def output_info(self):
        self.output_nodes_information()
        self.output_edges_information()

    def get_node_information(self, node_id):
        sfc_vnf_list = self.substrate_network.get_node_sfc_vnf_list(node_id)
        cpu_used = self.substrate_network.get_node_cpu_used(node_id)
        cpu_free = self.substrate_network.get_node_cpu_free(node_id)
        cpu_capacity = self.substrate_network.get_node_cpu_capacity(node_id)
        total_cpu_used = self.substrate_network.total_cpu_used
        total_cpu_capacity = self.substrate_network.total_cpu_capacity
        cache_used = self.substrate_network.get_node_cache_used(node_id)
        cache_free = self.substrate_network.get_node_cache_free(node_id)
        cache_capacity = self.substrate_network.get_node_cache_capacity(node_id)
        total_cache_used = self.substrate_network.total_cache_used
        total_cache_capacity = self.substrate_network.total_cache_capacity
        return (sfc_vnf_list, cpu_used, cpu_free, cpu_capacity, total_cpu_used, total_cpu_capacity, cache_used, cache_free, cache_capacity, total_cache_used, total_cache_capacity)

    def update(self):
        self.substrate_network.update()
        self.check_cpu_threshold()
        #logger.warn(self.over_threshold_nodes_list)
        #logger.warn(("sfc_list "+str(self.sfc_list)))
        # self.check_sfc_duration()
        # if not self.is_stopped:
        #     self.timer = Timer(self.update_interval, self.check_sfc_duration, ()).start()

    def check_node_cpu_threshold(self, node_id):
        cpu_used = self.substrate_network.get_node_cpu_used(node_id)
        cpu_capacity = self.substrate_network.get_node_cpu_capacity(node_id)
        if cpu_capacity == 0:
            return
        if float(cpu_used)/float(cpu_capacity) > self.cpu_threshold:
            self.over_threshold_nodes_list.append(node_id)
            # print "node:", node_id, "over threshold"
            #logger.warn("Node: %s, over threshold", str(node_id))

    def check_cpu_threshold(self):
        self.over_threshold_nodes_list = []
        for node in self.substrate_network.nodes():
            self.check_node_cpu_threshold(node)
    def check_sfc_duration(self):
        time1 = time.time()
        remove_list = []
        for sfc_id, duration in list(self.sfc_id_duration.items()):
            if duration <= 1:
                remove_list.append(sfc_id)
                # if duration is over, sfc routing info no longer needed 
                del self.sfcs_routing_info[sfc_id]
                if self.alg_name == 'musfico':
                    del self.sfcs_total_latency[sfc_id]
                continue
            self.sfc_id_duration[sfc_id] = duration - 1
        for sfc_id in remove_list:
            self.undeploy_sfc(sfc_id)
        #if len(self.players_sfc_list) != 0 and not self.lock:
        if len(self.players_sfc_list) != 0:
            idx_session = np.random.choice(np.arange(len(self.players_sfc_list)))
            session = copy.deepcopy(self.players_sfc_list[idx_session])
            for sfc_id in session:
                new_sfc_list = []
                try:
                    sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                except:
                    break
                clocation = sfc.dst.substrate_node
                src_location = sfc.src.substrate_node
                user_new_location = self._change_user_location(prob=self.prob, clocation=clocation, src_location=src_location)
                if user_new_location != clocation:
                    print(f"user changed location from {clocation} to {user_new_location}")
                    duration = self.sfc_id_duration[sfc_id]
                    self.undeploy_sfc(sfc_id)
                    new_vnfs_list_dict = copy.deepcopy(sfc.vnfs_dict)
                    if re.search('cache', sfc.id) is not None:
                        old_loc = sfc.dst.substrate_node
                        ma_old_key = 'MA_region_' + str(old_loc)
                        re_old_key = 'RE_region_' + str(old_loc)
                        if ma_old_key in self.sfcs_routing_info[sfc.id].keys(): 
                            stored_info = copy.deepcopy(self.sfcs_routing_info[sfc.id][ma_old_key])
                            del self.sfcs_routing_info[sfc.id][ma_old_key]
                            self.sfcs_routing_info[sfc.id]['MA_region_' + str(user_new_location)] = stored_info
                            new_vnfs_list_dict[1]['name'] = 'MA_region_' + str(user_new_location)

                        if re_old_key in self.sfcs_routing_info[sfc.id].keys():
                            stored_info = copy.deepcopy(self.sfcs_routing_info[sfc.id][re_old_key])
                            del self.sfcs_routing_info[sfc.id][re_old_key]
                            self.sfcs_routing_info[sfc.id]['RE_region_' + str(user_new_location)] = stored_info
                            new_vnfs_list_dict[2]['name'] = 'RE_region_' + str(user_new_location)

                    new_sfc_dict = {}
                    new_sfc_dict["name"] = sfc.id
                    new_sfc_dict["vnf_list"] = new_vnfs_list_dict
                    new_sfc_dict["bandwidth"] = sfc.input_throughput
                    new_sfc_dict["src_node"] = sfc.src.substrate_node
                    new_sfc_dict["dst_node"] = user_new_location 
                    new_sfc_dict["duration"] = duration 
                    new_sfc_dict["latency"] = sfc.latency_request
                    new_sfc = SFCGenerator(new_sfc_dict).generate()
                    new_sfc_list.append(new_sfc)
                if len(new_sfc_list) != 0:
                    self.sfc_queue.put_begin(new_sfc_list)
            #self.lock = True
        time2 = time.time()
        #logger.warn(self.sfc_id_duration)
        #print('CONTROLLER:', len(list(self.sfc_id_duration.items())))
        #print(self.last_sfc)
        if time2 - time1 > 1:
            if not self.is_stopped:
                self.timer = Timer(0, self.check_sfc_duration, ()).start()
        else:
            if not self.is_stopped:
                self.timer = Timer((self.update_interval - (time2 - time1)), self.check_sfc_duration, ()).start()
        
    def stop(self):
        self.is_stopped = True
        if self.timer:
            self.timer.cancel()
    def deploy_success(self, sfc):
        print("deploy succeed, sfc: ", sfc.id)

    def deploy_failed(self, sfc):
        print(" deploy FAILED, sfc: ", sfc.id)

    def get_route_info(self):
        return self.substrate_network.sfc_route_info

    def undeploy_sfc(self, sfc_id):
        if sfc_id not in self.sfc_list:
            print(sfc_id, "not on the substrate network")
            return -1
        # self.stop()
        # time.sleep(self.update_interval*2)
        self.substrate_network.undeploy_sfc(sfc_id)
        self.sfc_list.remove(sfc_id)
        del self.sfc_id_duration[sfc_id]
        self.update()
        #if len(self.sfc_list) == 0:
        #    print('subs net control stop 334')
        #    self.stop()
        #    return 0
        # self.start()
        
    def handle_cpu_over_threshold(self, alg):
        """Seems use to for test. 
        """
        ## undeploy the sfc, redeploy sfc by disable the over threshold cpu
        self.stop()
        for node in self.over_threshold_nodes_list:
            # (sfc_id, vnf) = sn.get_node_sfc_vnf_list(node)
            sfc_vnf_list = self.substrate_network.get_node_sfc_vnf_list(node)
            for (sfc_id, vnf) in sfc_vnf_list:
                ## todo: here we should consider which sfc need to be undployed. May according to priority or some history data or SLA. or cost...
                sfc = self.substrate_network.get_sfc_by_id(sfc_id)
                self.undeploy_sfc(sfc_id)
                sn = copy.deepcopy(self.substrate_network)
                sn.set_node_cpu_capacity(node, 0)
                sn.set_node_cpu_free(node, 0)
                alg.install_substrate_network(sn)
                alg.install_SFC(sfc)
                alg.start_algorithm()
                route_info = alg.get_route_info()
                if sfc.id in self.sfc_list:
                    print("sfc has been deployed")
                    return
                self.substrate_network.deploy_sfc(sfc, route_info)
                self.sfc_list.append(sfc.id)
                self.substrate_network.update()
        self.start()

    def run_old(self):
        while not self.is_stopped:
            sfc = self.sfc_queue.peek_sfc()# this is blocking
            print("Substrate network gets a new sfc: ", sfc.id)
            #print(str(sfc))
            print("queue_size: " + str(self.sfc_queue.qsize()))
            s = time.time()
            self.deploy_sfc(sfc)
            self.update()
            s2 = time.time()
            print("algorithm take time: ", s2 - s)
            if sfc.id == 'sfc_fi_p4_'+str(self.flows):
                print('run stopping muvr sfc')
                self.stop()
                exit(1)
            if sfc.id == 'sfc_p4_' + str(self.flows):
                print('run stopping muvr')
                self.stop()
                exit(1)

    def run(self):
        while not self.is_stopped:
            self.submit_sfcs() 

    def submit_sfcs(self):
        while not self.is_stopped:
            #if not self.lock:
            #    continue
            sfc_list = self.sfc_queue.peek_sfc()
            print("queue_size: " + str(self.sfc_queue.qsize()))
            player_sfc_id_list = []
            for sfc in sfc_list:
                #print(sfc)
                #print()
                #return   
                s = time.time()
                self.deploy_sfc(sfc)
                player_sfc_id_list.append(sfc.id)
                self.update()
                s2 = time.time()
                print("          algorithm take time: ", round(s2 - s,4))
                #if sfc.id == 'sfc_unique_p4_'+ str(self.flows) or sfc.id == 'sfc_mono_p4_'+str(self.flows):
                #    print('CONTROLLER: stopping muar session')
                #    self.stop()
                #    exit(1)
                if sfc.id == 'sfc_unique_p4_'+ str(self.flows) or sfc.id == 'sfc_mono_p4_'+str(self.flows):
                    print('Last SFC released')
                    self.stop()
                    exit()
            self.players_sfc_list.append(player_sfc_id_list)
            #self.lock = False
