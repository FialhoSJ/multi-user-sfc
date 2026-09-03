# config/muar_config.py

import random

import numpy as np

from muar_sfc.controllers.sfc_generator import SFCGenerator
from muar_sfc.core.services.catalog import build_default_catalog

min_latency_acc = 1000

fator = 0.33  # 1 players consumes fator*100 percentage of resources of an Edge Server
# https://ieeexplore.ieee.org/document/9417376

cpb = 10e6  # 10 cycles per Mbit
# print(cpb, "cycles per Mbit")

IA_bw = 150  # * n_players
IA = IA_bw * cpb  # total cycles
IA = 0

# 400x400 pixels RGB with about 0.48 MB per frame
DET_bw = 0.230  # 480 KB * 8 * 60 fps
DET = DET_bw * cpb
# 4 to 12 feature representations of VO, each have 25KB [100,300] KB per frame
FT_bw = 0.144  # 300 KB * 8 * 60  [48,144]
FT = FT_bw * cpb
IA_DET_FT_bw = int(IA_bw + DET_bw + FT_bw)

# each AR VO have 2500 KB in average, 2500 * 12
CA_size = 240  # 2500 KB * 8 * 12 objects 240 for 12 VOs

chr = 1 / 3  # cache hit ratio
MA_bw = int(CA_size * chr)
MA = MA_bw * cpb  #

# not in cache
UNI_bw = int(CA_size * (1 - chr))
UNI = UNI_bw * cpb  #

# matched and non-matched objects
RE_bw = MA_bw + UNI_bw
RE = int(RE_bw * cpb)  #

# https://ieeexplore.ieee.org/document/9316983
# final out put
EC_TC_bw = int(IA_bw * 0.9 * 0.9 * 0.8)
EC_TC = EC_TC_bw * cpb  #

total = IA + DET + FT + MA + UNI + RE + EC_TC
total_inst = round(total * (1 / fator) / 1000000)
total_inst = total * 3
servers_mips = 96_100  # MIPS SAP 96_100

IA = int(IA / total * fator * 100)
DET = DET / total * fator * 100
FT = FT / total * fator * 100
IA_DET_FT = int(IA + DET + FT)  #
MA = int(MA / total * fator * 100)  #
UNI = int(UNI / total * fator * 100)
RE = int(RE / total * fator * 100)  #
EC_TC = int(EC_TC / total * fator * 100)  #
MONO = int(DET + FT + MA + UNI + RE + EC_TC)
CA_size = CA_size / CA_size * fator * 100


class MuarScenario:
    def __init__(self, args, sfc_queue, topology, sfc_poisson_emitter):
        """
        Inicializa a configuração do cenário MUAR.
        """
        self.sfc_queue = sfc_queue
        self.sfc_poisson_emitter = sfc_poisson_emitter
        self.session_counter = 0
        self.topology = topology
        self.src_node = 0
        self.max_duration = int(args.sfc_lifetime)
        self.average_time_session_arrival = 20
        self.factor = 0.33  # 1 jogador consome `FACTOR*100%` dos recursos

        # Configuração com os parâmetros passados por args
        self.config_dict = {
            "n_sessions": args.n_sessions,
            "n_players": args.n_players,
            "mobility_activated": (args.mobility == "y"),
            "shareable": (args.share == "y"),
            "shareable_band": (args.shareband == "y"),
            "allow_delay": (args.allow_delay == "y"),
        }

        # F2: catálogo de serviços + pesos configuráveis (heterogeneidade)
        self.service_catalog = build_default_catalog()
        self.services = [s.strip() for s in args.service_mix.split(",") if s.strip()]
        raw_weights = [float(w.strip()) for w in args.service_weights.split(",") if w.strip()]
        if len(raw_weights) != len(self.services):
            raise ValueError(
                f"service_mix ({len(self.services)}) e service_weights ({len(raw_weights)}) "
                "precisam ter a mesma quantidade de itens"
            )
        self.service_weights = dict(zip(self.services, raw_weights, strict=True))
        for name in self.services:
            if name not in self.service_catalog.list():
                raise ValueError(
                    f"Serviço '{name}' não registrado. Disponíveis: {self.service_catalog.list()}"
                )

    def generate_sfc_session(self, parameter):
        self.session_counter += 1
        counter = str(self.session_counter)
        n_players = self.config_dict["n_players"]

        # F2: sorteia o serviço da sessão conforme os pesos configurados
        service = self.service_catalog.weighted_choice(weights=self.service_weights)

        if service.name == "muar":
            self._generate_muar_session(counter, n_players)
        else:
            self._generate_heterogeneous_session(service, counter, n_players)

        if self.session_counter >= self.config_dict["n_sessions"]:
            self.sfc_poisson_emitter.stop()

    def _generate_muar_session(self, counter, n_players):
        """
        Gera uma sessão de MUAR SFC.
        """
        print("Total Number of MUAR SFCs in session:", self.session_counter)
        routers = self.topology.get_topology_info()["routers"]

        closer_router = random.choice(routers)

        players_cache_sf_list = []
        players_unique_sf_list = []

        for i in range(1, n_players + 1):
            caching_sf_list = []
            caching_sf_list.append(
                {
                    "type": 2,
                    "name": "IA_DET_FT_" + counter,
                    "CPU": round(IA_DET_FT, 2),
                    "cache": 0,
                    "in_bw": round(IA_bw, 2),
                    "out_bw": round(IA_DET_FT_bw, 2),
                    "latency": round((IA_DET_FT * total_inst / servers_mips) * 10, 2),
                }
            )
            caching_sf_list.append(
                {
                    "type": 2,
                    "name": "MA_region_" + str(closer_router),
                    "CPU": round(MA, 2),
                    "cache": round(CA_size, 2),
                    "in_bw": round(IA_DET_FT_bw, 2),
                    "out_bw": round(MA_bw, 2),
                    "latency": round((MA * total_inst / servers_mips) * 10, 2),
                }
            )
            caching_sf_list.append(
                {
                    "type": 2,
                    "name": "RE_region_" + str(closer_router),
                    "CPU": round(RE * chr, 2),
                    "cache": 0,
                    "in_bw": round(MA_bw, 2),
                    "out_bw": round(RE_bw * chr, 2),
                    "latency": round((RE * chr * total_inst / servers_mips) * 10, 2),
                }
            )
            caching_sf_list.append(
                {
                    "type": 2,
                    # "name": "EC_TC_p" + str(i) + "_" + counter,
                    "name": "EC_TC_region_" + str(i) + "_" + counter,
                    "CPU": round(EC_TC * chr, 2),
                    "cache": 0,
                    "in_bw": round(RE_bw * chr, 2),
                    "out_bw": round(EC_TC_bw * chr, 2),
                    "latency": round((EC_TC * chr * total_inst / servers_mips) * 10, 2),
                }
            )
            players_cache_sf_list.append(caching_sf_list)

            unique_sf_list = []
            unique_sf_list.append(
                {
                    "type": 2,
                    "name": "IA_DET_FT_" + counter,
                    "CPU": round(IA_DET_FT, 2),
                    "cache": 0,
                    "in_bw": round(IA_bw, 2),
                    "out_bw": round(IA_DET_FT_bw, 2),
                    "latency": round((IA_DET_FT * total_inst / servers_mips) * 10, 2),
                }
            )
            unique_sf_list.append(
                {
                    "type": 2,
                    "name": "UNI_p" + str(i) + "_" + counter,
                    "CPU": round(UNI, 2),
                    "cache": 0,
                    "in_bw": round(IA_DET_FT_bw, 2),
                    "out_bw": round(UNI_bw, 2),
                    "latency": round((UNI * total_inst / servers_mips) * 10, 2),
                }
            )
            unique_sf_list.append(
                {
                    "type": 2,
                    "name": "RE_p" + str(i) + "_" + counter,
                    "CPU": round(RE * (1 - chr), 2),
                    "cache": 0,
                    "in_bw": round(UNI_bw, 2),
                    "out_bw": round(RE_bw * (1 - chr), 2),
                    "latency": round((RE * (1 - chr) * total_inst / servers_mips) * 10, 2),
                }
            )
            unique_sf_list.append(
                {
                    "type": 2,
                    "name": "EC_TC_p" + str(i) + "_" + counter,
                    "CPU": round(EC_TC * (1 - chr), 2),
                    "cache": 0,
                    "in_bw": round(RE_bw * (1 - chr), 2),
                    "out_bw": round(EC_TC_bw * (1 - chr), 2),
                    "latency": round((EC_TC * (1 - chr) * total_inst / servers_mips) * 10, 2),
                }
            )
            players_unique_sf_list.append(unique_sf_list)

        lifetime = np.random.poisson(self.max_duration)
        # lifetime = int(round(np.random.exponential(max_duration)))
        duration = lifetime
        players_sfc_cache_dict_list = []
        players_sfc_unique_dict_list = []
        for i in range(1, n_players + 1):
            player_cache_dict = {}
            player_cache_dict["name"] = "sfc_cache_p" + str(i) + "_" + counter
            player_cache_dict["vnf_list"] = players_cache_sf_list[i - 1]
            player_cache_dict["bandwidth"] = EC_TC_bw
            player_cache_dict["src_node"] = self.src_node
            player_cache_dict["dst_node"] = f"{i}{counter}"
            player_cache_dict["closer_router"] = closer_router

            player_cache_dict["duration"] = duration
            player_cache_dict["latency"] = min_latency_acc
            player_cache_dict["service_type"] = "muar"  # F4: herança de serviço
            players_sfc_cache_dict_list.append(player_cache_dict)

            player_unique_dict = {}
            player_unique_dict["name"] = "sfc_unique_p" + str(i) + "_" + counter
            player_unique_dict["vnf_list"] = players_unique_sf_list[i - 1]
            player_unique_dict["bandwidth"] = EC_TC_bw
            player_unique_dict["src_node"] = self.src_node
            player_unique_dict["dst_node"] = f"{i}{counter}"
            player_unique_dict["closer_router"] = closer_router

            player_unique_dict["duration"] = duration
            player_unique_dict["latency"] = min_latency_acc
            player_unique_dict["service_type"] = "muar"  # F4: herança de serviço
            players_sfc_unique_dict_list.append(player_unique_dict)

        players_sfc_list = []
        for i in range(1, n_players + 1):
            players_sfc_list.append(
                [
                    SFCGenerator(players_sfc_cache_dict_list[i - 1]).generate(),
                    SFCGenerator(players_sfc_unique_dict_list[i - 1]).generate(),
                ]
            )
            self.sfc_queue.put_sfc(players_sfc_list[i - 1])
            # heapq.heappush(sfc_queue, (1, counter, i, players_sfc_list[i-1]))

    def _generate_heterogeneous_session(self, template, counter, n_players):
        """F2: gera SFCs de serviços não-MUAR usando o template do catálogo."""
        print(f"Total Number of {template.name} SFCs in session:", self.session_counter)
        routers = self.topology.get_topology_info()["routers"]
        closer_router = random.choice(routers)
        duration = np.random.poisson(self.max_duration)

        for i in range(1, n_players + 1):
            chain = template.build_chain(player=i, counter=counter)
            input_bw = chain[0]["in_bw"]
            sfc_dict = template.build_sfc_dict(
                sfc_name=f"sfc_{template.name}_p{i}_{counter}",
                src_node=self.src_node,
                dst_node=f"{i}{counter}",
                closer_router=closer_router,
                bandwidth=input_bw,
                duration=duration,
                player=i,
                counter=counter,
            )
            self.sfc_queue.put_sfc([SFCGenerator(sfc_dict).generate()])

    # def generate_mono_session(self, parameter):
    #     """
    #     Gera uma sessão monolítica MUAR.
    #     """
    #     self.session_counter += 1
    #     n_players = self.config_dict["n_players"]

    #     print("Total Number of Monolithic MUAR in session:", self.session_counter)

    #     dst_node = random.randint(0, 10)
    #     players_mono_sf_list = []

    #     for i in range(1, n_players + 1):
    #         mono_sf_list = [
    #             {
    #                 "type": 2,
    #                 "name": f"IA_{self.session_counter}",
    #                 "CPU": 10,
    #                 "cache": 0,
    #                 "in_bw": 10,
    #                 "out_bw": 10,
    #             },
    #             {
    #                 "type": 2,
    #                 "name": f"MONO_p{i}_{self.session_counter}",
    #                 "CPU": 20,
    #                 "cache": 10,
    #                 "in_bw": 10,
    #                 "out_bw": 10,
    #             },
    #         ]
    #         players_mono_sf_list.append(mono_sf_list)

    #     duration = np.random.poisson(self.max_duration)
    #     players_mono_dict_list = [
    #         {
    #             "name": f"sfc_mono_p{i}_{self.session_counter}",
    #             "vnf_list": players_mono_sf_list[i - 1],
    #             "bandwidth": 10,
    #             "src_node": self.src_node,
    #             "dst_node": dst_node,
    #             "duration": duration,
    #             "latency": 7,
    #         }
    #         for i in range(1, n_players + 1)
    #     ]

    #     players_sfc_list = [
    #         [SFCGenerator(players_mono_dict_list[i - 1]).generate()]
    #         for i in range(1, n_players + 1)
    #     ]

    #     for sfc in players_sfc_list:
    #         self.sfc_queue.put_sfc(sfc)

    #     if self.session_counter >= self.config_dict["n_sessions"]:
    #         print("MONO MUAR Session ## poisson stop  ##")
    #         self.sfc_poisson_emitter.stop()
