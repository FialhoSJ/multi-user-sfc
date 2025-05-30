# MobilityManager.py
import time
import random
import threading
from typing import Optional

from sumo.luxembourg.luxembourg_trace import Sumo_Luxembourg
from sumo.tracer_instantiator import TracerInstantiator

class MobilityManager:
    def __init__(self,args):
        """
        Inicializa o MobilityManager com uma instância de tracer.
        
        Args:
            tracer: Uma instância de um tracer (como Sumo_Luxembourg).
        """
        self.activated = (args.mobility == 'y')
        self.tracer : Optional[Sumo_Luxembourg] = TracerInstantiator().instantiate_tracer(args.topology) if (args.mobility == 'y') else None
        self.players_tracker = {}
        self.to_remove_vehicles = []
        self.running_sfcs = []
        self.lock = threading.Lock()
        #self.service_to_vehicle_map = {}  # Mapeia serviços para veículos

    def start_simulation(self):
        """Inicia a simulação."""
        self.tracer.start_simulation()

    def get_md_distance_from_router(self,id,router):
        distance_from_connected_router = self.tracer.get_server_distance_from_car(id, router)
        print(distance_from_connected_router)
        return distance_from_connected_router
    
    def add_player(self, group_id, closer_router, sfc_id_list):
        if group_id in self.players_tracker:  
            self.players_tracker[group_id]['redeploying'] = False
            return
        else:
            self.tracer.create_vehicle(group_id, closer_router)
            self.players_tracker[group_id] = {'sfc_list':sfc_id_list,
                                              'connected_router':closer_router,
                                              'redeploying':False}
            
    def stop_simulation(self):
        """Encerra a simulação."""
        self.tracer.stop_simulation()

    def remove_player(self, player_id):
        if player_id in self.players_tracker:  
            self.tracer.disconnect_vehicle(player_id) # Veh ainda existirá no tracer, porém não será mais reroteado
            del self.players_tracker[player_id]
        else:
            raise ValueError(f'Veh do player {player_id} não encontrado na simulação')

    def mark_vehicle_as_redeploying(self,vehicle_id):
        self.players_tracker[vehicle_id]['redeploying'] = True

    def check_all_vehicles_position_changes(self):
        """
        Verifica se algum veículo mudou de posição e retorna os IDs das SFCs associadas a eles.

        Retorna:
            dict: Dicionário onde as chaves são os IDs dos veículos que se moveram e os valores são listas
                de SFCs associadas a cada veículo.
        """
        moved_sfcs = []
        new_locations = []

        #for vehicle_id in list(self.tracer.vehicles_info.keys()):  # Fazendo uma cópia dos itens
        for vehicle_id, vehicle_info in self.players_tracker.items():

            if  not vehicle_info['redeploying']:
                closest_router, dist_from_router = self.tracer.get_closest_server(vehicle_id)

                # Posição registrada do veículo no mapeamento
                connected_router = self.players_tracker[vehicle_id]['connected_router']
                distance_from_connected_router = self.tracer.get_server_distance_from_car(vehicle_id, connected_router)

                # Verificar se a posição mudou e a nova distância é pelo menos 35% menor
                if closest_router != connected_router: 
                    reduction_percent = (distance_from_connected_router - dist_from_router) / distance_from_connected_router
                    # A nova distância é pelo menos 30% menor
                    if reduction_percent >= 0.30:
                        # Atualiza a posição e a distância do veículo no mapeamento
                        self.players_tracker[vehicle_id]['connected_router'] = closest_router      
                        # Marca esse veh fazendo redeploy
                        self.mark_vehicle_as_redeploying(vehicle_id)

                        # Coleta os SFCs associados ao veículo
                        sfcs_ids = self.players_tracker[vehicle_id]['sfc_list']
                        
                        # Adiciona o veículo e os SFCs à lista de veículos que se moveram
                        moved_sfcs.append(sfcs_ids)
                        new_locations.append(closest_router)
        return moved_sfcs, new_locations

        #try:
        # with self.lock:
        # for sfc in sfc_list:
        #     player = sfc.id.split("_")[-2][1]
        #     p_session = sfc.id.split("_")[-1]
        #     player_id = int(player + p_session)
        #     vehicle_id = f"veh_{player_id}"
            
        #     if vehicle_id in list(self.vehicle_to_service_map.keys()):  
        #         # Não adiciona se o veh estiver bloqueado ou desconectado
        #         if self.vehicle_to_service_map[vehicle_id]['connected'] == False:
        #             return
                
        #         # Verificando se o SFC já está associado ao veículo
        #         if sfc.id in self.vehicle_to_service_map[vehicle_id]['sfcs']:
        #             self.vehicle_to_service_map[vehicle_id]['status'] = 'available' # Redeploy bem sucedido
        #             self.vehicle_to_service_map[vehicle_id]['connected'] = True
        #             #pass  # As associações são feitas no deploy e, como pode ter vindo de um redeploy, não tem erro claro.
        #         else:
        #             # Adiciona o SFC à lista de SFCS associada ao veículo
        #             self.vehicle_to_service_map[vehicle_id]['sfcs'].append(sfc.id)
        #             self.running_sfcs.append(sfc.id)
        #     else:
        #         # Caso o veículo não exista no mapeamento, cria o veículo e associa o SFC
        #         server_start = sfc.dst_node
        #         self.tracer.create_vehicle(vehicle_id, server_start)
                
        #         self.running_sfcs.append(sfc.id)
        #         # Adiciona o veículo com o SFC na lista de serviços
        #         self.vehicle_to_service_map[vehicle_id] = {
        #             'sfcs': [sfc.id],         # Lista com o ID do SFC associado
        #             'veh_location': sfc.dst_node,   # A localização inicial do veículo
        #             'distance': self.tracer.get_server_distance_from_car(vehicle_id,server_start),
        #             'time': time.time(),
        #             'connected': True,
        #             'status':'available' # 'available': pode alterar de servidor,'moving': movendo de servidor,'blocked':não 'está' mais na simulação; 
        #         }
        # except Exception as e:
        #     print(f"Erro ao adicionar veículo para o SFC {sfc.id}: {str(e)}")