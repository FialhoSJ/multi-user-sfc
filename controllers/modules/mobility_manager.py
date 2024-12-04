# MobilityManager.py
import threading

class MobilityManager:
    def __init__(self,tracer):
        """
        Inicializa o MobilityManager com uma instância de tracer.
        
        Args:
            tracer: Uma instância de um tracer (como Sumo_Luxembourg).
        """
        self.tracer = tracer
        self.vehicle_to_service_map = {}
        self.crashed_servers = []
        self.to_remove_vehicles = []
        self.running_sfcs = []
        self.lock = threading.Lock()
        #self.service_to_vehicle_map = {}  # Mapeia serviços para veículos

    def start_simulation(self):
        """Inicia a simulação."""
        self.tracer.start_simulation()

    def add_vehicle(self, sfc):
        """
        Adiciona um veículo à simulação e associa a um grupo de serviços.

        Args:
            player: Identificação do jogador.
            server_start: Servidor inicial do veículo.
            service_group: Grupo de serviços associado ao veículo.
        """
        with self.lock:
            player = sfc.id.split("_")[2][1]
            p_session = sfc.id.split("_")[3]
            player_id = int(player + p_session)
            vehicle_id = f"veh_{player_id}"
            
            if vehicle_id in list(self.vehicle_to_service_map.keys()):
                # Verificando se o SFC já está associado ao veículo
                if sfc.id in self.vehicle_to_service_map[vehicle_id]['sfcs']:
                    pass # As associações são feitas no deploy e, como pode ter vindo de um redeploy, não tem erro claro. 
                else:
                    # Adiciona o SFC à lista de SFCS associada ao veículo
                    self.vehicle_to_service_map[vehicle_id]['sfcs'].append(sfc.id)
                    self.running_sfcs.append(sfc.id)
            else:
                # Caso o veículo não exista no mapeamento, cria o veículo e associa o SFC
                server_start = sfc.dst_node
                self.tracer.create_vehicle(vehicle_id, server_start)
                
                self.running_sfcs.append(sfc.id)
                # Adiciona o veículo com o SFC na lista de serviços
                self.vehicle_to_service_map[vehicle_id] = {
                    'sfcs': [sfc.id],         # Lista com o ID do SFC associado
                    'veh_location': sfc.dst_node   # A localização inicial do veículo
                }

    def check_all_vehicles_position_changes(self):
        """
        Verifica se algum veículo mudou de posição e retorna os IDs das SFCs associadas a eles.

        Retorna:
            dict: Dicionário onde as chaves são os IDs dos veículos que se moveram e os valores são listas
                de SFCs associadas a cada veículo.
        """
        moved_sfcs = []
        new_locations = []
        with self.lock:
            for vehicle_id, vehicle_info in self.tracer.vehicles_info.copy().items():  # Fazendo uma cópia dos itens
                # Posição atual do veículo
                #current_position = vehicle_info['closest_server']
                current_position = self.tracer.get_closest_server(vehicle_id)
                
                # Posição registrada do veículo no mapeamento
                previous_position = self.vehicle_to_service_map[vehicle_id]['veh_location']
                
                # Verificar se a posição mudou
                if current_position != previous_position:
                    # Atualiza a posição do veículo
                    self.vehicle_to_service_map[vehicle_id]['veh_location'] = current_position
                    
                    # Coleta os SFCs associados ao veículo
                    sfc_ids = self.vehicle_to_service_map[vehicle_id]['sfcs']
                    
                    # Adiciona o veículo e os SFCs à lista de veículos que se moveram
                    moved_sfcs.append(sfc_ids)
                    new_locations.append(current_position)

            return moved_sfcs,new_locations

    def remove_sfc(self, sfc_id):
        """
        Remove um SFC de todos os veículos que estão associados a ele.

        Args:
            sfc_id: ID do SFC que deve ser removido.
        """
        with self.lock:
            running_sfcs = self.running_sfcs.copy()
            if sfc_id in running_sfcs:
                for vehicle_id, vehicle_info in list(self.vehicle_to_service_map.items()):
                    # Verifica se o veículo está associado ao SFC
                    if sfc_id in vehicle_info['sfcs']:
                        # Remove o SFC da lista associada ao veículo
                        self.vehicle_to_service_map[vehicle_id]['sfcs'].remove(sfc_id)

                        # Se o veículo não tiver mais SFCs associados, pode ser removido do mapeamento
                        if not self.vehicle_to_service_map[vehicle_id]['sfcs']:
                            self.remove_vehicle(vehicle_id)
                            self.running_sfcs.remove(sfc_id)
    
    def set_crashed_servers(self,crashed_servers):
        self.crashed_servers = crashed_servers
        self.tracer.crashed_servers = crashed_servers
        self.tracer.build_kdtree()

    def remove_vehicle(self, vehicle_id):
        """
        Remove um veículo da simulação e dissocia os serviços relacionados.

        Args:
            vehicle_id: Identificação do veículo.
        """
        del self.vehicle_to_service_map[vehicle_id]
        self.tracer.delete_vehicle(vehicle_id)

    def stop_simulation(self):
        """Encerra a simulação."""
        self.tracer.stop_simulation()
