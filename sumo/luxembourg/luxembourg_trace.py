import csv
import threading
import traceback

from sumo.models.trip import Trip
from sumo.models.vehicle import Vehicle
import traci
import time
import pytz
import datetime
import random
import math
from sumo.luxembourg.config_routes import topology,positions
# Tracer utilizando SUMO.
# Autor: Rodrigo Flexa 
# Data: 17/09/2023

class Sumo_Luxembourg:
    def __init__(self, user_manager,initial_routes, config_file="sumo//luxembourg//luxembourg.sumocfg"):
        self.config_file = config_file
        self.pre_defined_routes = initial_routes
        self.traci_connected = False
        self.counting = 0
        self.veiculos_a_excluir = []
        self.routes_created = []
        self.user_manager = user_manager
        self.lock = threading.Lock()
        self.last_check_time = time.time()
        self.topology = topology
        self.positions = positions
        
    def get_datetime(self):
        utc_now = pytz.utc.localize(datetime.datetime.utcnow())
        currentDT = utc_now.astimezone(pytz.timezone("America/Belem"))
        return currentDT.strftime("%Y-%m-%d %H:%M:%S")

    def calculate_kmph(self, m_per_s):
        return round(m_per_s * 3.6, 2)
    
    def connect_to_sumo(self):
        with self.lock:
            traci.start(["sumo", "-c", self.config_file])
            self.traci_connected = True

    def stop_sumo_simulation(self):
        with self.lock:
            if self.is_simulation_running():
                traci.close()
                self.traci_connected = False
 
    def validate_route(self,server_start,server_end):
        while server_start == server_end:
            server_end = random.choice(list(self.positions.keys()))
        return server_start,server_end

    def update_coords(self):
        try:
            with self.lock:
                if self.is_simulation_running():
                    vehicles = traci.vehicle.getIDList()
                    for vehicle_id in vehicles:
                        x, y = traci.vehicle.getPosition(vehicle_id)
                        # Faça algo com as coordenadas x e y
                        user_id = int(vehicle_id.split("_")[-1])
                        self.user_manager.set_vehicle_coordinates(user_id,(x,y))
        except Exception as e:
            print("Erro in updated coord")
            print(f"Simulation running status: {self.traci_connected}")

    def get_closest_server(self, user_id, crashed_nodes, sfc_location, threshold_factor=0.75):
        """
        Determine the closest server to a given user based on their coordinates, considering a threshold factor.
        
        Parameters:
        - user_id: The ID of the user for whom to find the closest server.
        - crashed_nodes: List of servers crashed. The tracer should not reroute users to those servers.
        - sfc_location: The current server where the user's SFC is located.
        - threshold_factor: A factor (between 0 and 1) to determine how much closer the new server must be.

        Returns:
        - The ID of the closest server to the specified user, considering non-crashed servers only.
        """
        try:
            # Obtém as coordenadas do usuário
            x, y = self.user_manager.get_vehicle_coordinates(user_id)

            # Inicializa com o servidor atual (sfc_location)
            closest_server_id = sfc_location
            closest_server_coord = self.topology[sfc_location]
            actual_min_distance = math.sqrt((x - closest_server_coord[0]) ** 2 + (y - closest_server_coord[1]) ** 2)

            # Define a distância mínima inicial como infinita
            min_distance = float('inf')

            for server_id, (server_x, server_y) in self.topology.items():
                # Pula o servidor se ele estiver na lista de nós quebrados
                if server_id in crashed_nodes:
                    continue

                # Calcula a distância entre o usuário e o servidor
                distance = math.sqrt((x - server_x) ** 2 + (y - server_y) ** 2)

                # Verifica se a nova distância é a menor encontrada até agora
                if distance < min_distance:
                    min_distance = distance
                    closest_server_id = server_id

            # Verifica se o servidor mais próximo encontrado está dentro do threshold multiplicativo
            if min_distance < actual_min_distance * threshold_factor:
                return closest_server_id
            else:
                return sfc_location
        except Exception as e:
            print(f"An error occurred: {e}")
            return -1

    
    def is_vehicle_created(self,user_id):
        with self.lock:
            try:
                # Verifica se o veículo com o ID desejado está na lista
                if self.user_manager.user_has_vehicle(user_id):
                    return True  # O veículo existe na simulação
                else:
                    return False  # O veículo não existe na simulação
                
            except Exception as e:
                print("Erro in is_created")
                print(f"Simulation running status: {self.traci_connected}")

    def is_simulation_running(self):
        if self.traci_connected:
            return True
        else: 
            return False

    def create_vehicle(self, id_number, server_start, vehicle_type="car"):
        # try:
        player_key = int(str(id_number)[0])
        session_key = int(str(id_number)[1:])
        
        if True:
            routes = self.pre_defined_routes[session_key]['route'][player_key][0]
            # del self.pre_defined_routes[session_key]['route'][0]

        trip = self.add_trip_to_simulation(id_number, server_start,routes)
        vehicle_id = f"vehicle_{id_number}"
        traci.vehicle.add(vehicle_id, trip.trip_id, typeID=vehicle_type)
        #x, y = traci.vehicle.getPosition(vehicle_id)
        x, y = self.topology[server_start] 
        vehicle = Vehicle(vehicle_id=vehicle_id,coord=(x, y), trip=trip)
        self.user_manager.set_vehicle_for_user(id_number, vehicle)
        self.user_manager.users_running.append(id_number)
        self.user_manager.vehicles_runnnig.append(vehicle_id)
        # except Exception as e:
        #     print(f"Error in vehicle route creation: {str(e)}")
        #     print(f"Simulation running status: {self.traci_connected}")
        #     #  print(f"Vehicle created: {vehicle_id}", " total:", len(traci.vehicle.getIDList()))

    def add_trip_to_simulation(self,id_number,server_start,routes=-1):
        server_end = random.choice(list(self.positions.keys()))    
        server_start, server_end = self.validate_route(server_start, server_end)

        if True:
            server_start = routes[0]
            server_end = routes[1]

        start_edge = random.choice(self.positions[server_start])
        end_edge = random.choice(self.positions[server_end])
        route = [start_edge, end_edge] 
        trip_id = f"trip_{id_number}"
        
        #if trip_id not in self.routes_created:
        traci.route.add(trip_id, route)
        self.routes_created.append(trip_id)
        return Trip(trip_id=trip_id, start_server=server_start, end_server=server_end, route=route)
        # else:
        #     return False

    def delete_vehicle(self, user_id):
        vehicle_id = f"vehicle_{user_id}"
        with self.lock:
            try:
                self.user_manager.users_running.remove(user_id)
            except:
                print(f"user_id {user_id} not found in users_running")

            try:
                self.user_manager.vehicles_runnnig.remove(vehicle_id)
            except:
                print(f"vehicle_id {vehicle_id} not found in vehicles_runnnig")
            try:
                traci.vehicle.remove(vehicle_id, reason=0)
            except:
                print("Erro in delete")
                print(f"Simulation running status: {self.traci_connected}")

    def check_backup_sfc_location(self,user_id,backup_sfc_loc):
        try:
            x_coord_b, y_coord_b = topology[backup_sfc_loc]
            x_coord_r, y_coord_r = self.user_manager.get_vehicle_coordinates(user_id)

            distance = math.sqrt((x_coord_b - x_coord_r) ** 2 + (y_coord_b - y_coord_r) ** 2)
            if distance < 200:
                return True
            else:
                return False
        except:
            return -1
        
    def gets_next_backup_server(self,actual_backup_sfc_loc,player_id,session_key):
        value_to_remove = actual_backup_sfc_loc
        
        #old_value = self.pre_defined_routes[session_key]['locations'][player_id]
        next_server = self.pre_defined_routes[session_key]['route'][player_id][actual_backup_sfc_loc]
        self.pre_defined_routes[session_key]['route'][player_id-1] = next_server

        return next_server    
        # index = self.pre_defined_routes[session_key]['route'].index(old_value)
        
        # new_server = self.pre_defined_routes[session_key]['route']

        # return new_server

    def update_vehicles(self):
        # try:
        if self.is_simulation_running():
            vehicles = traci.vehicle.getIDList()
            for vehicle_id in vehicles:
                if self.traci_connected:
                    arrived = traci.vehicle.getRouteIndex(vehicle_id) == len(traci.vehicle.getRoute(vehicle_id))-1
                    if arrived:
                        self.reroute_vehicle(vehicle_id)
        # except:
        #     print("Erro in rerouted")
        #     print(f"Simulation running status: {self.traci_connected}")
        #     pass

    def reroute_vehicle(self, vehicle_id):
        """
        Outra Alternativa seria deletar o veículo do usuário e criar outro. Porém, isso pode ser problemático também. 
        """
        user_id = int(vehicle_id.split("_")[-1])
        session_key = int(str(user_id)[1:])
        player_key = int(str(user_id)[0])
        routes = self.pre_defined_routes[session_key]['route'][player_key]
        if len(routes) != 0:
            del self.pre_defined_routes[session_key]['route'][player_key][0]
            routes = routes[0]
            #try:
            user_id = int(vehicle_id.split("_")[-1])

            # New destiny server
            # new_destiny_server = random.choice(list(self.positions.keys()))
            # new_destiny_edge = random.choice(self.positions[new_destiny_server]) 
            new_destiny_server = routes[1]
            new_destiny_edge = random.choice(self.positions[new_destiny_server]) 

            #New origin/closest server
            # new_origin_server = self.user_manager.get_trip_attribute(user_id, 'end_server')
            new_origin_server = routes[0]
            new_origin_edge   = self.user_manager.get_trip_attribute(user_id, 'route')[1]

            new_route = [new_origin_edge,new_destiny_edge]
            new_origin_server, new_destiny_server = self.validate_route(new_origin_server,new_destiny_server)

            new_trip_attributes = {
                'start_server': new_origin_server,
                'end_server':new_destiny_server ,
                'route': new_route}

            self.user_manager.update_trip_attributes(user_id, new_trip_attributes)

            traci.vehicle.changeTarget(vehicle_id, new_destiny_edge)
            #except:
            #    print("Erro in reroute")

    def check_sfc_for_user(self,user_id,sfc_id):
        if self.user_manager.sfc_exists_for_user(user_id, sfc_id):
           pass             
        else:    
            self.user_manager.add_sfc_to_user(user_id,sfc_id)

    def set_sfc_status_for_user(self,user_id,sfc_id,new_status):
        self.user_manager.set_sfc_status(user_id,sfc_id,new_status)

    def check_vehicles(self):
        # Caso todas as sfcs do usuário tenham terminado de rodar, o veículo deve ser excluído
        # TODO Caso as sfcs do usuário ainda estejam rodando, mas por algum motivo o veículo não existe mais (erro no reroute), o veículo deve ser recriado
        for user_id in self.user_manager.users_running:
            if self.user_manager.are_all_sfcs_completed(user_id):
                if self.is_simulation_running():
                    self.delete_vehicle(user_id=user_id)
        
    def simulation_step(self):
        if self.traci_connected:
            traci.simulationStep()
        # with self.lock:
        #     if self.traci_connected:
        #         traci.simulationStep()

    def vehicle_movement_thread(self):
        while self.traci_connected:
            if self.is_simulation_running() == False:
                self.stop_sumo_simulation()
            try:
                self.simulation_step()
                self.update_coords()
                self.update_vehicles()

                current_time = time.time()
                if current_time - self.last_check_time >= 50:
                    self.check_vehicles()
                    self.last_check_time = current_time

                time.sleep(1)
                
            except traci.exceptions.FatalTraCIError as e:
                print("Conexão com o SUMO foi fechada. Finalizando simulação.")
                self.stop_sumo_simulation()  # Chame a função para finalizar a simulação de forma limpa e segura
                break  # Sai do loop para evitar mais chamadas após a simulação ter sido fechada
            except OSError as e:
                if e.winerror == 10054:
                    print("******************Erro de WinError 10054******************")
                    self.stop_sumo_simulation()
                    # Adicione tratamento específico para este erro, se necessário
                else:
                    print("******************Erro tipo OS******************")
                   # print(f"Erro não esperado na movimentação do veículo: {e}")
                    #traceback.print_exc()
            #except Exception as e:
                #print("******************Erro tipo E********************")
                #print(f"Erro não esperado na movimentação do veículo: {e}")
                #traceback.print_exc()


    def start_sumo_simulation(self):
        self.connect_to_sumo()
        # Thread do movimento dos veículos
        movement_thread = threading.Thread(target=self.vehicle_movement_thread)
        movement_thread.start()
        # permite o movimento dos veículos

if __name__ == "__main__":
    sim = Sumo_Luxembourg()
    sim.start_simulation()


