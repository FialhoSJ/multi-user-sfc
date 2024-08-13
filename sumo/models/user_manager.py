from sumo.models.user import User
from sumo.models.vehicle import Vehicle


class UserManager:
    """
    Gerencia usuários em diferentes sessões, atribuindo veículos e SFCs a eles.
    """
    def __init__(self, num_sessions: int, users_per_session: int):
        self.users = {}
        self.users_running = []
        self.vehicles_runnnig = []
        self.num_sessions = num_sessions
        self.users_per_session = users_per_session
        self._initialize_users()

    def _initialize_users(self):
        """
        Inicializa os usuários para todas as sessões e atribui um ID único a cada um.
        """
        for session_id in range(self.num_sessions):
            for user_id in range(self.users_per_session):
                player_id = int(str(user_id+1) + str(session_id+1))
                user = User(user_id=player_id, player_id=user_id+1, session_id=session_id+1)
                self.users[player_id] = user

    def get_user(self, user_id: int) -> 'User':
        """
        Retorna o usuário com o ID especificado.
        """
        return self.users.get(user_id)

    def set_vehicle_for_user(self, user_id: int, vehicle: 'Vehicle'):
        """
        Atribui um veículo ao usuário especificado.
        """
        user = self.get_user(user_id)
        if user:
            user.add_vehicle(vehicle)

    def get_vehicle_for_user(self, user_id: int) -> 'Vehicle':
        """
        Retorna o veículo do usuário especificado.
        """
        user = self.get_user(user_id)
        if user:
            return user.vehicle

    def user_has_vehicle(self, user_id: int) -> bool:
        """
        Verifica se o usuário especificado possui um veículo.
        """
        return self.get_vehicle_for_user(user_id) is not None

    def add_sfc_to_user(self, user_id: int, sfc_id: str):
        """
        Adiciona um SFC ao usuário especificado.
        """
        user = self.get_user(user_id)
        if user:
            user.add_sfc(sfc_id)
    
    def sfc_exists_for_user(self, user_id: int, sfc_id: str) -> bool:
        """
        Verifica se um determinado sfc_id já existe para o usuário especificado.
        """
        user = self.get_user(user_id)
        if user:
            return sfc_id in user.sfcs
        return False

    def set_sfc_status(self, user_id: int, sfc_id: str, status: str):
        """
        Define o status de um SFC para o usuário especificado.
        """
        user = self.get_user(user_id)
        if user:
            user.set_sfc_status(sfc_id, status)

    def are_all_sfcs_completed(self, user_id: int) -> bool:
        """
        Verifica se todos os SFCs do usuário especificado estão concluídos.
        """
        user = self.get_user(user_id)
        if user:
            return user.are_all_sfcs_completed()
        return False

    def get_vehicle_coordinates(self, user_id: int) -> tuple:
        """
        Retorna as coordenadas do veículo do usuário especificado.
        """
        vehicle = self.get_vehicle_for_user(user_id)
        if vehicle:
            return vehicle.coord
        return None

    def set_vehicle_coordinates(self, user_id: int, coord: tuple):
        """
        Define as coordenadas do veículo do usuário especificado.
        """
        vehicle = self.get_vehicle_for_user(user_id)
        if vehicle:
            vehicle.coord = coord

    def get_trip_attribute(self, user_id: int, attribute: str):
        """
        Retorna um atributo específico da viagem do veículo do usuário especificado.
        """
        vehicle = self.get_vehicle_for_user(user_id)
        if vehicle and vehicle.trip:
            return getattr(vehicle.trip, attribute, None)
        return None

    def set_trip_attribute(self, user_id: int, attribute: str, value):
        """
        Define um atributo específico da viagem do veículo do usuário especificado.
        """
        vehicle = self.get_vehicle_for_user(user_id)
        if vehicle and vehicle.trip and hasattr(vehicle.trip, attribute) and attribute != 'trip_id':
            setattr(vehicle.trip, attribute, value)


    def get_trip_attributes(self, user_id: int) -> dict:
        """
        Retorna os atributos da viagem do veículo do usuário especificado.
        """
        vehicle = self.get_vehicle_for_user(user_id)
        if vehicle and vehicle.trip:
            return {
                'trip_id': vehicle.trip.trip_id,
                'start_server': vehicle.trip.start_server,
                'end_server': vehicle.trip.end_server,
                'route': vehicle.trip.route
            }
        return None

    def update_trip_attributes(self, user_id: int, trip_attributes: dict):
        """
        Atualiza os atributos da viagem do veículo do usuário especificado (exceto o ID da viagem).
        """
        vehicle = self.get_vehicle_for_user(user_id)
        if vehicle and vehicle.trip:
            for attr, value in trip_attributes.items():
                if hasattr(vehicle.trip, attr) and attr != 'trip_id':
                    setattr(vehicle.trip, attr, value)

    def __str__(self) -> str:
        return "\n".join(str(user) for user in self.users.values())