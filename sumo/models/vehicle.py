from sumo.models.trip import Trip


class Vehicle:
    """
    Representa um veículo com coordenadas e uma viagem.
    """
    def __init__(self, vehicle_id: int, coord: tuple = (0, 0), trip: 'Trip' = None):
        self.vehicle_id = vehicle_id
        self.coord = coord
        self.trip = trip
        self.closest_server = None

    def update_closest_server(self, new_position: str):
        """
        Atualiza o servidor mais próximo com base na nova posição.
        """
        self.closest_server = new_position

    def __str__(self) -> str:
        return f"Coord: {self.coord}, {self.trip}, Closest Server: {self.closest_server}"
