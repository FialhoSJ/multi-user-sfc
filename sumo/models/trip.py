class Trip:
    """
    Representa uma viagem de um ponto de partida a um ponto de destino.
    """
    def __init__(self, trip_id: int = None, start_server: str = None, end_server: str = None, route: list = None):
        self.trip_id = trip_id
        self.start_server = start_server  # closest server
        self.end_server = end_server
        self.route = route

    def set_route(self, route: list):
        """
        Define a rota da viagem.
        """
        self.route = route

    def __str__(self) -> str:
        return f"Trip ID: {self.trip_id}, Start Server: {self.start_server}, End Server: {self.end_server}, Route: {self.route}"