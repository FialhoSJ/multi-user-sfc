from sumo.models.vehicle import Vehicle

class User:
    """
    Representa um usuário com um veículo e uma coleção de SFCs.
    """
    def __init__(self, user_id: int, player_id: int, session_id: int):
        self.user_id = user_id
        self.player_id = player_id
        self.session_id = session_id
        self.vehicle = None
        self.sfcs = {}

    def add_vehicle(self, vehicle: 'Vehicle'):
        """
        Atribui um veículo ao usuário.
        """
        self.vehicle = vehicle

    def add_sfc(self, sfc_id: str):
        """
        Adiciona um SFC ao usuário.
        """
        self.sfcs[sfc_id] = {'status': 'running', 'dropped': False}

    def set_sfc_status(self, sfc_id: str, status: str):
        """
        Define o status de um SFC.
        """
        if sfc_id in self.sfcs:
            self.sfcs[sfc_id]['status'] = status

    def drop_sfc(self, sfc_id: str):
        """
        Marca um SFC como descartado.
        """
        if sfc_id in self.sfcs:
            self.sfcs[sfc_id]['dropped'] = True

    def are_all_sfcs_completed(self) -> bool:
        """
        Verifica se todos os SFCs estão concluídos.
        """
        if self.sfcs != {}:
            return all(sfc['status'] == 'completed' for sfc in self.sfcs.values())
        else: 
            return False

        return all(sfc['status'] == 'completed' for sfc in self.sfcs.values())

    def __str__(self) -> str:
        return (f"User ID: {self.user_id}, Player ID: {self.player_id}, "
                f"Session ID: {self.session_id}, Vehicle: {self.vehicle}, SFCs: {self.sfcs}")
