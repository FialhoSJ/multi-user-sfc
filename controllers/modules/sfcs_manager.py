class SFCManager:
    def __init__(self):
        self.sfc_list = []
        self.sfc_queue = []
        self.sfcs_routing_info = {}
        self.sfcs_that_deployed = []
        self.sfcs_that_crashed = []
        self.cpu_threshold = 0.8

    def create_sfc(self, sfc_id, service_functions, user_id):
        sfc = {
            "id": sfc_id,
            "functions": service_functions,
            "user": user_id,
            "status": "pending"
        }
        self.sfc_queue.append(sfc)

    def deploy_sfc(self, sfc_id):
        sfc = self.get_sfc_by_id(sfc_id)
        if sfc:
            sfc["status"] = "deployed"
            self.sfcs_that_deployed.append(sfc_id)
            self.sfc_list.append(sfc)
            self.sfc_queue.remove(sfc)

    def crash_sfc(self, sfc_id):
        sfc = self.get_sfc_by_id(sfc_id)
        if sfc:
            sfc["status"] = "crashed"
            self.sfcs_that_crashed.append(sfc_id)
            self.sfc_list.remove(sfc)

    def get_sfc_by_id(self, sfc_id):
        for sfc in self.sfc_list + self.sfc_queue:
            if sfc["id"] == sfc_id:
                return sfc
        return None
