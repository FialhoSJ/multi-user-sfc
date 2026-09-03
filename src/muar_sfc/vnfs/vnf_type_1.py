from muar_sfc.core.vnf import VNF, VNFType


class VNFType1(VNF):
    """
    VNF transparente: a banda de egressa é definida pelo perfil do serviço
    (out_bw do dicionário), preservando o modelo de estágios do MUAR.
    """

    def __init__(self, id):
        VNF.__init__(self, id)
        self.type = VNFType.TYPE1

    def vnf_bw(self, i):
        return self.get_outcome_interface_bandwidth()
