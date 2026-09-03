from muar_sfc.core.vnf import VNF, VNFType


class VNFType2(VNF):
    """
    VNF amplificadora (ex.: encoding): aumenta o volume de tráfego de saída.
    """

    def __init__(self, id, factor=1.0):
        VNF.__init__(self, id)
        self.type = VNFType.TYPE2
        self.factor = factor

    def vnf_bw(self, i):
        return i * self.factor