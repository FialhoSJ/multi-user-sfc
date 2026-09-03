from muar_sfc.core.vnf import VNF, VNFType


class VNFType4(VNF):
    """
    VNF agregadora (ex.: telemetria/IoT): consolida fluxos em um volume fixo.
    """

    def __init__(self, id, out_bw=0.0):
        VNF.__init__(self, id)
        self.type = VNFType.TYPE4
        self._out_bw = out_bw

    def vnf_bw(self, i):
        return self._out_bw if self._out_bw else i