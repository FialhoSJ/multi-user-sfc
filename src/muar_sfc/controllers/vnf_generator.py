from muar_sfc.core.vnf import VNFType
from muar_sfc.vnfs.vnf_type_1 import VNFType1
from muar_sfc.vnfs.vnf_type_2 import VNFType2
from muar_sfc.vnfs.vnf_type_3 import VNFType3
from muar_sfc.vnfs.vnf_type_4 import VNFType4

"""
{type: "TYPE1", "name": "vnf1", "CPU": 100},
"""

class VNFGenerator:
    _registry = {
        VNFType.TYPE1: VNFType1,
        VNFType.TYPE2: VNFType2,
        VNFType.TYPE3: VNFType3,
        VNFType.TYPE4: VNFType4,
    }

    @classmethod
    def register(cls, vnf_type, vnf_class):
        """Permite registrar novos tipos sem editar esta classe (OCP)."""
        cls._registry[vnf_type] = vnf_class


    @classmethod
    def generate(cls, vnf_dict):
        vnf_type = vnf_dict["type"]
        vnf_name = vnf_dict["name"]
        vnf_cpu_request = vnf_dict["CPU"]
        vnf_cache_request = vnf_dict["cache"]
        vnf_in_bw_request = vnf_dict["in_bw"]
        vnf_out_bw_request = vnf_dict["out_bw"]
        vnf_class = cls._registry.get(vnf_type)
        if vnf_class is None:
            raise ValueError(f"VNF type {vnf_type} is not registered")

        params = vnf_dict.get("params", {})
        vnf = vnf_class(vnf_name, **params)
        vnf.set_cpu_request(vnf_cpu_request)
        vnf.set_cache_request(vnf_cache_request)
        vnf.set_income_interface_bandwidth(vnf_in_bw_request)
        vnf.set_outcome_interface_bandwidth(vnf_out_bw_request)
        vnf.service_type = vnf_dict.get("service_type")
        vnf.shareable = vnf_dict.get("shareable", False)  # F3: flag do template do serviço
        return vnf
