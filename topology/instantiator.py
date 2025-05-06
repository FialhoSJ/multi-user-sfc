from topology.predefined.luxembourg import Luxembourg
from topology.predefined.luxembourg_netv2 import LuxembourgV2
from topology.predefined.paloalto import PaloAlto
from topology.experimental.nsfnet import NSFNet
from topology.predefined.santamonica import SantaMonica
from topology.predefined.sample_topology import SampleTopology

class TopologyInstantiator(object):
    def __init__(self):
        self.topology_classes = {
            'nsfnet': NSFNet,
            'paloalto': PaloAlto,
            'santamonica': SantaMonica,
            'luxembourg': Luxembourg,
            'luxembourgv2': LuxembourgV2,
            'test': SampleTopology
        }

    def instantiate_topology(self, type):
        try:
            topology_class = self.topology_classes[type]
            return topology_class()
        except KeyError:
            raise ValueError(f"Topology '{type}' not found")
