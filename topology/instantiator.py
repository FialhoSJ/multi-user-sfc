from topology.luxembourg import Luxembourg
from topology.paloalto import PaloAlto
from topology.nsfnet import NSFNet
from topology.santamonica import SantaMonica
from topology.sample_topology import SampleTopology
from topology.small_luxembourg import Small_Luxembourg

class TopologyInstantiator(object):
    def instantiate_topology(self, type):
        if type == 'nsfnet':
            topology = NSFNet()
        elif type == 'paloalto':
            topology = PaloAlto()
        elif type == 'santamonica':
            topology = SantaMonica()
        elif type == 'luxembourg':
            topology = Luxembourg() 
        elif type == 'small luxembourg':
            topology = Small_Luxembourg()
        elif type == 'test':
            topology = SampleTopology()
        else:
            raise ValueError('topology not found')

        return topology