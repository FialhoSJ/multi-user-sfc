from sumo.luxembourg.luxembourg_trace import Sumo_Luxembourg
from sumo.small_luxembourg.small_luxembourg_trace import Sumo_Small_Luxembourg


class TracerInstantiator(object):
    def instantiate_tracer(self, type,user_manager):
        if type == 'luxembourg':
            tracer = Sumo_Luxembourg(user_manager=user_manager)
        elif type == 'small luxembourg':
            tracer = Sumo_Small_Luxembourg()
        else:
            raise ValueError('tracer not found')
        
        return tracer