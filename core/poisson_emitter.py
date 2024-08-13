import numpy as np
import time
from threading import Timer

class PoissonEmitter():
    def __init__(self, lam, initial_delay=0):
        self.current_time = 0
        self.next_time = 0
        self.lam = lam
        self.is_output = False
        self.callback = None
        self.is_stop = True
        self.timer = None
        self.initial_delay = initial_delay

    def start(self, func, *args):
        print("Generator is started")
        self.is_stop = False
        self.callback = func
        self.args = args[0]
        if self.initial_delay > 0:
            self.timer = Timer(self.initial_delay, self.reset_timer)
            self.timer.start()
        else:
            self.reset_timer()

    def reset_timer(self):
        self.is_output = False
        interval = np.random.poisson(self.lam)
        print("interval is: ", interval)
        self.timer = Timer(interval, self.run, ())
        self.timer.start()

    def run(self):
        self.is_output = True
        self.callback(self.args)
        if not self.is_stop:
            self.reset_timer()

    def stop(self):
        self.is_stop = True
        self.timer.cancel()
        print("Generator is stopped")
