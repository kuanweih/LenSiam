import os
import numpy as np

from collections import OrderedDict
from torch import Tensor

from .plotter import Plotter


class Logger:
    def __init__(self, log_dir, matplotlib=True):
        self.reset(log_dir, matplotlib)

    def reset(self, log_dir=None, matplotlib=True):
        if log_dir is not None:
            self.log_dir=log_dir 
        self.plotter = Plotter() if matplotlib else None
        self.counter = OrderedDict()
        
    def update_scalers(self, ordered_dict):
        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                if value.numel() > 1:     
                    value = value.mean()  
                else:                     
                    value = value         
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1
        
        if self.plotter: 
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, 'plotter.pdf'))
            np.save(os.path.join(self.log_dir, 'logger.npy'), self.plotter.logger)
