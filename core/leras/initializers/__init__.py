import numpy as np
from core.leras import nn
from .CA import CAInitializer

class initializers():
    class ca():
        def __call__(self, shape, dtype=None):
            # In PyTorch, we don't return a tensor here usually.
            # But to satisfy the API check if used elsewhere as an object:
            return None # Placeholder? Or we can return zeros.
        
        @staticmethod
        def generate_batch( data_list, eps_std=0.05 ):
            # data_list: list of (shape, dtype)
            # CAInitializer returns a list of numpy arrays
            return CAInitializer(data_list).run()

nn.initializers = initializers
