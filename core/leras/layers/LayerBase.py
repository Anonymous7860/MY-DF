import torch
from core.leras import nn

class LayerBase(torch.nn.Module, nn.Saveable):
    def __init__(self, name=None, **kwargs):
        torch.nn.Module.__init__(self)
        nn.Saveable.__init__(self, name=name)
        # We call build_weights here so that parameters are registered in __init__
        self.build_weights()

    def build_weights(self):
        pass
    
    def forward(self, *args, **kwargs):
        pass
        
    # __call__ is handled by torch.nn.Module

nn.LayerBase = LayerBase