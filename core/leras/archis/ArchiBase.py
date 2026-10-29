from core.leras import nn

class ArchiBase():
    
    def __init__(self, *args, name=None, **kwargs):
        self.name=name
        
    #overridable 
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This archi does not support a direct forward pass. Implement in derived classes.")
    
    #overridable
    def get_weights(self):
        # In PyTorch, weights are typically accessed via state_dict() or by iterating over parameters()
        return self.state_dict()
    
nn.ArchiBase = ArchiBase