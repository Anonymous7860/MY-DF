import torch
from core.leras import nn
from pathlib import Path

class RMSprop:
    """Legacy RMSprop optimizer wrapper (not used by SAENXT - kept for Classic SAEHD compat)."""
    def __init__(self, params=None, lr=0.001, rho=0.9, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, name=None, **kwargs):
        self.name = name
        self.lr = lr
        self.rho = rho
        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.clipnorm = clipnorm
        
        self.optim = None
        self.params = []
        
        if params is not None:
            self.initialize_variables(params)

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        # trainable_weights is a list of parameters (tensors)
        self.params = trainable_weights
        self.optim = torch.optim.RMSprop(self.params, lr=self.lr, alpha=self.rho)
        
    def step(self):
        if self.optim:
            # Handle clipnorm
            if self.clipnorm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.params, self.clipnorm)
                
            self.optim.step()
            
    def zero_grad(self):
        if self.optim:
            self.optim.zero_grad()
            
    def get_weights(self):
        # State dict
        if self.optim:
            return self.optim.state_dict()
        return None

    def load_weights(self, weights):
        if self.optim and weights is not None:
            try:
                if isinstance(weights, (str, Path)):
                    weights = torch.load(str(weights), map_location='cpu', weights_only=False)
                self.optim.load_state_dict(weights)
            except Exception as e:
                msg = str(e)
                if "Invalid magic number" in msg or "unpickling" in msg.lower():
                     print(f"RMSprop: Found legacy optimizer weights. Starting fresh (normal for first run).")
                else:
                     print(f"RMSprop: Optimizer weights not found or incompatible. Starting fresh optimizer state. Error: {e}")

    def save_weights(self, filename):
        if self.optim:
             from pathlib import Path
             path = Path(filename)
             path.parent.mkdir(parents=True, exist_ok=True)
             torch.save(self.optim.state_dict(), str(path))

nn.RMSprop = RMSprop