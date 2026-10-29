import types
import numpy as np
import torch
from core.interact import interact as io
from core.leras import nn

class ModelBase(nn.LayerBase):
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(name=name)
        self.layers = torch.nn.ModuleList()
        self.layers_by_name = {}
        self.built = False
        self.args = args
        self.kwargs = kwargs
        
    def _build_sub(self, layer, name):
        if isinstance (layer, list):
            for i,sublayer in enumerate(layer):
                self._build_sub(sublayer, f"{name}_{i}")
        elif isinstance (layer, dict):
            for subname in layer.keys():
                sublayer = layer[subname]
                self._build_sub(sublayer, f"{name}_{subname}")
        elif isinstance (layer, nn.LayerBase) or \
                isinstance (layer, ModelBase):

            if layer.name is None:
                layer.name = name

            # In PyTorch, we trigger build_weights if not built?
            # Creating the layer usually builds weights in our PyTorch port.
            # But Leras was lazy.
            
            # If layer is ModelBase, it might need build()
            if isinstance(layer, ModelBase):
                if not layer.built:
                    layer.build()

            self.layers.append (layer)
            self.layers_by_name[layer.name] = layer
            
            # Register as submodule for PyTorch to track parameters
            # clean name for attribute assignment?
            safe_name = layer.name.replace(' ', '_').replace(':', '_').replace('/', '_')
            if not hasattr(self, safe_name):
                 self.add_module(safe_name, layer)
            else:
                 # fallback if name conflict
                 self.add_module(f"layer_{len(self.layers)}", layer)

    def xor_list(self, lst1, lst2):
        return  [value for value in lst1+lst2 if (value not in lst1) or (value not in lst2)  ]

    def build(self):
        # Emulate lazy build using the generator approach or just call on_build
        # The generator approach allowed interleaving build steps.
        # We can support it by iterating.
        
        current_keys = set(vars(self).keys()).union(self._modules.keys())
        generator = None
        while True:
            if generator is None:
                generator = self.on_build(*self.args, **self.kwargs)
                if not isinstance(generator, types.GeneratorType):
                    generator = None

            if generator is not None:
                try:
                    next(generator)
                except StopIteration:
                    generator = None

            # Check for new attributes
            new_keys = set(vars(self).keys()).union(self._modules.keys())
            diff = new_keys - current_keys


            for name in sorted(diff):
                val = getattr(self, name)
                # Only treat Layers/Models as sub-components to register
                if isinstance(val, (nn.LayerBase, ModelBase, list, dict)):
                    self._build_sub(val, name)

            current_keys = new_keys

            if generator is None:
                break

        self.built = True
        
        # Move to device
        self.to(nn.device)

    def get_weights(self):
        if not self.built:
            self.build()
        # PyTorch recursively finds weights
        return list(self.parameters()) + list(self.buffers())

    def get_layer_by_name(self, name):
        return self.layers_by_name.get(name, None)

    def get_layers(self):
        if not self.built:
            self.build()
        layers = []
        for layer in self.layers:
            if isinstance (layer, nn.LayerBase):
                layers.append(layer)
            else:
                layers += layer.get_layers()
        return layers

    def on_build(self, *args, **kwargs):
        """
        init model layers here
        return 'yield' if build is not finished
        """
        pass

    def forward(self, *args, **kwargs):
        # flow layers/models/tensors here
        pass

    def build_for_run(self, shapes_list):
        # Irrelevant in PyTorch, just ensure built
        if not self.built:
            self.build()

    def run(self, inputs):
        # Helper to run model with numpy inputs/outputs
        if not self.built:
            self.build()
            
        # Convert inputs to torch tensors
        t_inputs = []
        for inp in inputs:
            if isinstance(inp, np.ndarray):
                t = torch.from_numpy(inp)
                if t.dtype == torch.float64:
                    t = t.float() # Convert double to float
                t = t.to(next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu')
                t_inputs.append(t)
            else:
                t_inputs.append(inp)
                
        # Run forward
        self.eval() # Set eval mode? Leras run usually meant inference?
        # But 'run' is generic.
        # Assuming inference for 'run'. Train uses update_op.
        
        with torch.no_grad():
            out = self.forward(t_inputs)
            
        # Convert outputs to numpy
        if isinstance(out, (list, tuple)):
            return [ o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in out ]
        else:
            return out.cpu().numpy() if isinstance(out, torch.Tensor) else out

    def summary(self):
        layers = self.get_layers()
        layers_names = []
        layers_params = []

        max_len_str = 0
        max_len_param_str = 0
        delim_str = "-"

        total_params = 0

        #Get layers names and str lenght for delim
        for l in layers:
            s_l = f"{l.__class__.__name__}:{l.name}"
            if len(s_l)>max_len_str:
                max_len_str = len(s_l)
            layers_names+=[s_l]

        #Get params for each layer
        # PyTorch: sum(p.numel())
        layers_params = [ int(np.sum([p.numel() for p in l.parameters()])) for l in layers ]
        total_params = np.sum(layers_params)

        #Get str lenght for delim
        for p in layers_params:
            if len(str(p))>max_len_param_str:
                max_len_param_str=len(str(p))

        #Set delim
        for i in range(max_len_str+max_len_param_str+3):
            delim_str += "-"

        output = "\n"+delim_str+"\n"

        #Format model name str
        model_name_str = "| "+ (self.name.capitalize() if self.name else "Model")
        len_model_name_str = len(model_name_str)
        for i in range(len(delim_str)-len_model_name_str):
            model_name_str+= " " if i!=(len(delim_str)-len_model_name_str-2) else " |"

        output += model_name_str +"\n"
        output += delim_str +"\n"


        #Format layers table
        for i in range(len(layers_names)):
            output += delim_str +"\n"

            l_name = layers_names[i]
            l_param = str(layers_params[i])
            l_param_str = ""
            if len(l_name)<=max_len_str:
                for i in range(max_len_str - len(l_name)):
                    l_name+= " "

            if len(l_param)<=max_len_param_str:
                for i in range(max_len_param_str - len(l_param)):
                    l_param_str+= " "

            l_param_str += l_param


            output +="| "+l_name+"|"+l_param_str+"| \n"

        output += delim_str +"\n"

        #Format sum of params
        total_params_str = "| Total params count: "+str(total_params)
        len_total_params_str = len(total_params_str)
        for i in range(len(delim_str)-len_total_params_str):
            total_params_str+= " " if i!=(len(delim_str)-len_total_params_str-2) else " |"

        output += total_params_str +"\n"
        output += delim_str +"\n"

        io.log_info(output)

nn.ModelBase = ModelBase
