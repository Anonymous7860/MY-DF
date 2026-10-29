import torch
from pathlib import Path
from core import pathex
import numpy as np
from core.leras import nn


def _squeezed_shape(shape):
    """Return a shape with singleton dimensions removed, preserving order."""
    return tuple(int(dim) for dim in shape if int(dim) != 1)


class Saveable(object):
    def __init__(self, name=None):
        self.name = name

    def get_weights(self):
        # In PyTorch, we rely on state_dict or parameters()
        # This method might be used by legacy code for inspection.
        if isinstance(self, torch.nn.Module):
            return list(self.parameters()) + list(self.buffers())
        return []

    def get_weights_np(self):
        # Return list of numpy arrays
        return [p.detach().cpu().numpy() for p in self.get_weights()]

    def set_weights(self, new_weights):
        # new_weights: list of numpy arrays or tensors
        current_params = self.get_weights()
        if len(current_params) != len(new_weights):
             raise ValueError (f'len of lists mismatch {len(current_params)} != {len(new_weights)}')

        converted_weights = []
        for index, (param, new_weight) in enumerate(zip(current_params, new_weights)):
            tensor = (
                torch.from_numpy(new_weight)
                if isinstance(new_weight, np.ndarray)
                else torch.as_tensor(new_weight)
            )

            # Convert only layouts whose TensorFlow and PyTorch ordering is known.
            if param.shape != tensor.shape:
                if param.ndim == 4 and tensor.ndim == 4:
                    # Conv2D: TF (H,W,I,O) -> PyTorch (O,I,H,W).
                    if param.shape == (
                        tensor.shape[3],
                        tensor.shape[2],
                        tensor.shape[0],
                        tensor.shape[1],
                    ):
                        print(f"Converting Conv2D: {tensor.shape} -> {param.shape}")
                        tensor = tensor.permute(3, 2, 0, 1)
                    # Depthwise: TF (H,W,I,M) -> PyTorch (I*M,1,H,W).
                    elif (
                        param.numel() == tensor.numel()
                        and param.shape[1] == 1
                        and param.shape[2] == tensor.shape[0]
                        and param.shape[3] == tensor.shape[1]
                        and param.shape[0] == tensor.shape[2] * tensor.shape[3]
                    ):
                        print(f"Converting Depthwise: {tensor.shape} -> {param.shape}")
                        tensor = tensor.permute(2, 3, 0, 1).reshape(param.shape)
                elif (
                    param.ndim == 2
                    and tensor.ndim == 2
                    and param.shape == (tensor.shape[1], tensor.shape[0])
                ):
                    # Dense: TF (I,O) -> PyTorch (O,I).
                    print(f"Converting Dense: {tensor.shape} -> {param.shape}")
                    tensor = tensor.permute(1, 0)

            if param.shape != tensor.shape:
                if _squeezed_shape(param.shape) == _squeezed_shape(tensor.shape):
                    tensor = tensor.reshape(param.shape)
                else:
                    raise ValueError(
                        "unsafe weight shape mismatch at index "
                        f"{index}: target={tuple(param.shape)} source={tuple(tensor.shape)}"
                    )

            converted_weights.append(tensor)

        # Validate every tensor before mutating the module so a later mismatch
        # cannot leave a partially converted checkpoint in memory.
        with torch.no_grad():
            for param, tensor in zip(current_params, converted_weights):
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))

    def save_weights(self, filename, force_dtype=None):
        if self.name is None:
             raise Exception("name must be defined.")
             
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self, torch.nn.Module):
             state = self.state_dict()
             torch.save(state, str(path))
        else:
             pass

    def load_weights(self, filename):
        filepath = Path(filename)
        if not filepath.exists():
            return False
            
        try:
            if isinstance(self, torch.nn.Module):
                state = torch.load(str(filepath), map_location='cpu', weights_only=False)
                if not isinstance(state, dict):
                    raise ValueError(
                        f"Expected a PyTorch state dict, got {type(state).__name__}"
                    )

                model_state = self.state_dict()
                compatible_state = {}
                unexpected_keys = []
                shape_mismatches = []
                for key, value in state.items():
                    target = model_state.get(key)
                    if target is None:
                        unexpected_keys.append(key)
                    elif not isinstance(value, torch.Tensor):
                        shape_mismatches.append(key)
                    elif tuple(value.shape) != tuple(target.shape):
                        shape_mismatches.append(key)
                    else:
                        compatible_state[key] = value

                if model_state and not compatible_state:
                    raise ValueError(
                        f"No compatible tensors found in PyTorch checkpoint {filepath.name}"
                    )

                keys = self.load_state_dict(compatible_state, strict=False)
                if len(keys.missing_keys) > 0:
                     print(f"Warning: {len(keys.missing_keys)} missing keys in {filepath.name}! (Model might be initialized randomly)")
                     print(f"Example missing: {keys.missing_keys[:5]}")
                if unexpected_keys:
                     print(f"Warning: {len(unexpected_keys)} unexpected keys in pretrained file!")
                if shape_mismatches:
                     print(f"Warning: {len(shape_mismatches)} incompatible tensor shapes kept fresh!")

                print(
                    f"Successfully loaded PyTorch weights: {filepath.name} "
                    f"({len(compatible_state)}/{len(model_state)} tensors matched)"
                )
                return True
        except Exception as e_torch:

            msg = str(e_torch)
            if "No compatible tensors found in PyTorch checkpoint" in msg:
                print(f"Warning: {msg}")
                return False
            if "Invalid magic number" in msg or "unpickling" in msg.lower():
                print(f"Legacy file detected: {filepath.name} - Converting options/weights...")
            else:
                print(f"Warning: torch.load failed for {filename}: {e_torch}")
            # Try loading as legacy TF pickle or Numpy
            try:
                import pickle
                weights_np = None
                try:
                    with open(filepath, 'rb') as f:
                        weights_np = pickle.load(f)
                except Exception:
                    # Try np.load
                    try:
                        weights_np = np.load(filepath, allow_pickle=True)
                    except Exception:
                        # Third-party forks (e.g. ICE) save with joblib.
                        try:
                            import joblib
                            weights_np = joblib.load(filepath)
                        except Exception:
                            pass
                
                if weights_np is not None:
                    if isinstance(weights_np, dict):
                         # Legacy DFL: dict of { name: weight_array }
                         # We need to map PyTorch param names to TF names.
                         # PyTorch: layer.sublayer.weight
                         # TF: layer/sublayer/weight:0
                         
                         
                         ordered_weights = []
                         missing_count = 0
                         mapped_count = 0
                         
                         # Get all parameters/buffers for this module
                         params_map = dict()
                         for name, param in self.named_parameters():
                             params_map[name] = param
                         for name, buf in self.named_buffers():
                             params_map[name] = buf
                         
                         for name, param in params_map.items():
                             # Name resolution: 'layers.0.layers.0.weight' -> 'down1/conv1/weight'
                             parts = name.split('.')
                             resolved_parts = []
                             
                             current_mod = self
                             i = 0
                             while i < len(parts):
                                 part = parts[i]
                                 
                                 # specific logic regarding ModelBase layers
                                 if part == 'layers' and i+1 < len(parts) and parts[i+1].isdigit():
                                     idx = int(parts[i+1])
                                     if hasattr(current_mod, 'layers') and idx < len(current_mod.layers):
                                         child = current_mod.layers[idx]
                                         if hasattr(child, 'name') and child.name:
                                             resolved_parts.append(child.name)
                                         else:
                                              resolved_parts.append(f"layers/{idx}")
                                         current_mod = child
                                         i += 2 # Skip 'layers' and '0'
                                         continue

                                 # Default traversal
                                 resolved_parts.append(part)
                                 
                                 # Advance current_mod if possible, for next iteration checking
                                 # Note: 'weight' or 'bias' are terminal, usually tensor, not module
                                 if hasattr(current_mod, part):
                                      attr = getattr(current_mod, part)
                                      if isinstance(attr, torch.nn.Module):
                                          current_mod = attr
                                 
                                 i += 1
                                 
                             tf_base_name = '/'.join(resolved_parts)
                             tf_name_candidates = [tf_base_name + ':0']

                             # Add synonyms for TF compatibility
                             last_part = parts[-1]
                             if last_part == 'weight':
                                  tf_name_candidates.append( '/'.join(resolved_parts[:-1] + ['gamma:0']) )
                                  tf_name_candidates.append( '/'.join(resolved_parts[:-1] + ['kernel:0']) )
                             elif last_part == 'bias':
                                  tf_name_candidates.append( '/'.join(resolved_parts[:-1] + ['beta:0']) )
                             elif last_part == 'running_mean':
                                  tf_name_candidates.append( '/'.join(resolved_parts[:-1] + ['moving_mean:0']) )
                             elif last_part == 'running_var':
                                  tf_name_candidates.append( '/'.join(resolved_parts[:-1] + ['moving_variance:0']) )

                             found_weight = None
                             for tf_name in tf_name_candidates:
                                 if tf_name in weights_np:
                                     found_weight = weights_np[tf_name]
                                     break
                             
                             if found_weight is not None:
                                 ordered_weights.append(found_weight)
                                 mapped_count += 1
                             else:
                                  # Search by the primary name without its tensor suffix.
                                  primary_candidate = tf_name_candidates[0]
                                  search_key = primary_candidate.replace(':0','')
                                  
                                  found = False
                                  for k in weights_np.keys():
                                      if search_key in k:
                                           ordered_weights.append(weights_np[k])
                                           found = True
                                           mapped_count += 1
                                           break
                                  
                                  if not found:
                                      print(f"Warning: Could not find legacy weight for {name} (Tried: {tf_name_candidates})")
                                      missing_count += 1
                                      ordered_weights.append(param.detach().cpu().numpy())

                         if params_map and mapped_count == 0:
                             raise ValueError(
                                 f"No compatible legacy tensors found in {filepath.name}"
                             )

                         weights_np = ordered_weights
                         print(
                             f"Mapped {mapped_count}/{len(weights_np)} weights from dict "
                             f"(Missing: {missing_count})."
                         )

                    if isinstance(weights_np, list):
                        print(f"Loading legacy TF/Numpy weights: {filepath.name} (Count: {len(weights_np)})")
                        self.set_weights(weights_np)
                        
                        # Save converted weights in PyTorch format for faster future loads
                        print(f"Caching converted weights to PyTorch format: {filepath.name}")
                        try:
                            self.save_weights(filepath)
                            print(f"Successfully cached PyTorch weights. Future loads will be instant!")
                        except Exception as e_save:
                            print(f"Warning: Could not cache converted weights: {e_save}")
                        
                        return True
                    else:
                        print(f"Failed to load weights {filename}: Content is not list/ndarray/dict-of-list. Got {type(weights_np)}")
                else:
                    print(f"Failed to load weights {filename}: Torch load failed ({e_torch}) and generic load return None/Invalid.")
            except Exception as e_leg:
                print(f"Failed to load weights from {filename}: Torch error: {e_torch}, Legacy error: {e_leg}")
                return False
        return False

    def init_weights(self):
        # PyTorch layers are usually initialized on creation.
        pass

nn.Saveable = Saveable
