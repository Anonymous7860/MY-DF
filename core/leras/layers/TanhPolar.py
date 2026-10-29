import numpy as np
import torch
import torch.nn.functional as F
from core.leras import nn

class TanhPolar(nn.LayerBase):
    """
    RoI Tanh-polar Transformer Network for Face Parsing in the Wild
    https://github.com/hhj1897/roi_tanh_warping
    """

    def __init__(self, width, height, angular_offset_deg=270, **kwargs):
        self.width = width
        self.height = height

        warp_gridx, warp_gridy = TanhPolar._get_tanh_polar_warp_grids(width,height,angular_offset_deg=angular_offset_deg)
        restore_gridx, restore_gridy = TanhPolar._get_tanh_polar_restore_grids(width,height,angular_offset_deg=angular_offset_deg)

        # Registers as buffers (constant state)
        # Store as (1, H, W)
        self.register_buffer('warp_gridx', torch.from_numpy(warp_gridx).unsqueeze(0))
        self.register_buffer('warp_gridy', torch.from_numpy(warp_gridy).unsqueeze(0))
        self.register_buffer('restore_gridx', torch.from_numpy(restore_gridx).unsqueeze(0))
        self.register_buffer('restore_gridy', torch.from_numpy(restore_gridy).unsqueeze(0))

        super().__init__(**kwargs)

    def warp(self, inp):
        return self._apply_grid(inp, self.warp_gridx, self.warp_gridy)

    def restore(self, inp):
        # Restore logic in Leras had symmetric padding?
        # inp_t = tf.pad(inp_t, [(0,0), (1, 1), (1, 0), (0, 0)], "SYMMETRIC")
         # NCHW: (0,0, 0,0, 1,1, 1,0)? Wait TF pad is per dimension.
         # TF Input: N, H, W, C likely?
         # Leras code:
         # if data_format == "NCHW": inp_t = transpose...
         # pad: dim 0 (N): 0,0. Dim 1 (H): 1,1. Dim 2 (W): 1,0. Dim 3 (C): 0,0.
         # So padding H top/bottom by 1. W left by 1, right by 0.
         # PyTorch pad: (left, right, top, bottom).
         # So pad=(1, 0, 1, 1).
         
        if nn.data_format == "NHWC":
            inp = inp.permute(0, 3, 1, 2)
        
        # Apply strict symmetric padding as in original logic
        inp = F.pad(inp, (1, 0, 1, 1), mode='reflect') # SYMMETRIC -> reflect? or replicate? TF SYMMETRIC is reflect-ish.
        # TF SYMMETRIC: Reflects but includes edge. PyTorch 'reflect' excludes edge, 'replicate' repeats edge.
        # Usually 'reflect' is closer to what we want for images.
        
        # After padding, shape changes?
        # The restoration grid likely expects cached coordinates that map to this padded space or output space?
        # Leras restore grid generation depended only on W, H (original).
        # The sampler samples FROM inp using grid.
        # If inp is padded, the grid coordinates might need adjustment or they are already in "padded space"?
        # Actually Leras doesn't adjust grid. It pads input.
        # This implies the grid coordinates fall outside original bounds slightly?
        
        out = self._apply_grid(inp, self.restore_gridx, self.restore_gridy)
        
        if nn.data_format == "NHWC":
            out = out.permute(0, 2, 3, 1)
            
        return out

    def _apply_grid(self, inp, gridx, gridy):
        # input is NCHW (converted if needed before call or here)
        # gridx, gridy are 1HW.
        
        if nn.data_format == "NHWC" and inp.shape[-1] == self.in_ch: # Checking if not permuted yet
             inp = inp.permute(0, 3, 1, 2)
        
        N, C, H_in, W_in = inp.shape
        # grid shape broadcasting
        gridx = gridx.expand(N, -1, -1)
        gridy = gridy.expand(N, -1, -1)
        
        # Create grid (N, H_out, W_out, 2)
        # Check alignment: gridx is width coordinates?
        # Leras numpy returns src_x_indices.
        
        # PyTorch grid_sample expects [-1, 1].
        # We need to normalize pixel coordinates to [-1, 1].
        # Norm = 2 * (x / (W-1)) - 1
        
        gx = 2.0 * gridx / (max(W_in - 1, 1)) - 1.0
        gy = 2.0 * gridy / (max(H_in - 1, 1)) - 1.0
        
        # Stack to (N, H, W, 2)
        grid = torch.stack((gx, gy), dim=3)
        
        out = F.grid_sample(inp, grid, align_corners=True) # Check align_corners behavior vs TF
        
        if nn.data_format == "NHWC":
            out = out.permute(0, 2, 3, 1)
            
        return out

    def get_weights(self):
        return []
        
    def build_weights(self):
        pass

    @staticmethod
    def _get_tanh_polar_warp_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)
        normalised_dest_indices = np.stack(np.meshgrid(np.arange(0.0, 1.0, 1.0 / W),np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / H)), axis=-1)
        radii = normalised_dest_indices[..., 0]
        orientation_x = np.cos(normalised_dest_indices[..., 1])
        orientation_y = np.sin(normalised_dest_indices[..., 1])

        src_radii = np.arctanh(radii) * (roi_radii[0] * roi_radii[1] / np.sqrt(roi_radii[1] ** 2 * orientation_x ** 2 + roi_radii[0] ** 2 * orientation_y ** 2))
        src_x_indices = src_radii * orientation_x
        src_y_indices = src_radii * orientation_y
        src_x_indices, src_y_indices = (roi_center[0] + cos_offset * src_x_indices - sin_offset * src_y_indices,
                                        roi_center[1] + cos_offset * src_y_indices + sin_offset * src_x_indices)

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

    @staticmethod
    def _get_tanh_polar_restore_grids(W,H,angular_offset_deg):
        angular_offset_pi = angular_offset_deg * np.pi / 180.0

        roi_center = np.array([ W//2, H//2], np.float32 )
        roi_radii = np.array([W, H], np.float32 ) / np.pi ** 0.5
        cos_offset, sin_offset = np.cos(angular_offset_pi), np.sin(angular_offset_pi)

        dest_indices = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(float)
        normalised_dest_indices = np.matmul(dest_indices - roi_center, np.array([[cos_offset, -sin_offset],
                                                                                [sin_offset, cos_offset]]))
        radii = np.linalg.norm(normalised_dest_indices, axis=-1)
        normalised_dest_indices[..., 0] /= np.clip(radii, 1e-9, None)
        normalised_dest_indices[..., 1] /= np.clip(radii, 1e-9, None)
        radii *= np.sqrt(roi_radii[1] ** 2 * normalised_dest_indices[..., 0] ** 2 +
                        roi_radii[0] ** 2 * normalised_dest_indices[..., 1] ** 2) / roi_radii[0] / roi_radii[1]

        src_radii = np.tanh(radii)


        src_x_indices = src_radii * W + 1.0
        src_y_indices = np.mod((np.arctan2(normalised_dest_indices[..., 1], normalised_dest_indices[..., 0]) /
                                2.0 / np.pi) * H, H) + 1.0

        return src_x_indices.astype(np.float32), src_y_indices.astype(np.float32)

nn.TanhPolar = TanhPolar