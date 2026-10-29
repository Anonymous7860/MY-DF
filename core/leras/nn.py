"""
Leras.

like lighter keras.
This is my lightweight neural network library written from scratch.
Ported to PyTorch.

Provides:
+ full freedom of pytorch operations
+ convenient and understandable logic
"""

import os
import sys
import warnings
import logging
from pathlib import Path
import numpy as np
import torch
from core.interact import interact as io
from .device import Devices

class nn():
    current_DeviceConfig = None

    device = None
    data_format = None
    conv2d_ch_axis = None
    conv2d_spatial_axes = None
    
    floatx = None
    float32 = torch.float32
    float16 = torch.float16

    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NCHW"):
        if nn.device is None:
            if device_config is None:
                device_config = nn.getCurrentDeviceConfig()
            nn.setCurrentDeviceConfig(device_config)

            if len(device_config.devices) == 0:
                nn.device = torch.device('cpu')
            else:
                # Use the first selected device. Multi-GPU logic might need more work later (e.g. DataParallel)
                d = device_config.devices[0]
                nn.device = torch.device(f"cuda:{d.index}")
                torch.cuda.set_device(nn.device)

            # PyTorch defaults to NCHW mostly, but we support setting it.
            # However, PyTorch layers (Conv2d) expect NCHW.
            nn.set_data_format(data_format)
            
            if floatx == "float32":
                nn.floatx = torch.float32
                torch.set_default_dtype(torch.float32)
            elif floatx == "float16":
                nn.floatx = torch.float16
                torch.set_default_dtype(torch.float16)
            else:
                raise ValueError(f"unsupported floatx {floatx}")

    @staticmethod
    def initialize_main_env():
        Devices.initialize_main_env()

    @staticmethod
    def set_floatx(torch_dtype):
        nn.floatx = torch_dtype
        torch.set_default_dtype(torch_dtype)

    @staticmethod
    def set_data_format(data_format):
        # We strongly encourage NCHW for PyTorch
        if data_format != "NHWC" and data_format != "NCHW":
            raise ValueError(f"unsupported data_format {data_format}")
        nn.data_format = data_format

        if data_format == "NHWC":
            nn.conv2d_ch_axis = 3
            nn.conv2d_spatial_axes = [1,2]
        elif data_format == "NCHW":
            nn.conv2d_ch_axis = 1
            nn.conv2d_spatial_axes = [2,3]

    @staticmethod
    def get4Dshape ( w, h, c ):
        if nn.data_format == "NHWC":
            return (None,h,w,c)
        else:
            return (None,c,h,w)

    @staticmethod
    def flatten(x):
        """Flatten tensor from [B, C, H, W] to [B, C*H*W] (PyTorch port of tf flatten)."""
        return x.reshape(x.shape[0], -1)

    @staticmethod
    def reshape_4D(x, h, w, c):
        """Reshape flat tensor back to [B, C, H, W] (NCHW format)."""
        return x.reshape(x.shape[0], c, h, w)

    @staticmethod
    def to_data_format( x, to_data_format, from_data_format):
        if to_data_format == from_data_format:
            return x
        
        # x is numpy or tensor? Assuming tensor for PyTorch flow, but usually this helper was for numpy in TF code sometimes.
        # Let's handle torch tensors.
        if isinstance(x, torch.Tensor):
            if to_data_format == "NHWC":
                return x.permute(0, 2, 3, 1)
            elif to_data_format == "NCHW":
                return x.permute(0, 3, 1, 2)
        else:
            # Numpy fallback
            if to_data_format == "NHWC":
                return np.transpose(x, (0,2,3,1) )
            elif to_data_format == "NCHW":
                return np.transpose(x, (0,3,1,2) )
        
        raise ValueError(f"unsupported to_data_format {to_data_format}")

    @staticmethod
    def getCurrentDeviceConfig():
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = nn.DeviceConfig.BestGPU()
        return nn.current_DeviceConfig

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        nn.current_DeviceConfig = device_config

    @staticmethod
    def reset_session():
        pass # Not applicable for PyTorch

    @staticmethod
    def close_session():
        pass # Not applicable for PyTorch

    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
        devices = Devices.getDevices()
        if len(devices) == 0:
            unsupported = os.environ.get('NN_UNSUPPORTED_CUDA_DEVICES', '')
            if unsupported:
                io.log_info('CUDA device is not supported by this bundled runtime:')
                for device_name in unsupported.split('|'):
                    io.log_info(f'  {device_name}')
                io.log_info('Falling back to CPU extraction.')
            return []

        all_devices_indexes = [device.index for device in devices]

        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_gpu = False

        if suggest_all_gpu:
            best_device_indexes = all_devices_indexes
        elif suggest_best_multi_gpu:
            best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device()) ]
        else:
            best_device_indexes = [ devices.get_best_device().index ]
        best_device_indexes = ",".join([str(x) for x in best_device_indexes])

        io.log_info ("")
        if choose_only_one:
            io.log_info ("Choose one GPU idx.")
        else:
            io.log_info ("Choose one or several GPU idxs (separated by comma).")
        io.log_info ("")

        if allow_cpu:
            io.log_info ("[CPU] : CPU")
        for device in devices:
            io.log_info (f"  [{device.index}] : {device.name}")

        io.log_info ("")

        while True:
            try:
                if choose_only_one:
                    choosed_idxs = io.input_str("Which GPU index to choose?", best_device_indexes)
                else:
                    choosed_idxs = io.input_str("Which GPU indexes to choose?", best_device_indexes)

                if allow_cpu and choosed_idxs.lower() == "cpu":
                    choosed_idxs = []
                    break

                choosed_idxs = [ int(x) for x in choosed_idxs.split(',') ]

                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break
                else:
                    if all( [idx in all_devices_indexes for idx in choosed_idxs] ):
                        break
            except:
                pass
        io.log_info ("")

        return choosed_idxs

    class DeviceConfig():
        @staticmethod
        def ask_choose_device(*args, **kwargs):
            return nn.DeviceConfig.GPUIndexes( nn.ask_choose_device_idxs(*args,**kwargs) )

        def __init__ (self, devices=None):
            devices = devices or []

            if not isinstance(devices, Devices):
                devices = Devices(devices)

            self.devices = devices
            self.cpu_only = len(devices) == 0

            # PyTorch device translation
            if self.cpu_only:
                self.torch_device = torch.device('cpu')
            else:
                self.torch_device = torch.device(f'cuda:{devices[0].index}')


        @staticmethod
        def BestGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()
            return nn.DeviceConfig([devices.get_best_device()])

        @staticmethod
        def WorstGPU():
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()
            return nn.DeviceConfig([devices.get_worst_device()])

        @staticmethod
        def GPUIndexes(indexes):
            if len(indexes) != 0:
                devices = Devices.getDevices().get_devices_from_index_list(indexes)
            else:
                devices = []
            return nn.DeviceConfig(devices)

        @staticmethod
        def CPU():
            return nn.DeviceConfig([])
