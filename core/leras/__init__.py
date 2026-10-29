from .nn import nn

# Import Ops first as they attach themselves to nn
from . import ops

# Layers
from .layers.LayerBase import LayerBase
nn.LayerBase = LayerBase

from .layers.Saveable import Saveable
nn.Saveable = Saveable

from .layers.Conv2D import Conv2D
nn.Conv2D = Conv2D

from .layers.Conv2DTranspose import Conv2DTranspose
nn.Conv2DTranspose = Conv2DTranspose

from .layers.DepthwiseConv2D import DepthwiseConv2D
nn.DepthwiseConv2D = DepthwiseConv2D

from .layers.Dense import Dense
nn.Dense = Dense

from .layers.DenseNorm import DenseNorm
nn.DenseNorm = DenseNorm

from .layers.BatchNorm2D import BatchNorm2D
nn.BatchNorm2D = BatchNorm2D

from .layers.InstanceNorm2D import InstanceNorm2D
nn.InstanceNorm2D = InstanceNorm2D

from .layers.FRNorm2D import FRNorm2D
nn.FRNorm2D = FRNorm2D

from .layers.AdaIN import AdaIN
nn.AdaIN = AdaIN

from .layers.BlurPool import BlurPool
nn.BlurPool = BlurPool

from .layers.TLU import TLU
nn.TLU = TLU

from .layers.ScaleAdd import ScaleAdd
nn.ScaleAdd = ScaleAdd

from .layers.TanhPolar import TanhPolar
nn.TanhPolar = TanhPolar

from .layers.MsSsim import MsSsim
nn.MsSsim = MsSsim

# Initializers
from .initializers import CA
nn.CAInitializer = CA.CAInitializer

# Optimizers
from .optimizers.OptimizerBase import OptimizerBase
nn.OptimizerBase = OptimizerBase

from .optimizers.optimizer_factory import OptimizerFactory, OptimizerRegistry
nn.OptimizerFactory = OptimizerFactory
nn.OptimizerRegistry = OptimizerRegistry

from .optimizers.RMSprop import RMSprop
nn.RMSprop = RMSprop

from .optimizers.AdaBelief import AdaBelief
nn.AdaBelief = AdaBelief

# Models & Archis
from .archis.DeepFakeArchi import DeepFakeArchi
nn.DeepFakeArchi = DeepFakeArchi

from .models.CodeDiscriminator import CodeDiscriminator
nn.CodeDiscriminator = CodeDiscriminator

from .models.PatchDiscriminator import PatchDiscriminator, UNetPatchDiscriminator
nn.PatchDiscriminator = PatchDiscriminator
nn.UNetPatchDiscriminator = UNetPatchDiscriminator

from .models.XSeg import XSeg
nn.XSeg = XSeg
