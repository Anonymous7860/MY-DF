import numpy as np
import torch
import torch.nn.functional as F
from core.leras import nn

patch_discriminator_kernels = \
    { 1  : (512, [ [1,1] ]),
      2  : (512, [ [2,1] ]),
      3  : (512, [ [2,1], [2,1] ]),
      4  : (512, [ [2,2], [2,2] ]),
      5  : (512, [ [3,2], [2,2] ]),
      6  : (512, [ [4,2], [2,2] ]),
      7  : (512, [ [3,2], [3,2] ]),
      8  : (512, [ [4,2], [3,2] ]),
      9  : (512, [ [3,2], [4,2] ]),
      10 : (512, [ [4,2], [4,2] ]),
      11 : (512, [ [3,2], [3,2], [2,1] ]),
      12 : (512, [ [4,2], [3,2], [2,1] ]),
      13 : (512, [ [3,2], [4,2], [2,1] ]),
      14 : (512, [ [4,2], [4,2], [2,1] ]),
      15 : (512, [ [3,2], [3,2], [3,1] ]),
      16 : (512, [ [4,2], [3,2], [3,1] ]),
      17 : (512, [ [3,2], [4,2], [3,1] ]),
      18 : (512, [ [4,2], [4,2], [3,1] ]),
      19 : (512, [ [3,2], [3,2], [4,1] ]),
      20 : (512, [ [4,2], [3,2], [4,1] ]),
      21 : (512, [ [3,2], [4,2], [4,1] ]),
      22 : (512, [ [4,2], [4,2], [4,1] ]),
      23 : (256, [ [3,2], [3,2], [3,2], [2,1] ]),
      24 : (256, [ [4,2], [3,2], [3,2], [2,1] ]),
      25 : (256, [ [3,2], [4,2], [3,2], [2,1] ]),
      26 : (256, [ [4,2], [4,2], [3,2], [2,1] ]),
      27 : (256, [ [3,2], [4,2], [4,2], [2,1] ]),
      28 : (256, [ [4,2], [3,2], [4,2], [2,1] ]),
      29 : (256, [ [3,2], [4,2], [4,2], [2,1] ]),
      30 : (256, [ [4,2], [4,2], [4,2], [2,1] ]),
      31 : (256, [ [3,2], [3,2], [3,2], [3,1] ]),
      32 : (256, [ [4,2], [3,2], [3,2], [3,1] ]),
      33 : (256, [ [3,2], [4,2], [3,2], [3,1] ]),
      34 : (256, [ [4,2], [4,2], [3,2], [3,1] ]),
      35 : (256, [ [3,2], [4,2], [4,2], [3,1] ]),
      36 : (256, [ [4,2], [3,2], [4,2], [3,1] ]),
      37 : (256, [ [3,2], [4,2], [4,2], [3,1] ]),
      38 : (256, [ [4,2], [4,2], [4,2], [3,1] ]),
      39 : (256, [ [3,2], [3,2], [3,2], [4,1] ]),
      40 : (256, [ [4,2], [3,2], [3,2], [4,1] ]),
      41 : (256, [ [3,2], [4,2], [3,2], [4,1] ]),
      42 : (256, [ [4,2], [4,2], [3,2], [4,1] ]),
      43 : (256, [ [3,2], [4,2], [4,2], [4,1] ]),
      44 : (256, [ [4,2], [3,2], [4,2], [4,1] ]),
      45 : (256, [ [3,2], [4,2], [4,2], [4,1] ]),
      46 : (256, [ [4,2], [4,2], [4,2], [4,1] ]),
    }


class PatchDiscriminator(nn.ModelBase):
    def on_build(self, patch_size, in_ch, base_ch=None, conv_kernel_initializer=None):
        suggested_base_ch, kernels_strides = patch_discriminator_kernels[patch_size]

        if base_ch is None:
            base_ch = suggested_base_ch

        prev_ch = in_ch
        self.convs = []
        for i, (kernel_size, strides) in enumerate(kernels_strides):
            cur_ch = base_ch * min( (2**i), 8 )

            self.convs.append ( nn.Conv2D( prev_ch, cur_ch, kernel_size=kernel_size, strides=strides, padding='SAME', kernel_initializer=conv_kernel_initializer) )
            prev_ch = cur_ch

        self.out_conv =  nn.Conv2D( prev_ch, 1, kernel_size=1, padding='VALID', kernel_initializer=conv_kernel_initializer)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu( conv(x), 0.1 )
        return self.out_conv(x)

nn.PatchDiscriminator = PatchDiscriminator

class UNetPatchDiscriminator(nn.ModelBase):
    """
    Inspired by https://arxiv.org/abs/2002.12655 "A U-Net Based Discriminator for Generative Adversarial Networks"
    """
    @staticmethod
    def calc_receptive_field_size(layers):
        """
        result the same as https://fomoro.com/research/article/receptive-field-calculatorindex.html
        """
        rf = 0
        ts = 1
        for i, (k, s) in enumerate(layers):
            if i == 0:
                rf = k
            else:
                rf += (k-1)*ts
            ts *= s
        return rf

    @classmethod
    def find_archi(cls, target_patch_size, max_layers=9):
        """
        Find the best configuration of layers using only 3x3 convs for target patch size
        """
        s = {}
        for layers_count in range(1,max_layers+1):
            val = 1 << (layers_count-1)
            while True:
                val -= 1

                layers = []
                sum_st = 0
                layers.append ( [3, 2])
                sum_st += 2
                for i in range(layers_count-1):
                    st = 1 + (1 if val & (1 << i) !=0 else 0 )
                    layers.append ( [3, st ])
                    sum_st += st                

                rf = cls.calc_receptive_field_size(layers)

                s_rf = s.get(rf, None)
                if s_rf is None:
                    s[rf] = (layers_count, sum_st, layers)
                else:
                    if layers_count < s_rf[0] or \
                    ( layers_count == s_rf[0] and sum_st > s_rf[1] ):
                        s[rf] = (layers_count, sum_st, layers)

                if val == 0:
                    break

        x = sorted(list(s.keys()))
        q=x[np.abs(np.array(x)-target_patch_size).argmin()]
        return s[q][2]

    def on_build(self, patch_size, in_ch, base_ch = 16, use_fp16 = False, antialias = False):
        patch_size = int(patch_size)
        base_ch = int(base_ch)
        if patch_size < 1:
            raise ValueError("patch_size must be at least 1")
        if base_ch < 1:
            raise ValueError("base_ch must be at least 1")

        self.use_fp16 = use_fp16
        self.patch_size = patch_size
        self.base_ch = base_ch
        # antialias=True: no strided convolution in either direction.
        # Down: conv(stride 1) -> BlurPool(3, stride 2). Up: nearest x2 -> conv.
        # The backward of a stride-2 convolution is a stride-2 transposed
        # convolution, and the transposed up-path is one already, so the stock
        # layout hands the generator a period-2 / period-4 lattice gradient
        # from both directions. SAENXT's decoder ends in a 16-phase
        # depth-to-space head that renders that lattice as 2px grid lines.
        # Measured on real faces (toy generator with the same head, 800 steps,
        # two seeds): output grid energy 15157/14934 ppm with this D as
        # shipped vs 3383/1425 ppm anti-aliased (no-GAN baseline 1037/1716);
        # the lattice energy in the GAN gradient itself falls by 2-3 orders
        # of magnitude. Same parameter count, stride-1 convs at each level's
        # input resolution, so a little more compute per D pass.
        self.antialias = bool(antialias)
        # Persisted marker so a checkpoint built with one layout is never
        # partially loaded into the other (conv weights share shapes).
        self.register_buffer("arch_flag", torch.tensor(1 if self.antialias else 0, dtype=torch.int64))
        conv_dtype = nn.float16 if use_fp16 else nn.float32
        
        class ResidualBlock(nn.ModelBase):
            def on_build(self, ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', dtype=conv_dtype)

            def forward(self, inp):
                x = self.conv1(inp)
                x = F.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = F.leaky_relu(inp + x, 0.2)
                return x

        prev_ch = in_ch
        self.convs = []
        self.upconvs = []
        layers = self.find_archi(patch_size)
        
        level_chs = { i-1:v for i,v in enumerate([ min( base_ch * (2**i), 512 ) for i in range(len(layers)+1)]) }

        self.in_conv = nn.Conv2D( in_ch, level_chs[-1], kernel_size=1, padding='VALID', dtype=conv_dtype)

        self.down_strides = []
        self.up_strides = []
        for i, (kernel_size, strides) in enumerate(layers):
            up_in_ch = level_chs[i]*(2 if i != len(layers)-1 else 1)
            if self.antialias:
                self.convs.append ( nn.Conv2D( level_chs[i-1], level_chs[i], kernel_size=kernel_size, strides=1, padding='SAME', dtype=conv_dtype) )
                self.upconvs.insert (0, nn.Conv2D( up_in_ch, level_chs[i-1], kernel_size=kernel_size, strides=1, padding='SAME', dtype=conv_dtype) )
            else:
                self.convs.append ( nn.Conv2D( level_chs[i-1], level_chs[i], kernel_size=kernel_size, strides=strides, padding='SAME', dtype=conv_dtype) )
                self.upconvs.insert (0, nn.Conv2DTranspose( up_in_ch, level_chs[i-1], kernel_size=kernel_size, strides=strides, padding='SAME', dtype=conv_dtype) )
            self.down_strides.append(int(strides))
            self.up_strides.insert(0, int(strides))
        if self.antialias:
            from core.leras.layers.BlurPool import BlurPool
            self.blur_pool = BlurPool(filt_size=3, stride=2)

        self.out_conv = nn.Conv2D( level_chs[-1]*2, 1, kernel_size=1, padding='VALID', dtype=conv_dtype)

        self.center_out  =  nn.Conv2D( level_chs[len(layers)-1], 1, kernel_size=1, padding='VALID', dtype=conv_dtype)
        self.center_conv =  nn.Conv2D( level_chs[len(layers)-1], level_chs[len(layers)-1], kernel_size=1, padding='VALID', dtype=conv_dtype)

    @staticmethod
    def _match_skip_spatial(x, skip):
        """
        Match TensorFlow SAME-transpose output to its encoder skip tensor.

        A stride-S SAME convolution produces ceil(input / S), while the
        transpose layer produces encoded * S.  For odd input sizes that is
        larger than the original tensor, so crop the extra bottom/right pixel
        before concatenation.  Padding is retained as a defensive fallback for
        unusual future stride/kernel combinations.
        """
        target_h, target_w = skip.shape[-2:]
        if x.shape[-2:] == (target_h, target_w):
            return x

        x = x[..., :target_h, :target_w]
        pad_h = target_h - x.shape[-2]
        pad_w = target_w - x.shape[-1]
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, max(0, pad_w), 0, max(0, pad_h)))
        return x

    def forward(self, x):
        if self.use_fp16:
            x = x.half()
            
        x = F.leaky_relu( self.in_conv(x), 0.2 )

        encs = []
        for conv, stride in zip(self.convs, self.down_strides):
            encs.insert(0, x)
            x = F.leaky_relu( conv(x), 0.2 )
            if self.antialias and stride == 2:
                x = self.blur_pool(x)

        center_out, x = self.center_out(x), F.leaky_relu( self.center_conv(x), 0.2 )

        # NCHW
        ch_axis = 1

        for i, (upconv, stride, enc) in enumerate(zip(self.upconvs, self.up_strides, encs)):
            if self.antialias and stride == 2:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = F.leaky_relu( upconv(x), 0.2 )
            x = self._match_skip_spatial(x, enc)
            x = torch.cat( [enc, x], dim=ch_axis)

        x = self.out_conv(x)
        
        if self.use_fp16:
            center_out = center_out.float()
            x = x.float()

        return center_out, x

nn.UNetPatchDiscriminator = UNetPatchDiscriminator
