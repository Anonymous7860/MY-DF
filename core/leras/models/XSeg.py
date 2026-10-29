import torch
from core.leras import nn

class XSeg(nn.ModelBase):
    
    def on_build (self, in_ch, base_ch, out_ch, resolution=256):
        
        class ConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):              
                self.conv = nn.Conv2D (in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):                
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x

        class UpConvBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch):
                self.conv = nn.Conv2DTranspose (in_ch, out_ch, kernel_size=3, padding='SAME')
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)

            def forward(self, x):
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                return x
                
        self.base_ch = base_ch
        self.resolution = int(resolution) if resolution is not None else 256
        # 256 keeps the classic six-stage stack. 512 models (e.g. the ICE
        # high-res XSeg) add a seventh stage so the bottleneck is still 4x4,
        # widen the last two stages to base_ch*16, and use a 384-wide dense.
        self.deep = self.resolution >= 512
        self.bottleneck_ch = base_ch * (16 if self.deep else 8)
        self.dense_ch = 384 if self.deep else 512

        self.conv01 = ConvBlock(in_ch, base_ch)
        self.conv02 = ConvBlock(base_ch, base_ch)
        self.bp0 = nn.BlurPool (filt_size=4)

        self.conv11 = ConvBlock(base_ch, base_ch*2)
        self.conv12 = ConvBlock(base_ch*2, base_ch*2)
        self.bp1 = nn.BlurPool (filt_size=3)

        self.conv21 = ConvBlock(base_ch*2, base_ch*4)
        self.conv22 = ConvBlock(base_ch*4, base_ch*4)
        self.bp2 = nn.BlurPool (filt_size=2)

        self.conv31 = ConvBlock(base_ch*4, base_ch*8)
        self.conv32 = ConvBlock(base_ch*8, base_ch*8)
        self.conv33 = ConvBlock(base_ch*8, base_ch*8)
        self.bp3 = nn.BlurPool (filt_size=2)

        self.conv41 = ConvBlock(base_ch*8, base_ch*8)
        self.conv42 = ConvBlock(base_ch*8, base_ch*8)
        self.conv43 = ConvBlock(base_ch*8, base_ch*8)
        self.bp4 = nn.BlurPool (filt_size=2)
        
        c5 = base_ch*16 if self.deep else base_ch*8
        self.conv51 = ConvBlock(base_ch*8, c5)
        self.conv52 = ConvBlock(c5, c5)
        self.conv53 = ConvBlock(c5, c5)
        self.bp5 = nn.BlurPool (filt_size=2)
        
        if self.deep:
            self.conv61 = ConvBlock(base_ch*16, base_ch*16)
            self.conv62 = ConvBlock(base_ch*16, base_ch*16)
            self.conv63 = ConvBlock(base_ch*16, base_ch*16)
            self.bp6 = nn.BlurPool (filt_size=2)

        self.dense1 = nn.Dense ( 4*4* self.bottleneck_ch, self.dense_ch)
        self.dense2 = nn.Dense ( self.dense_ch, 4*4* self.bottleneck_ch)

        if self.deep:
            self.up6 = UpConvBlock (base_ch*16, base_ch*8)
            self.uconv63 = ConvBlock(base_ch*24, base_ch*16)
            self.uconv62 = ConvBlock(base_ch*16, base_ch*16)
            self.uconv61 = ConvBlock(base_ch*16, base_ch*16)
                
        if self.deep:
            self.up5 = UpConvBlock (base_ch*16, base_ch*8)
            self.uconv53 = ConvBlock(base_ch*24, base_ch*16)
            self.uconv52 = ConvBlock(base_ch*16, base_ch*16)
            self.uconv51 = ConvBlock(base_ch*16, base_ch*8)
        else:
            self.up5 = UpConvBlock (base_ch*8, base_ch*4)
            self.uconv53 = ConvBlock(base_ch*12, base_ch*8)
            self.uconv52 = ConvBlock(base_ch*8, base_ch*8)
            self.uconv51 = ConvBlock(base_ch*8, base_ch*8)
        
        self.up4 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv43 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv42 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv41 = ConvBlock(base_ch*8, base_ch*8)

        self.up3 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv33 = ConvBlock(base_ch*12, base_ch*8)
        self.uconv32 = ConvBlock(base_ch*8, base_ch*8)
        self.uconv31 = ConvBlock(base_ch*8, base_ch*8)

        self.up2 = UpConvBlock (base_ch*8, base_ch*4)
        self.uconv22 = ConvBlock(base_ch*8, base_ch*4)
        self.uconv21 = ConvBlock(base_ch*4, base_ch*4)

        self.up1 = UpConvBlock (base_ch*4, base_ch*2)
        self.uconv12 = ConvBlock(base_ch*4, base_ch*2)
        self.uconv11 = ConvBlock(base_ch*2, base_ch*2)

        self.up0 = UpConvBlock (base_ch*2, base_ch)
        self.uconv02 = ConvBlock(base_ch*2, base_ch)
        self.uconv01 = ConvBlock(base_ch, base_ch)
        self.out_conv = nn.Conv2D (base_ch, out_ch, kernel_size=3, padding='SAME')
    
        
    def forward(self, inp, pretrain=False):
        ch_axis = 1  # Channel axis for NCHW format in PyTorch
        x = inp


        x = self.conv01(x)
        x = x0 = self.conv02(x)
        x = self.bp0(x)

        x = self.conv11(x)
        x = x1 = self.conv12(x)
        x = self.bp1(x)

        x = self.conv21(x)
        x = x2 = self.conv22(x)
        x = self.bp2(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = x3 = self.conv33(x)
        x = self.bp3(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = x4 = self.conv43(x)
        x = self.bp4(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = x5 = self.conv53(x)
        x = self.bp5(x)

        x6 = None
        if self.deep:
            x = self.conv61(x)
            x = self.conv62(x)
            x = x6 = self.conv63(x)
            x = self.bp6(x)

        # Flatten
        # x shape NCHW
        x = nn.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = nn.reshape_4D (x, 4, 4, self.bottleneck_ch )

        if self.deep:
            x = self.up6(x)
            if pretrain:
                x6 = torch.zeros_like(x6)
            x = self.uconv63(torch.cat([x,x6], dim=ch_axis))
            x = self.uconv62(x)
            x = self.uconv61(x)

        x = self.up5(x)
        if pretrain:
            x5 = torch.zeros_like(x5)
        x = self.uconv53(torch.cat([x,x5], dim=ch_axis))
        x = self.uconv52(x)
        x = self.uconv51(x)
        
        x = self.up4(x)
        if pretrain:
            x4 = torch.zeros_like(x4)
        x = self.uconv43(torch.cat([x,x4], dim=ch_axis))
        x = self.uconv42(x)
        x = self.uconv41(x)

        x = self.up3(x)
        if pretrain:
            x3 = torch.zeros_like(x3)
        x = self.uconv33(torch.cat([x,x3], dim=ch_axis))
        x = self.uconv32(x)
        x = self.uconv31(x)

        x = self.up2(x)
        if pretrain:
            x2 = torch.zeros_like(x2)
        x = self.uconv22(torch.cat([x,x2], dim=ch_axis))
        x = self.uconv21(x)

        x = self.up1(x)
        if pretrain:
            x1 = torch.zeros_like(x1)
        x = self.uconv12(torch.cat([x,x1], dim=ch_axis))
        x = self.uconv11(x)

        x = self.up0(x)
        if pretrain:
            x0 = torch.zeros_like(x0)
        x = self.uconv02(torch.cat([x,x0], dim=ch_axis))
        x = self.uconv01(x)

        logits = self.out_conv(x)
        return logits, torch.sigmoid(logits)
        
nn.XSeg = XSeg