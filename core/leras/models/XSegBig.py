import torch

from core.leras import nn


class XSegBig(nn.ModelBase):
    def on_build(self, in_ch, base_ch, out_ch, resolution):
        self.base_ch = base_ch

        class SEBlock(nn.ModelBase):
            def on_build(self, channels, se_ratio=8):
                reduced_channels = max(1, channels // se_ratio)
                self.dense1 = nn.Dense(channels, reduced_channels)
                self.dense2 = nn.Dense(reduced_channels, channels)

            def forward(self, x):
                se = torch.mean(x, dim=[2, 3])
                se = self.dense1(se)
                se = torch.relu(se)
                se = self.dense2(se)
                se = torch.sigmoid(se)
                se = se.view(se.shape[0], se.shape[1], 1, 1)
                return x * se

        class DownConvBlockSE(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=3, padding='SAME', strides=2, dilations=1):
                self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=kernel_size, padding=padding, strides=strides, dilations=dilations)
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)
                self.se_block = SEBlock(out_ch, se_ratio=8)
                self.shortcut = nn.Conv2D(in_ch, out_ch, kernel_size=1, strides=strides, padding=padding)

            def forward(self, x):
                residual = x
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                x = self.se_block(x)
                x = x + self.shortcut(residual)
                x = self.tlu(x)
                return x

        class ConvBlockSE(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=3, padding='SAME', strides=1, dilations=1):
                self.conv = nn.Conv2D(in_ch, out_ch, kernel_size=kernel_size, padding=padding, strides=strides, dilations=dilations)
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)
                self.se_block = SEBlock(out_ch, se_ratio=8)

            def forward(self, x):
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                x = self.se_block(x)
                return x

        class UpConvBlockSE(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=3, padding='SAME'):
                self.conv = nn.Conv2DTranspose(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
                self.frn = nn.FRNorm2D(out_ch)
                self.tlu = nn.TLU(out_ch)
                self.se_block = SEBlock(out_ch, se_ratio=8)

            def forward(self, x):
                x = self.conv(x)
                x = self.frn(x)
                x = self.tlu(x)
                x = self.se_block(x)
                return x

        class DenseBlock(nn.ModelBase):
            def on_build(self, in_c, out_ch):
                self.dense = nn.Dense(in_c, out_ch)

            def forward(self, x):
                return self.dense(x)

        self.in_conv = DownConvBlockSE(in_ch, base_ch, strides=1)
        self.conv01 = DownConvBlockSE(base_ch, base_ch, strides=1)
        self.conv11 = DownConvBlockSE(base_ch, base_ch * 2, strides=2)
        self.conv21 = DownConvBlockSE(base_ch * 2, base_ch * 4, strides=2)
        self.conv31 = DownConvBlockSE(base_ch * 4, base_ch * 8, strides=2)
        self.conv41 = DownConvBlockSE(base_ch * 8, base_ch * 8, strides=2)
        self.conv51 = DownConvBlockSE(base_ch * 8, base_ch * 8, strides=2)
        self.conv61 = DownConvBlockSE(base_ch * 8, base_ch * 8, strides=2)

        self.dense1 = DenseBlock(16 * base_ch * 8, 8 * base_ch * 8)
        self.dense2 = DenseBlock(8 * base_ch * 8, 16 * base_ch * 8)

        self.uconv61 = ConvBlockSE(base_ch * 16, base_ch * 8)

        self.up5 = UpConvBlockSE(base_ch * 8, base_ch * 8)
        self.uconv51 = ConvBlockSE(base_ch * 16, base_ch * 8)

        self.up4 = UpConvBlockSE(base_ch * 8, base_ch * 8)
        self.uconv41 = ConvBlockSE(base_ch * 16, base_ch * 8)

        self.up3 = UpConvBlockSE(base_ch * 8, base_ch * 8)
        self.uconv31 = ConvBlockSE(base_ch * 16, base_ch * 8)

        self.up2 = UpConvBlockSE(base_ch * 8, base_ch * 4)
        self.uconv21 = ConvBlockSE(base_ch * 8, base_ch * 4)

        self.up1 = UpConvBlockSE(base_ch * 4, base_ch * 2)
        self.uconv11 = ConvBlockSE(base_ch * 4, base_ch * 2)

        self.up0 = UpConvBlockSE(base_ch * 2, base_ch)
        self.uconv01 = ConvBlockSE(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2D(base_ch, out_ch, kernel_size=3, padding='SAME')

    def forward(self, inp, pretrain=False):
        ch_axis = 1
        x = inp

        x = self.in_conv(x)

        x0 = self.conv01(x)
        x1 = self.conv11(x0)
        x2 = self.conv21(x1)
        x3 = self.conv31(x2)
        x4 = self.conv41(x3)
        x5 = self.conv51(x4)
        x = x6 = self.conv61(x5)

        x = nn.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = nn.reshape_4D(x, 4, 4, self.base_ch * 8)

        if pretrain:
            x6 = torch.zeros_like(x6)
        x = self.uconv61(torch.cat([x, x6], dim=ch_axis))

        x = self.up5(x)
        if pretrain:
            x5 = torch.zeros_like(x5)
        x = self.uconv51(torch.cat([x, x5], dim=ch_axis))

        x = self.up4(x)
        if pretrain:
            x4 = torch.zeros_like(x4)
        x = self.uconv41(torch.cat([x, x4], dim=ch_axis))

        x = self.up3(x)
        if pretrain:
            x3 = torch.zeros_like(x3)
        x = self.uconv31(torch.cat([x, x3], dim=ch_axis))

        x = self.up2(x)
        if pretrain:
            x2 = torch.zeros_like(x2)
        x = self.uconv21(torch.cat([x, x2], dim=ch_axis))

        x = self.up1(x)
        if pretrain:
            x1 = torch.zeros_like(x1)
        x = self.uconv11(torch.cat([x, x1], dim=ch_axis))

        x = self.up0(x)
        if pretrain:
            x0 = torch.zeros_like(x0)
        x = self.uconv01(torch.cat([x, x0], dim=ch_axis))

        logits = self.out_conv(x)
        return logits, torch.sigmoid(logits)


nn.XSegBig = XSegBig
