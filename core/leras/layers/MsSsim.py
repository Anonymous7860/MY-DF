import torch
import torch.nn.functional as F
import math
from core.leras import nn

class MsSsim(nn.LayerBase):
    default_power_factors = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    default_l1_alpha = 0.84

    def __init__(self, batch_size, in_ch, resolution, kernel_size=11, use_l1=False, **kwargs):
        # restrict mssim factors to those greater/equal to kernel size
        power_factors = [p for i, p in enumerate(self.default_power_factors) if resolution//(2**i) >= kernel_size]
        # normalize power factors if reduced because of size
        if sum(power_factors) < 1.0:
            power_factors = [x/sum(power_factors) for x in power_factors]
        self.power_factors = power_factors
        self.num_scale = len(power_factors)
        self.kernel_size = kernel_size
        self.use_l1 = use_l1
        
        # We need a gaussian kernel for SSIM.
        # Window generation:
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
        gauss = gauss / gauss.sum()
        self.window_1d = gauss
        
        # We will create 2D window on the fly or buffer it if channels fixed.
        # But channels passed in init.
        # shape: (in_ch, 1, k, k)
        _1d = gauss.unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d.expand(in_ch, 1, kernel_size, kernel_size).contiguous()
        
        self.register_buffer('window', window)
        
        super().__init__(**kwargs)

    def forward(self, y_true, y_pred, max_val=1.0):
        if nn.data_format == "NHWC":
            y_true = y_true.permute(0, 3, 1, 2)
            y_pred = y_pred.permute(0, 3, 1, 2)

        # Keep originals for L1
        if self.use_l1:
            y_true_orig = y_true
            y_pred_orig = y_pred

        # MS-SSIM calculation
        window = self.window
        
        msssim = []
        mcs = []
        
        # Iterate downscales
        curr_true = y_true
        curr_pred = y_pred
        
        for i in range(self.num_scale):
            # SSIM at this scale
            mu1 = F.conv2d(curr_pred, window, padding=self.kernel_size//2, groups=curr_pred.shape[1])
            mu2 = F.conv2d(curr_true, window, padding=self.kernel_size//2, groups=curr_true.shape[1])

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(curr_pred * curr_pred, window, padding=self.kernel_size//2, groups=curr_pred.shape[1]) - mu1_sq
            sigma2_sq = F.conv2d(curr_true * curr_true, window, padding=self.kernel_size//2, groups=curr_true.shape[1]) - mu2_sq
            sigma12 = F.conv2d(curr_pred * curr_true, window, padding=self.kernel_size//2, groups=curr_true.shape[1]) - mu1_mu2

            C1 = (0.01 * max_val) ** 2
            C2 = (0.03 * max_val) ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            
            msssim.append(ssim_map.mean())
            mcs.append(cs_map.mean())
            
            # Downsample for next scale
            if i < self.num_scale - 1:
                curr_pred = F.avg_pool2d(curr_pred, 2)
                curr_true = F.avg_pool2d(curr_true, 2)

        # Combine
        weights = torch.tensor(self.power_factors).to(y_true.device)
        
        term_prod = 1.0
        for i in range(self.num_scale - 1):
            term_prod = term_prod * (mcs[i] ** weights[i])
            
        term_prod = term_prod * (msssim[-1] ** weights[-1])
        ms_ssim_val = term_prod
        ms_ssim_loss = 1.0 - ms_ssim_val
        
        if self.use_l1:
            # Simple L1 loss on original inputs
            l1_loss = F.l1_loss(y_true_orig, y_pred_orig)
            return self.default_l1_alpha * ms_ssim_loss + (1 - self.default_l1_alpha) * l1_loss

        return ms_ssim_loss
        
    def __call__(self, y_true, y_pred, max_val=1.0):
        # Allow calling with max_val argument
        return self.forward(y_true, y_pred, max_val=max_val)

nn.MsSsim = MsSsim
