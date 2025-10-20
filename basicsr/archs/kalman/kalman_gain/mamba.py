import torch
from torch import nn as nn


class KalmanGainCalculatorMambaSimple(nn.Module):
    def __init__(self, dim: int):
        super(KalmanGainCalculatorMambaSimple, self).__init__()
        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.ss2d = SS2DChanelFirst(dim)
        self.projector = nn.Conv2d(dim, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ss2d(x)
        x = self.projector(x)
        return nn.functional.sigmoid(x)


class KalmanGainCalculatorMambaBlock(nn.Module):
    def __init__(self, dim: int):
        super(KalmanGainCalculatorMambaBlock, self).__init__()
        from basicsr.archs.modules_mamba import SS2D, Mlp, VSSBlockFabric
        self.mamba = VSSBlockFabric(
            dim=dim,
            ssm_block=SS2D(d_model=dim),
            mlp_block=Mlp(dim),
            post_norm=False,
        )
        self.projector = nn.Conv2d(dim, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
        x = self.mamba(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
        x = self.projector(x)
        return nn.functional.sigmoid(x)
