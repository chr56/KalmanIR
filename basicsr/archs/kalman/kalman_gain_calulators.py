import torch
from torch import nn


class KalmanGainCalculatorV0(nn.Module):
    def __init__(self, dim: int):
        super(KalmanGainCalculatorV0, self).__init__()
        from .convolutional_res_block import ConvolutionalResBlock
        self.block = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
