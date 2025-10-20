import torch
from torch import nn


class KalmanPredictorV0(nn.Module):
    def __init__(self, dim: int):
        super(KalmanPredictorV0, self).__init__()
        from ..convolutional_res_block import ConvolutionalResBlock
        self.block = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            ConvolutionalResBlock(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

