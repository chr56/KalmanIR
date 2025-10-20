import torch
from torch import nn


class KalmanPredictorDeepConvolutionalV1(nn.Module):
    def __init__(self, dim: int):
        super(KalmanPredictorDeepConvolutionalV1, self).__init__()
        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, padding='same', stride=1)
        self.block1 = ConvolutionalResBlockGroupNorm(dim, 3, activation_type='leaky_relu')
        self.block2 = ConvolutionalResBlockGroupNorm(dim, 3, activation_type='silu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class KalmanPredictorDeepConvolutionalV2(nn.Module):
    def __init__(self, dim: int):
        super(KalmanPredictorDeepConvolutionalV2, self).__init__()
        from ..convolutional_res_block import ResidualConvBlock
        self.block1 = ResidualConvBlock(
            dim, dim, num_layers=1,
            norm_type='group', norm_group=3, activation_type='silu',
        )
        self.block2 = ResidualConvBlock(
            dim, dim, num_layers=3,
            norm_type='group', norm_group=3, activation_type='sigmoid',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x
