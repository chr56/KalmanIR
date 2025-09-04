import torch
from torch import nn as nn


def build_predictor(mode, dim, seq_length) -> nn.Module:
    if mode == "convolutional":
        return KalmanPredictorV0(dim=dim)
    else:
        if mode:
            import warnings
            warnings.warn(f"Unknown kalman preditor mode `{mode}`, using default!")
        return KalmanPredictorV0(dim=dim)


class KalmanPredictorV0(nn.Module):
    def __init__(self, dim: int):
        super(KalmanPredictorV0, self).__init__()
        from .convolutional_res_block import ConvolutionalResBlock
        self.block = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            ConvolutionalResBlock(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
