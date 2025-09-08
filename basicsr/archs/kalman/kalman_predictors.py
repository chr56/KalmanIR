import torch
from torch import nn as nn


def build_predictor(mode, dim, seq_length) -> nn.Module:
    if mode == "convolutional":
        return KalmanPredictorV0(dim=dim)
    elif mode == "mamba_adjustment":
        return KalmanPredictorMambaAdjustment(dim=dim)
    elif mode == "mamba_latent_adjustment":
        return KalmanPredictorMambaLatentAdjustment(dim=dim)
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


class KalmanPredictorMambaAdjustment(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super(KalmanPredictorMambaAdjustment, self).__init__()

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma = SS2DChanelFirst(d_model=dim, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.mamba_sigma(x)
        bias = self.mamba_bias(x)
        x = (x / torch.exp(-sigma)) + bias
        return x


class KalmanPredictorMambaLatentAdjustment(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 36, **kwargs):
        super(KalmanPredictorMambaLatentAdjustment, self).__init__()
        self.hidden_dim = hidden_dim

        from .utils import LayerNorm2d
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding='same'),
            nn.LeakyReLU(),
            LayerNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )
        self.proj_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, padding='same')

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma = SS2DChanelFirst(d_model=hidden_dim, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=hidden_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj_in(x)
        sigma = self.mamba_sigma(z)
        bias = self.mamba_bias(z)
        z = (z / torch.exp(-sigma)) + bias
        x = self.proj_out(z)
        return x
