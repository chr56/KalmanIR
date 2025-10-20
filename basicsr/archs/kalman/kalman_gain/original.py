import torch
from torch import nn as nn


class KalmanGainCalculatorV0(nn.Module):
    def __init__(self, dim: int):
        super(KalmanGainCalculatorV0, self).__init__()
        from ..convolutional_res_block import ConvolutionalResBlock
        self.block = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LinearConvolutionalMultipleChannels(nn.Module):
    def __init__(self, channel: int, seq_length: int):
        super(LinearConvolutionalMultipleChannels, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        self.merger = nn.Sequential(
            nn.GroupNorm(2, 2 * channel, eps=1e-6),
            nn.Conv2d(2 * channel, channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )
        self.residual_rate = 0.4

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image = image_sequence[:, i, ...]
            gain = self.merger(torch.cat((uncertainty, image), dim=1))  # [B, 2C, H, W] -> [B, C, H, W]
            gain = self.residual_rate * uncertainty + (1 - self.residual_rate) * gain
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class SimpleConvolutionalMultipleChannels(nn.Module):
    def __init__(self, channel: int, seq_length: int):
        super(SimpleConvolutionalMultipleChannels, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        self.merger = nn.Sequential(
            nn.GroupNorm(2, 2 * channel, eps=1e-6),
            nn.Conv2d(2 * channel, 2 * channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(2 * channel, channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.GroupNorm(1, channel, eps=1e-6),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
        )
        self.residual_rate = 0.4

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image = image_sequence[:, i, ...]
            gain = self.merger(torch.cat((uncertainty, image), dim=1))  # [B, 2C, H, W] -> [B, C, H, W]
            gain = self.residual_rate * uncertainty + (1 - self.residual_rate) * gain
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class ComplexConvolutionalMultipleChannels(nn.Module):
    def __init__(self, channel: int, seq_length: int):
        super(ComplexConvolutionalMultipleChannels, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..utils import LayerNorm2d
        self.proj_uncertainty = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
            LayerNorm2d(channel),
        )
        self.proj_image = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
            LayerNorm2d(channel),
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(2 * channel, 3 * channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            LayerNorm2d(3 * channel),
            nn.Conv2d(3 * channel, channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
        )

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        u = self.proj_uncertainty(uncertainty)  # [B, C, H, W]
        kalman_gains = []
        for i in range(self.seq_length):
            image = self.proj_image(image_sequence[:, i, ...])  # [B, C, H, W]
            gain = self.block_1(torch.cat((u, image), dim=1))  # 2 * [B, C, H, W] -> [B, C, H, W]
            gain = self.block_2(torch.cat((gain, u), dim=1))  # 2 * [B, C, H, W] -> [B, C, H, W]
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains
