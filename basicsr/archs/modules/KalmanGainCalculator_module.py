from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV1(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
    ):
        super(KalmanGainCalculatorV1, self).__init__()

        self.channels = channels
        self.num_images = num_images

        from .residual_conv_block import ResidualConvBlock
        self.image_input_block = ResidualConvBlock(
            channels, norm_type='group', norm_group=3, activation_type='leaky_relu'
        )
        self.image_features_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    channels,
                    norm_type='group', norm_group=3,
                    num_layers=3, activation_type=['leaky_relu', 'leaky_relu', 'silu']
                )
                for _ in range(num_images)
            ]
        )
        self.merge_block = ResidualConvBlock(
            in_channels=channels * 2, out_channels=channels,
            norm_type='group', norm_group=[6, 3], activation_type='silu'
        )

    def _forward_one_image(self, difficult_zone: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((difficult_zone, image_features), dim=1)  # [B, 2C, H, W]
        x = self.merge_block(x)  # [B, 2C, H, W] -> [B, C, H, W]
        return x

    def forward(self, **kwargs) -> torch.Tensor:
        difficult_zone = kwargs.get('difficult_zone')  # [B, C, H, W]
        image_sequence = kwargs.get('images')  # n * [B, C, H, W]

        kalman_gains = []
        for i in range(self.num_images):
            image_features = self.image_input_block(image_sequence[:, i, ...])
            image_features = self.image_features_blocks[i](image_features)
            gain = self._forward_one_image(difficult_zone, image_features)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return kalman_gains  # [B, L, C, H, W]
