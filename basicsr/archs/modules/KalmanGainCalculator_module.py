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


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV2e(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            dz_amplify: float = 2,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.dz_amplify = dz_amplify

        from .residual_conv_block import ResidualConvBlock
        dim_hidden = channels * 3
        self.image_conv_block_1 = ResidualConvBlock(
            in_channels=channels, out_channels=dim_hidden,
            activation_type='leaky_relu', norm_type='layer'
        )
        self.image_conv_block_2 = ResidualConvBlock(
            in_channels=dim_hidden, out_channels=dim_hidden,
            activation_type='silu', norm_type='group', norm_group=3
        )
        self.dz_conv_block = ResidualConvBlock(
            in_channels=channels, out_channels=dim_hidden,
            activation_type='leaky_relu', norm_type='layer',
            norm_after_conv=True,
        )
        self.merge_conv_block_1 = ResidualConvBlock(
            in_channels=dim_hidden * 2, out_channels=dim_hidden * 2,
            norm_type='group', norm_group=2, activation_type='silu'
        )
        self.merge_conv_block_2 = ResidualConvBlock(
            in_channels=dim_hidden * 2, out_channels=channels,
            norm_type='group', norm_group=channels, activation_type='silu'
        )
        self.dim_hidden = dim_hidden

    # noinspection PyPep8Naming
    def _forward_image_features(self, image_sequence: torch.Tensor):
        # input [B, L, C, H, W]
        B, L, C, H, W = image_sequence.shape
        x = image_sequence.view(B * L, C, H, W)
        x = self.image_conv_block_1(x)
        x = self.image_conv_block_2(x)
        image_features = x.view(B, L, self.dim_hidden, H, W)
        # output [B, L, C', H, W]
        return image_features

    # noinspection PyPep8Naming
    def _forward_difficult_zone(self, difficult_zone: torch.Tensor):
        # input [B, C, H, W]
        B, C, H, W = difficult_zone.shape
        dz_mean = torch.mean(
            difficult_zone.view(B, C, H * W),
            dim=-1, keepdim=True
        ).view(B, C, 1, 1)
        difficult_zone = (difficult_zone - dz_mean) * self.dz_amplify # norm & amplify
        difficult_zone = torch.pow(difficult_zone, 3) + dz_mean * self.dz_amplify # non-linear transform & recover
        dz_feature = self.dz_conv_block(difficult_zone)
        return dz_feature  # [B, C', H, W]

    def _forward_one_iter(self, dz_features: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((dz_features, image_features), dim=1)  # [B, 2C', H, W]
        x = self.merge_conv_block_1(x)
        x = self.merge_conv_block_2(x)
        return x  # [B, C, H, W]

    def forward(self, **kwargs) -> torch.Tensor:
        difficult_zone = kwargs.get('difficult_zone')  # [B, C, H, W]
        image_sequence = kwargs.get('images')  # [B, L, C, H, W]

        dz_feature = self._forward_difficult_zone(difficult_zone)  # [B, C', H, W]
        image_features = self._forward_image_features(image_sequence)  # [B, L, C', H, W]

        kalman_gains = []
        for i in range(self.num_images):
            gain = self._forward_one_iter(dz_feature, image_features[:, i, ...])
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return kalman_gains  # [B, L, C, H, W]
