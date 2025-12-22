from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY


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
        difficult_zone = (difficult_zone - dz_mean) * self.dz_amplify  # norm & amplify
        difficult_zone = torch.pow(difficult_zone, 3) + dz_mean * self.dz_amplify  # non-linear transform & recover
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


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV2ep(nn.Module):
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
            activation_type='leaky_relu', norm_type='group', norm_group=channels,
        )
        self.image_conv_block_2 = ResidualConvBlock(
            in_channels=dim_hidden, out_channels=dim_hidden,
            activation_type='silu', norm_type='group', norm_group=channels,
        )
        self.dz_conv_block_1 = ResidualConvBlock(
            in_channels=2 * channels, out_channels=2 * dim_hidden,
            activation_type='leaky_relu', norm_type='group', norm_group=2,
        )
        self.dz_conv_block_2 = ResidualConvBlock(
            in_channels=2 * dim_hidden, out_channels=dim_hidden,
            activation_type='silu', norm_type='group', norm_group=channels,
        )
        self.merge_conv_block_1 = ResidualConvBlock(
            in_channels=dim_hidden * 2, out_channels=dim_hidden * 2,
            activation_type='leaky_relu', norm_type='group', norm_group=2,
        )
        self.merge_conv_block_2 = ResidualConvBlock(
            in_channels=dim_hidden * 2, out_channels=channels,
            activation_type='silu', norm_type='group', norm_group=channels,
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
    def _forward_difficult_zone_features(self, difficult_zone: torch.Tensor):
        # input [B, C, H, W]
        B, C, H, W = difficult_zone.shape
        dz_mean = torch.mean(
            difficult_zone.view(B, C, H * W), dim=-1, keepdim=True
        ).view(B, C, 1, 1)
        x = difficult_zone - dz_mean  # norm & amplify
        x = self.dz_amplify * torch.pow(x, 3)  # non-linear transform
        x = x + torch.pow(dz_mean, 3)  # recover
        dz_feature = self.dz_conv_block_1(torch.cat((difficult_zone, x), dim=1))
        dz_feature = self.dz_conv_block_2(dz_feature)
        return dz_feature  # [B, C', H, W]

    def _forward_one_iter(self, dz_features: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((dz_features, image_features), dim=1)  # [B, 2C', H, W]
        x = self.merge_conv_block_1(x)
        x = self.merge_conv_block_2(x)
        return x  # [B, C, H, W]

    def forward(self, **kwargs) -> torch.Tensor:
        difficult_zone = kwargs.get('difficult_zone')  # [B, C, H, W]
        image_sequence = kwargs.get('images')  # [B, L, C, H, W]

        dz_feature = self._forward_difficult_zone_features(difficult_zone)  # [B, C', H, W]
        image_features = self._forward_image_features(image_sequence)  # [B, L, C', H, W]

        kalman_gains = []
        for i in range(self.num_images):
            gain = self._forward_one_iter(dz_feature, image_features[:, i, ...])
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return kalman_gains  # [B, L, C, H, W]


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV3exp(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            dz_amplify: float = 2,
            dz_scale: float = 0.005,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.dz_amplify = dz_amplify
        self.dz_scale = dz_scale

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
    def _forward_difficult_zone_features(self, difficult_zone: torch.Tensor):
        # input [B, C, H, W]
        B, C, H, W = difficult_zone.shape
        dz_mean = torch.mean(
            difficult_zone.view(B, C, H * W), dim=-1, keepdim=True
        ).view(B, C, 1, 1)
        x = difficult_zone - dz_mean  # norm
        x = self.dz_amplify * torch.tanh(self.dz_scale * x)  # amplify with limitation
        x = torch.exp(x)  # suppression
        dz_feature = self.dz_conv_block(x)
        return dz_feature  # [B, C', H, W]

    def _forward_one_iter(self, dz_features: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((dz_features, image_features), dim=1)  # [B, 2C', H, W]
        x = self.merge_conv_block_1(x)
        x = self.merge_conv_block_2(x)
        return x  # [B, C, H, W]

    def forward(self, **kwargs) -> torch.Tensor:
        difficult_zone = kwargs.get('difficult_zone')  # [B, C, H, W]
        image_sequence = kwargs.get('images')  # [B, L, C, H, W]

        dz_feature = self._forward_difficult_zone_features(difficult_zone)  # [B, C', H, W]
        image_features = self._forward_image_features(image_sequence)  # [B, L, C', H, W]

        kalman_gains = []
        for i in range(self.num_images):
            gain = self._forward_one_iter(dz_feature, image_features[:, i, ...])
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return kalman_gains  # [B, L, C, H, W]


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV4sn(nn.Module):
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
        self.dz_norm = nn.InstanceNorm2d(channels)

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
        self.dz_conv_block_1 = ResidualConvBlock(
            in_channels=channels * 2, out_channels=channels, kernel_size=1,
            activation_type='leaky_relu', norm_type='layer',
            norm_after_conv=True,
        )
        self.dz_conv_block_2 = ResidualConvBlock(
            in_channels=channels, out_channels=dim_hidden,
            activation_type='silu', norm_type='layer',
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
        dz_transformed = self.dz_norm(self.dz_amplify * torch.pow(difficult_zone, 3))
        dz_mean = torch.mean(difficult_zone.view(B, C, H * W), dim=-1, keepdim=True).view(B, C, 1, 1)
        dz_feature = torch.cat((dz_transformed, dz_mean.expand(-1, -1, H, W)), dim=1) # [B, 2C, H, W]
        dz_feature = self.dz_conv_block_1(dz_feature)
        dz_feature = self.dz_conv_block_2(dz_feature)
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


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV4snf(nn.Module):
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
        self.dz_norm = nn.InstanceNorm2d(channels)

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
        self.dz_conv_block_1 = ResidualConvBlock(
            in_channels=channels * 2, out_channels=channels, kernel_size=1,
            activation_type='leaky_relu', norm_type='layer',
            norm_after_conv=True,
        )
        self.dz_conv_block_2 = ResidualConvBlock(
            in_channels=channels, out_channels=dim_hidden,
            activation_type='silu', norm_type='layer',
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

        from basicsr.utils.visualizer import Visualizer
        self.visualizer = Visualizer.instance()

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
        dz_transformed = torch.sigmoid(torch.pow(self.dz_norm(difficult_zone) * self.dz_amplify, 3))

        self.visualizer.visualize(dz_transformed, 'difficult_zone_amplified')

        dz_mean = torch.mean(difficult_zone.view(B, C, H * W), dim=-1, keepdim=True).view(B, C, 1, 1)
        dz_feature = torch.cat((dz_transformed, dz_mean.expand(-1, -1, H, W)), dim=1)  # [B, 2C, H, W]
        dz_feature = self.dz_conv_block_1(dz_feature)
        dz_feature = self.dz_conv_block_2(dz_feature)
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
