from typing import List, Literal

import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY


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


from .utils import NormLayerType, ActivationFunction


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV4flexible(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            dz_amplify: float = 2.,
            dz_amplify_method: Literal['cubic', 'exp'] = 'cubic',
            dz_expand: float = 2.,
            img_expand: float = 2.,
            norm_dz_conv_block: NormLayerType = 'instance',
            norm_img_conv_block: NormLayerType = 'instance',
            activation_dz_conv_block: ActivationFunction = 'sigmoid',
            activation_img_conv_block: ActivationFunction = 'sigmoid',
            merge_expand: float = 1.2,
            norm_merge_conv_block: NormLayerType = 'layer',
            activation_merge_conv_block: ActivationFunction = 'silu',
    ):
        from .residual_conv_block import ResidualConvBlock
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.dz_amplify = dz_amplify
        self.dz_expand = dz_expand
        self.img_expand = img_expand
        self.merge_expand = merge_expand

        #################################

        self.norm_img_conv_block = norm_img_conv_block
        self.activation_img_conv_block = activation_img_conv_block

        self.dim_img_features = int(round(self.channels * self.img_expand))
        self.conv_block_img_1 = ResidualConvBlock(
            in_channels=self.channels, out_channels=self.dim_img_features,
            activation_type=self.activation_img_conv_block, norm_type=self.norm_img_conv_block,
        )
        self.conv_block_img_2 = ResidualConvBlock(
            in_channels=self.dim_img_features, out_channels=self.dim_img_features,
            activation_type=self.activation_img_conv_block, norm_type=self.norm_img_conv_block,
        )

        ###########################

        self.norm_dz = nn.InstanceNorm2d(channels)
        self.dz_amplify_method = dz_amplify_method
        assert self.dz_amplify_method in ['cubic', 'exp'], f"unsupported dz_amplify_method: {self.dz_amplify_method}"

        self.norm_dz_conv_block = norm_dz_conv_block
        self.activation_dz_conv_block = activation_dz_conv_block

        self.dim_dz_features = int(round(self.channels * self.dz_expand))
        self.conv_block_dz_1 = ResidualConvBlock(
            in_channels=self.channels * 2, out_channels=self.dim_dz_features,
            activation_type=self.activation_dz_conv_block, norm_type=self.norm_dz_conv_block, norm_after_conv=True,
        )
        self.conv_block_dz_2 = ResidualConvBlock(
            in_channels=self.dim_dz_features, out_channels=self.dim_dz_features,
            activation_type=self.activation_dz_conv_block, norm_type=self.norm_dz_conv_block, norm_after_conv=True,
        )

        ###########################

        self.norm_merge_conv_block = norm_merge_conv_block
        self.activation_merge_conv_block = activation_merge_conv_block

        self.dim_features = self.dim_dz_features + self.dim_img_features
        self.dim_hidden = int(round(self.dim_features * self.merge_expand))
        self.conv_block_merge_1 = ResidualConvBlock(
            in_channels=self.dim_features, out_channels=self.dim_hidden,
            activation_type=self.activation_merge_conv_block, norm_type=self.norm_merge_conv_block,
        )
        self.conv_block_merge_2 = ResidualConvBlock(
            in_channels=self.dim_hidden, out_channels=self.channels,
            activation_type=self.activation_merge_conv_block, norm_type=self.norm_merge_conv_block,
        )

    # noinspection PyPep8Naming
    def _extract_image_features(self, image_sequence: torch.Tensor):
        # input [B, L, C, H, W]
        B, L, C, H, W = image_sequence.shape
        x = image_sequence.view(B * L, C, H, W)
        x = self.conv_block_img_1(x)
        x = self.conv_block_img_2(x)
        image_features = x.view(B, L, self.dim_img_features, H, W)
        # output [B, L, C_img, H, W]
        return image_features

    # noinspection PyPep8Naming
    def _enhance_difficult_zone(self, difficult_zone: torch.Tensor):
        # input [B, C, H, W]
        dz_normalized = self.norm_dz(difficult_zone)

        if self.dz_amplify_method == 'cubic':
            dz_enhanced = torch.pow(dz_normalized * self.dz_amplify, 3)
        elif self.dz_amplify_method == 'exp':
            dz_enhanced = torch.exp(torch.tanh(dz_normalized)) * self.dz_amplify
        else:
            raise NotImplementedError

        dz_all = torch.cat([dz_normalized, dz_enhanced], dim=1)
        return dz_all  # [B, 2C, H, W]

    # noinspection PyPep8Naming
    def _extract_difficult_zone_features(self, difficult_zone: torch.Tensor):
        # input [B, C, H, W]
        enhanced = self._enhance_difficult_zone(difficult_zone)  # [B, 2C, H, W]
        dz_feature = self.conv_block_dz_1(enhanced)
        dz_feature = self.conv_block_dz_2(dz_feature)
        return dz_feature  # [B, C_dz, H, W]

    def _forward_one(self, dz_features: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((dz_features, image_features), dim=1)  # [B, C_dz + C_img, H, W]
        x = self.conv_block_merge_1(x)
        x = self.conv_block_merge_2(x)
        return x  # [B, C, H, W]

    def forward(self, **kwargs) -> torch.Tensor:
        difficult_zone = kwargs.get('difficult_zone')  # [B, C, H, W]
        image_sequence = kwargs.get('images')  # [B, L, C, H, W]

        image_features = self._extract_image_features(image_sequence)  # [B, L, C_img, H, W]
        dz_feature = self._extract_difficult_zone_features(difficult_zone)  # [B, C_dz, H, W]

        kalman_gains = []
        for i in range(self.num_images):
            gain = self._forward_one(dz_feature, image_features[:, i, ...])
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return kalman_gains  # [B, L, C, H, W]


@MODULES_REGISTRY.register()
class KalmanGainCalculatorV5base(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            dz_amplify: float = 2.,
            dz_expand: int = 2,
            img_expand: int = 2,
            merge_expand: int = 2,
    ):
        from .residual_conv_block import ResidualConvBlock
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.img_expand = img_expand
        self.dz_expand = dz_expand
        self.dz_amplify = dz_amplify
        self.dz_norm = nn.InstanceNorm2d(self.channels)
        self.merge_expand = merge_expand

        dim_dz_enhanced = self.channels * 3
        dim_dz_features = self.channels * self.dz_expand
        self.conv_block_dz = ResidualConvBlock(
            in_channels=dim_dz_enhanced, out_channels=dim_dz_features,
            activation_type='sigmoid', norm_type='instance', norm_after_conv=True,
        )

        dim_img_in = self.channels
        dim_img_features = self.channels * self.img_expand
        self.conv_block_img_1 = ResidualConvBlock(
            in_channels=dim_img_in, out_channels=dim_img_features,
            activation_type='leaky_relu', norm_type='instance',
        )
        self.conv_block_img_2 = ResidualConvBlock(
            in_channels=dim_img_features, out_channels=dim_img_features,
            activation_type='silu', norm_type='instance',
        )

        dim_in = dim_dz_features + dim_img_features
        dim_hidden = dim_in * self.merge_expand
        self.conv_block_merge_1 = ResidualConvBlock(
            in_channels=dim_in, out_channels=dim_hidden,
            activation_type='leaky_relu', norm_type='instance',
        )
        self.conv_block_merge_2 = ResidualConvBlock(
            in_channels=dim_hidden, out_channels=self.channels,
            activation_type='tanh', norm_type='instance',
        )

    # noinspection PyPep8Naming
    def _enhance_difficult_zone(self, difficult_zone: torch.Tensor):
        # input [B, C, H, W]
        normalized = self.dz_norm(difficult_zone)
        dz_enhanced_pow = torch.pow(normalized * self.dz_amplify, 3)
        dz_enhanced_exp = torch.exp(torch.tanh(normalized) * 2)
        enhanced = torch.concat([difficult_zone, dz_enhanced_exp, dz_enhanced_pow], dim=1)
        return enhanced  # [B, 3C, H, W]

    # noinspection PyPep8Naming
    def _extract_image_features(self, image_sequence: torch.Tensor):
        # input [B, L, C, H, W]
        B, L, C, H, W = image_sequence.shape
        x = image_sequence.view(B * L, C, H, W)
        x = self.conv_block_img_1(x)
        x = self.conv_block_img_2(x)
        image_features = x.view(B, L, self.channels * self.img_expand, H, W)
        # output [B, L, C_img, H, W]
        return image_features

    def _forward_one(self, dz_features: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((dz_features, image_features), dim=1)  # [B, C_dz + C_img, H, W]
        x = self.conv_block_merge_1(x)
        x = self.conv_block_merge_2(x)
        return x  # [B, C, H, W]

    def forward(self, **kwargs) -> torch.Tensor:
        difficult_zone = kwargs.get('difficult_zone')  # [B, C, H, W]
        image_sequence = kwargs.get('images')  # [B, L, C, H, W]

        dz_enhanced = self._enhance_difficult_zone(difficult_zone)  # [B, 3C, H, W]
        dz_features = self.conv_block_dz(dz_enhanced)  # [B, C_dz, H, W]

        image_features = self._extract_image_features(image_sequence)  # [B, L, C_img, H, W]

        kalman_gains = []
        for i in range(self.num_images):
            gain = self._forward_one(dz_features, image_features[:, i, ...])
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return kalman_gains  # [B, L, C, H, W]
