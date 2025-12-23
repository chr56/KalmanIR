from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY
from .utils_calculation import cal_ae, cal_bce, cal_bce_sigmoid, cal_cs, cal_kl

def _get_transform_layer(final_transform):
    if final_transform == 'tanh':
        return nn.Tanh()
    elif final_transform == 'sigmoid':
        return nn.Sigmoid()
    elif final_transform == 'none' or final_transform == '':
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unsupported {final_transform}")


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV4plain(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            merge_ratio: float = 1.0,
            final_transform: str = 'none',
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.merge_ratio = merge_ratio

        self.final_transform = _get_transform_layer(final_transform)

        from .residual_conv_block import ResidualConvBlock
        diff_metric_methods = 2
        ch_layer_0 = channels * num_images * diff_metric_methods
        ch_layer_1 = channels * num_images
        ch_layer_2 = channels
        self.conv_block_0 = ResidualConvBlock(
            in_channels=ch_layer_0, out_channels=ch_layer_0, num_layers=3,
            activation_type='silu', norm_type='group', norm_group=num_images * diff_metric_methods,
        )
        self.conv_block_1 = ResidualConvBlock(
            in_channels=ch_layer_0, out_channels=ch_layer_1, num_layers=2,
            activation_type='silu', norm_type='layer',
        )
        self.conv_block_2 = ResidualConvBlock(
            in_channels=ch_layer_1, out_channels=ch_layer_2, num_layers=2,
            activation_type='sigmoid', norm_type='layer',
        )

    def _calculate_difference(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param images: sequence of images [B, L, C, H, W]
        :return: tuple of difference tensors [B, L, C, H, W]
        """

        images_shifted = torch.roll(images, shifts=1, dims=1)  # [B, L, C, H, W] (shifted in L)

        ae_diff = cal_ae(images, images_shifted)  # [B, L, C, H, W]
        bce_diff = cal_bce_sigmoid(images, images_shifted)  # [B, L, C, H, W]

        return ae_diff, bce_diff

    def _forward_differences(self, ae_differences: torch.Tensor, bce_differences: torch.Tensor):
        """
        :param ae_differences: [B, L, C, H, W]
        :param bce_differences: [B, L, C, H, W]
        :return: dz prediction [B, C, H, W]
        """
        B, L, C, H, W = ae_differences.shape

        ae_cat = ae_differences.reshape(B, C * L, H, W)
        bce_cat = bce_differences.reshape(B, C * L, H, W)

        differences = torch.cat((ae_cat, bce_cat), dim=1)

        x = differences  # [B, 2 * L * C, H, W]
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: image sequence, list of [B, C, H, W] tensors
        :return: Difficult Zone, shape [B, C, H, W]
        """

        # L * [B, C, H, W], L * [B, C, H, W]
        delta_ae, delta_bce = self._calculate_difference(images)

        dz_branch_avg = torch.mean(delta_ae, dim=1)  # [B, C, H, W]
        dz_branch_dl = self._forward_differences(delta_ae, delta_bce)  # [B, C, H, W]

        difficult_zone = dz_branch_avg + dz_branch_dl * self.merge_ratio

        return self.final_transform(difficult_zone)


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV4dl(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            expand: int = 2,
            final_transform: str = 'none',
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.final_transform = _get_transform_layer(final_transform)

        from .residual_conv_block import ResidualConvBlock
        self.diff_methods = 2
        self.expand = expand
        ch_in = self.diff_methods * self.num_images * self.channels
        ch_mid = self.diff_methods * self.num_images * self.channels * self.expand
        self.norm_ae = nn.InstanceNorm2d(num_features=self.num_images * self.channels, affine=True)
        self.norm_bce = nn.InstanceNorm2d(num_features=self.num_images * self.channels, affine=True)
        self.conv_expand = nn.Conv2d(
            in_channels=ch_in, out_channels=ch_mid, groups=self.diff_methods,
            kernel_size=3, padding='same',
        )
        self.conv_block_1 = ResidualConvBlock(
            in_channels=ch_mid, out_channels=ch_mid,
            activation_type='leaky_relu', norm_type='instance',
        )
        self.conv_block_2 = ResidualConvBlock(
            in_channels=ch_mid, out_channels=ch_mid,
            activation_type='leaky_relu', norm_type='instance',
        )
        self.conv_block_3 = ResidualConvBlock(
            in_channels=ch_mid, out_channels=ch_mid,
            activation_type='silu', norm_type='instance',
        )
        self.conv_shrink = nn.Conv2d(
            in_channels=ch_mid, out_channels=self.channels, kernel_size=1,
        )

    def _calculate_difference(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param images: sequence of images [B, L, C, H, W]
        :return: tuple of difference tensors [B, L, C, H, W]
        """

        images_shifted = torch.roll(images, shifts=1, dims=1)  # [B, L, C, H, W] (shifted in L)

        ae_diff = cal_ae(images, images_shifted)  # [B, L, C, H, W]
        bce_diff = cal_bce_sigmoid(images, images_shifted)  # [B, L, C, H, W]

        return ae_diff, bce_diff

    def _forward_differences(self, ae_differences: torch.Tensor, bce_differences: torch.Tensor):
        """
        :param ae_differences: [B, L, C, H, W]
        :param bce_differences: [B, L, C, H, W]
        :return: dz prediction [B, C, H, W]
        """
        B, L, C, H, W = ae_differences.shape

        ae_cat = self.norm_ae(ae_differences.reshape(B, C * L, H, W))
        bce_cat = self.norm_bce(bce_differences.reshape(B, C * L, H, W))

        differences = torch.cat((ae_cat, bce_cat), dim=1)

        x = self.conv_expand(differences)  # [B, 2 * L * C * expand, H, W]
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_shrink(x)

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: image sequence, list of [B, C, H, W] tensors
        :return: Difficult Zone, shape [B, C, H, W]
        """

        # L * [B, C, H, W], L * [B, C, H, W]
        delta_ae, delta_bce = self._calculate_difference(images)

        dz_branch_dl = self._forward_differences(delta_ae, delta_bce)  # [B, C, H, W]

        difficult_zone = dz_branch_dl

        return self.final_transform(difficult_zone)


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV5base(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            ae_amplify: float = 4.,
            bce_scale: float = 0.33,
            expand: int = 2,
            final_transform: str = 'none',
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.ae_amplify = ae_amplify
        self.bce_scale = bce_scale

        self.final_transform = _get_transform_layer(final_transform)

        from .residual_conv_block import ResidualConvBlock
        self.diff_methods = 2
        self.expand = expand
        ch_in = self.diff_methods * self.num_images * self.channels
        ch_mid = self.diff_methods * self.num_images * self.channels * self.expand
        self.conv_expand = nn.Conv2d(
            in_channels=ch_in, out_channels=ch_mid, groups=self.diff_methods, kernel_size=3, padding='same',
        )
        self.conv_block_a = ResidualConvBlock(
            in_channels=ch_mid, out_channels=ch_mid, num_layers=2,
            activation_type='leaky_relu', norm_type='instance',
        )
        self.conv_block_b = ResidualConvBlock(
            in_channels=ch_mid, out_channels=ch_mid, num_layers=3,
            activation_type='silu', norm_type='instance',
        )
        self.conv_shrink = nn.Conv2d(
            in_channels=ch_mid, out_channels=self.channels, kernel_size=1,
        )

    def _calculate_difference(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param images: sequence of images [B, L, C, H, W]
        :return: tuple of difference tensors [B, L, C, H, W]
        """

        images_shifted = torch.roll(images, shifts=1, dims=1)  # [B, L, C, H, W] (shifted in L)

        ae_diff = cal_ae(images, images_shifted)  # [B, L, C, H, W]
        bce_diff = cal_bce_sigmoid(images, images_shifted)  # [B, L, C, H, W]

        return ae_diff, bce_diff

    def _forward_differences(self, ae_differences: torch.Tensor, bce_differences: torch.Tensor):
        """
        :param ae_differences: [B, L, C, H, W]
        :param bce_differences: [B, L, C, H, W]
        :return: dz prediction [B, C, H, W]
        """
        B, L, C, H, W = ae_differences.shape

        ae_cat = torch.sigmoid(ae_differences.reshape(B, C * L, H, W) * self.ae_amplify)
        bce_cat = torch.tanh(bce_differences.reshape(B, C * L, H, W) * self.bce_scale)

        differences = torch.cat((ae_cat, bce_cat), dim=1)

        x = self.conv_expand(differences)  # [B, 2 * L * C * expand, H, W]

        x = self.conv_block_a(x)
        x = self.conv_block_b(x)

        x = self.conv_shrink(x)

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: image sequence, list of [B, C, H, W] tensors
        :return: Difficult Zone, shape [B, C, H, W]
        """

        # [B, L, C, H, W], [B, L, C, H, W]
        delta_ae, delta_bce = self._calculate_difference(images)

        difficult_zone = self._forward_differences(delta_ae, delta_bce)  # [B, C, H, W]

        return self.final_transform(difficult_zone)
