from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY
from .utils_calculation import cal_ae, cal_bce, cal_bce_sigmoid, cal_cs, cal_kl


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV1(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
    ):
        super(DifficultZoneEstimatorV1, self).__init__()

        self.channels = channels
        self.num_images = num_images

        from .residual_conv_block import ResidualConvBlock
        diff_metric_methods = 2
        self.main_block1 = ResidualConvBlock(
            in_channels=channels * num_images * diff_metric_methods, out_channels=channels * num_images,
            activation_type='silu', norm_type='group', norm_group=[num_images * diff_metric_methods, num_images],
        )
        self.main_block2 = ResidualConvBlock(
            in_channels=channels * num_images, out_channels=channels,
            activation_type='silu', norm_type='layer',
        )
        self.liner1 = nn.Conv2d(
            num_images * channels, num_images * channels,
            kernel_size=1, padding='same'
        )
        self.liner2 = nn.Conv2d(
            num_images * channels, channels,
            kernel_size=1, padding='same'
        )

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param images: image sequence, shape [B, L, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        ae_differences = []
        bce_differences = []
        for i in range(self.num_images):
            ae_differences.append(cal_ae(images[:, i, ...], images[:, i - 1, ...]))
            bce_differences.append(cal_bce(images[:, i, ...], images[:, i - 1, ...]))

        ##################################

        ae_differences = torch.cat(ae_differences, dim=1)
        bce_differences = torch.cat(bce_differences, dim=1)
        differences = torch.cat((ae_differences, bce_differences), dim=1)

        ##################################

        x = differences
        x = self.main_block1(x)
        x = self.main_block2(x)

        y = ae_differences
        y = self.liner1(y)
        y = nn.functional.leaky_relu(y)
        y = self.liner2(y)
        y = torch.sigmoid(y)

        difficult_zone = x + y

        return torch.sigmoid(difficult_zone)


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV2s(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            merge_ratio: float = 1.0,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.merge_ratio = merge_ratio

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

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param images: image sequence, shape [B, L, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        ae_differences = []
        bce_differences = []
        for i in range(self.num_images):
            ae_differences.append(cal_ae(images[:, i, ...], images[:, i - 1, ...]))
            bce_differences.append(cal_bce(images[:, i, ...], images[:, i - 1, ...]))

        ##################################

        avg_ae_differences = torch.mean(torch.stack(ae_differences), dim=0)

        ae_differences = torch.cat(ae_differences, dim=1)
        bce_differences = torch.cat(bce_differences, dim=1)
        differences = torch.cat((ae_differences, bce_differences), dim=1)

        ##################################

        x = differences
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        difficult_zone = x * self.merge_ratio + avg_ae_differences

        return torch.tanh(difficult_zone)


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV3base(nn.Module):
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
        self.final_transform = final_transform

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

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param images: image sequence, shape [B, L, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        ae_differences = []
        bce_differences = []
        for i in range(self.num_images):
            ae_differences.append(cal_ae(images[:, i, ...], images[:, i - 1, ...]))
            bce_differences.append(cal_bce(images[:, i, ...], images[:, i - 1, ...]))

        ##################################

        avg_ae_differences = torch.mean(torch.stack(ae_differences), dim=0)

        ae_differences = torch.cat(ae_differences, dim=1)
        bce_differences = torch.cat(bce_differences, dim=1)
        differences = torch.cat((ae_differences, bce_differences), dim=1)

        ##################################

        x = differences
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        difficult_zone = x * self.merge_ratio + avg_ae_differences

        if self.final_transform == 'tanh':
            return torch.tanh(difficult_zone)
        elif self.final_transform == 'sigmoid':
            return torch.sigmoid(difficult_zone)
        else:
            return difficult_zone


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV3bces(nn.Module):
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
        self.final_transform = final_transform

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

    def _calculate_difference(self, images: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ae_differences = []  # L * [B, C, H, W]
        bce_differences = []  # L * [B, C, H, W]
        for i in range(self.num_images):
            img1 = images[:, i, ...]
            img2 = images[:, i - 1, ...]
            ae_differences.append(
                cal_ae(img1, img2)
            )
            bce_differences.append(
                cal_bce_sigmoid(img1, img2)
            )
        return ae_differences, bce_differences

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param images: image sequence, shape [B, L, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        # L * [B, C, H, W], L * [B, C, H, W]
        ae_differences, bce_differences = self._calculate_difference(images)

        avg_ae_differences = torch.mean(torch.stack(ae_differences), dim=0)

        ae_differences = torch.cat(ae_differences, dim=1)
        bce_differences = torch.cat(bce_differences, dim=1)
        differences = torch.cat((ae_differences, bce_differences), dim=1)

        ##################################

        x = differences
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        difficult_zone = x * self.merge_ratio + avg_ae_differences

        if self.final_transform == 'tanh':
            return torch.tanh(difficult_zone)
        elif self.final_transform == 'sigmoid':
            return torch.sigmoid(difficult_zone)
        else:
            return difficult_zone


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV3avg(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            merge_ratio: float = 1.0,
            final_transform: str = 'none',
            expand: int = 8,
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        self.merge_ratio = merge_ratio
        self.expand = expand
        self.final_transform = final_transform

        from .residual_conv_block import ResidualConvBlock
        difference_methods = 2
        ch_layer_0 = channels * difference_methods
        ch_layer_1 = channels * difference_methods * self.expand
        ch_layer_2 = channels * self.expand
        ch_layer_3 = channels
        self.conv_block_0 = ResidualConvBlock(
            in_channels=ch_layer_0, out_channels=ch_layer_1, num_layers=2,
            activation_type='leaky_relu', norm_type='group', norm_group=difference_methods,
        )
        self.conv_block_1 = ResidualConvBlock(
            in_channels=ch_layer_1, out_channels=ch_layer_2, num_layers=3,
            activation_type='silu', norm_type='group', norm_group=self.expand,
        )
        self.conv_block_2 = ResidualConvBlock(
            in_channels=ch_layer_2, out_channels=ch_layer_3, num_layers=2,
            activation_type='silu', norm_type='layer',
        )

    def _calculate_avg_difference(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        ae_differences = []  # L * [B, C, H, W]
        bce_differences = []  # L * [B, C, H, W]
        for i in range(self.num_images):
            img1 = images[:, i, ...]
            img2 = images[:, i - 1, ...]
            ae_differences.append(
                cal_ae(img1, img2)
            )
            bce_differences.append(
                cal_bce(img1, img2)
            )
        avg_ae_differences = torch.mean(torch.stack(ae_differences, dim=0), dim=0)  # [B, C, H, W]
        avg_bce_differences = torch.mean(torch.stack(bce_differences, dim=0), dim=0)  # [B, C, H, W]
        return avg_ae_differences, avg_bce_differences

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param images: image sequence, shape [B, L, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        # [B, C, H, W], [B, C, H, W]
        avg_ae_differences, avg_bce_differences = self._calculate_avg_difference(images)

        ##################################

        x = torch.cat((avg_ae_differences, avg_bce_differences), dim=1)  # [B, mC, H, W]
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        ##################################

        difficult_zone = x * self.merge_ratio + avg_ae_differences

        if self.final_transform == 'tanh':
            return torch.tanh(difficult_zone)
        elif self.final_transform == 'sigmoid':
            return torch.sigmoid(difficult_zone)
        else:
            return difficult_zone


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV3w(nn.Module):
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
        self.final_transform = final_transform

        from .residual_conv_block import ResidualConvBlock
        ch_layer_1 = channels * num_images
        ch_layer_2 = channels
        self.conv_block_0 = ResidualConvBlock(
            in_channels=ch_layer_1, out_channels=ch_layer_1, num_layers=3,
            activation_type='silu', norm_type='group', norm_group=num_images,
        )
        self.conv_block_1 = ResidualConvBlock(
            in_channels=ch_layer_1, out_channels=ch_layer_1, num_layers=2,
            activation_type='silu', norm_type='layer',
        )
        self.conv_block_2 = ResidualConvBlock(
            in_channels=ch_layer_1, out_channels=ch_layer_2, num_layers=2,
            activation_type='sigmoid', norm_type='layer',
        )

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param images: image sequence, shape [B, L, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        ae_differences = []
        for i in range(self.num_images):
            ae_differences.append(cal_ae(images[:, i, ...], images[:, i - 1, ...]))

        avg_ae_differences = torch.mean(torch.stack(ae_differences), dim=0)

        ##################################

        x = torch.cat(ae_differences, dim=1)
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        difficult_zone = x * self.merge_ratio + avg_ae_differences

        if self.final_transform == 'tanh':
            return torch.tanh(difficult_zone)
        elif self.final_transform == 'sigmoid':
            return torch.sigmoid(difficult_zone)
        else:
            return difficult_zone


@MODULES_REGISTRY.register()
class DifficultZoneEstimatorV3pb(nn.Module):
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
        self.final_transform = final_transform

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

    def _calculate_difference(self, images: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        ae_differences = []  # L * [B, C, H, W]
        bce_differences = []  # L * [B, C, H, W]
        for i in range(self.num_images):
            img1 = images[:, i, ...]
            img2 = images[:, i - 1, ...]
            ae_differences.append(
                cal_ae(img1, img2)
            )
            bce_differences.append(
                torch.sigmoid(cal_bce(img1, img2))
            )
        return ae_differences, bce_differences

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        :param images: image sequence, shape [B, L, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        # L * [B, C, H, W], L * [B, C, H, W]
        ae_differences, bce_differences = self._calculate_difference(images)

        avg_ae_differences = torch.mean(torch.stack(ae_differences), dim=0)

        ae_differences = torch.cat(ae_differences, dim=1)
        bce_differences = torch.cat(bce_differences, dim=1)
        differences = torch.cat((ae_differences, bce_differences), dim=1)

        ##################################

        x = differences
        x = self.conv_block_0(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        difficult_zone = x * self.merge_ratio + avg_ae_differences

        if self.final_transform == 'tanh':
            return torch.tanh(difficult_zone)
        elif self.final_transform == 'sigmoid':
            return torch.sigmoid(difficult_zone)
        else:
            return difficult_zone


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
class DifficultZoneEstimatorV4norm(nn.Module):
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

        self.norm_ae = nn.InstanceNorm2d(channels)
        self.norm_bce = nn.InstanceNorm2d(channels)

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
        # noinspection PyPep8Naming
        B, L, C, H, W = images.shape

        images_shifted = torch.roll(images, shifts=1, dims=1)  # [B, L, C, H, W] (shifted in L)

        ae_diff = cal_ae(images, images_shifted)  # [B, L, C, H, W]
        bce_diff = cal_bce_sigmoid(images, images_shifted)  # [B, L, C, H, W]

        ae_diff_norm = self.norm_ae(ae_diff.view(B * L, C, H, W)).view(B, L, C, H, W)
        bce_diff_norm = self.norm_bce(bce_diff.view(B * L, C, H, W)).view(B, L, C, H, W)

        return ae_diff_norm, bce_diff_norm

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
class DifficultZoneEstimatorV4ref(nn.Module):
    def __init__(
            self,
            channels: int,
            num_images: int,
            expand: int = 2,
            ref_img_index: int = 0,
            final_transform: str = 'none',
    ):
        super().__init__()

        self.channels = channels
        self.num_images = num_images

        assert -1 < ref_img_index < num_images, f"Reference image index ({ref_img_index}) is out of range"
        self.ref_img_index = ref_img_index

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
        self.conv_block_diff = ResidualConvBlock(
            in_channels=ch_mid, out_channels=ch_mid,
            activation_type='silu', norm_type='instance',
        )
        self.conv_shrink = nn.Conv2d(
            in_channels=ch_mid, out_channels=self.channels, kernel_size=1,
        )
        self.conv_bias = ResidualConvBlock(
            in_channels=self.channels, out_channels=self.channels, num_layers=3,
            activation_type='leaky_relu',
        )
        self.conv_weight = ResidualConvBlock(
            in_channels=self.channels, out_channels=self.channels, num_layers=3,
            activation_type='leaky_relu', norm_after_conv=True,
        )

    def _forward_differences(
            self,
            diff_ae: torch.Tensor,
            diff_bce: torch.Tensor,
            reference: torch.Tensor,
    ):
        """
        :param diff_ae: [B, L, C, H, W]
        :param diff_bce: [B, L, C, H, W]
        :param reference: [B, C, H, W]
        :return: predicted difficult zone [B, C, H, W]
        """
        B, L, C, H, W = diff_ae.shape
        ae = diff_ae.reshape(B, C * L, H, W)
        bce = diff_bce.reshape(B, C * L, H, W)

        ae_normed = self.norm_ae(ae)
        bce_normed = self.norm_bce(bce)

        x = self.conv_expand(torch.cat((ae_normed, bce_normed), dim=1))  # [B, 2 * L * C * expand, H, W]
        x = self.conv_block_diff(x)
        x = self.conv_shrink(x)

        bias = self.conv_bias(reference)
        weight = torch.exp(self.conv_weight(reference))
        x = weight * x + bias

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: image sequence, list of [B, C, H, W] tensors
        :return: Difficult Zone, shape [B, C, H, W]
        """

        #################################

        images_shifted = torch.roll(images, shifts=1, dims=1)  # [B, L, C, H, W] (shift in dim L)

        diff_ae = cal_ae(images, images_shifted)  # [B, L, C, H, W]
        diff_bce = cal_bce_sigmoid(images, images_shifted)  # [B, L, C, H, W]

        #################################

        ref_img = images[:, self.ref_img_index, ...]  # [B, C, H, W]

        difficult_zone = self._forward_differences(diff_ae, diff_bce, ref_img)  # [B, C, H, W]

        return self.final_transform(difficult_zone)
