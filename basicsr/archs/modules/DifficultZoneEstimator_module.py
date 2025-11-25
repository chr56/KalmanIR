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
