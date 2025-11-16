from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from . import MODULES_REGISTRY
from .utils_calculation import cal_ae, cal_bce, cal_cs, cal_kl


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
