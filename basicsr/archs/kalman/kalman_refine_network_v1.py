from typing import List

import torch
import torch.nn as nn
from einops import rearrange

from basicsr.archs.arch_util import init_weights
from .convolutional_res_block import ConvolutionalResBlock
from .kalman_filter import KalmanFilter


class KalmanRefineNetV1(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(self, dim: int):
        super(KalmanRefineNetV1, self).__init__()

        self.uncertainty_estimator = UncertaintyEstimator(dim)

        kalman_gain_calculator = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            ConvolutionalResBlock(dim, dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        predictor = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            ConvolutionalResBlock(dim, dim),
            nn.Sigmoid(),
        )

        self.kalman_filter = KalmanFilter(
            kalman_gain_calculator=kalman_gain_calculator,
            predictor=predictor,
        )

        self.apply(init_weights)

    # noinspection PyPep8Naming
    def forward(
            self,
            sequence: List[torch.Tensor],
            factor_dz: torch.Tensor,
            sigma: torch.Tensor,
    ) -> torch.Tensor:

        sequence = torch.stack(sequence, dim=1).contiguous()
        B, L, C, H, W = sequence.shape

        uncertainty = self.uncertainty_estimator(sequence, factor_dz, sigma)
        kalman_gain = self.kalman_filter.calc_gain(uncertainty, B)

        del uncertainty
        refined_with_kf = self.kalman_filter.perform_filtering(sequence, kalman_gain)

        return refined_with_kf


class UncertaintyEstimator(nn.Module):
    def __init__(
            self, channel: int,
    ):
        super(UncertaintyEstimator, self).__init__()
        self.block = ConvolutionalResBlock(
            3 * channel, channel,
            norm_num_groups_1=3, norm_num_groups_2=channel // 4,
        )

    def forward(
            self,
            sequence: torch.Tensor,
            factor_dz: torch.Tensor,
            sigma: torch.Tensor,
    ):
        """
        :param sequence: Image sequence, shape [B, L, C, H, W]
        :param factor_dz: difficulty zone factor, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B*L, C, H, W]
        """
        _, sequence_length, _, _, _ = sequence.shape

        merged = []
        for l in range(sequence_length):
            stacked = torch.cat((sequence[:, l, ...], factor_dz, sigma), dim=1)  # -> [B, 3C, H, W]
            merged.append(
                self.block(stacked)  # [B, 3C, H, W] -> [B, C, H, W]
            )
        uncertainty = torch.cat(merged, dim=1)  # L * [B, C, H, W] -> [B, L * C, H, W]

        uncertainty = rearrange(uncertainty, "b (l c) h w -> (b l) c h w", l=sequence_length)
        return uncertainty
