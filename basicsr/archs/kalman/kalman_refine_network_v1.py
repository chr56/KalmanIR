from typing import List

import torch
import torch.nn as nn
from einops import rearrange

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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

        z_hat = None
        previous_z = None
        for i in range(L):
            if i == 0:
                z_hat = sequence[:, i, ...]  # initialize Z_hat with first z
            else:
                z_prime = self.kalman_filter.predict(previous_z.detach())
                z_hat = self.kalman_filter.update(
                    sequence[:, i, ...],
                    z_prime,
                    kalman_gain[:, i, ...]
                )

            previous_z = z_hat
            pass

        return z_hat


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
