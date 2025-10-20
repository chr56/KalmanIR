import torch
from torch import nn as nn
from torch.nn import functional as F

from ..utils import cal_ae, cal_bce, cal_cs, mean_difference


class DifficultZoneEstimatorV4(nn.Module):
    def __init__(self, channel: int):
        super(DifficultZoneEstimatorV4, self).__init__()

        self.channel = channel

        self.channel_stacked = 3 * channel + 1
        self.estimator_main = nn.Sequential(
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, channel, kernel_size=3, padding='same'),
        )
        self.residual_ratio = 0.4

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param a: shape [B, C, H, W]
        :param b: shape [B, C, H, W]
        :param c: shape [B, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        ##### Calculate Difference #######
        ##################################

        ae_difference = self.calculate_difference(cal_ae, b, a, c)  # [B, C, H, W]
        # kl_difference = self.calculate_deference(cal_kl, b, a, c)  # [B, C, H, W]
        bce_difference = self.calculate_difference(cal_bce, b, a, c)  # [B, C, H, W]
        cs_difference = self.calculate_difference(cal_cs, b, a, c)  # [B, 1, H, W]

        all_difference = torch.cat((b, cs_difference, ae_difference, bce_difference), dim=1)  # [B, C', H, W]

        ##################################
        #### Difficult Zone Estimation ###
        ##################################

        # ([B, C', H, W] -> [B, C, H, W]) * (1-r) + [B, C, H, W] * r
        difficult_zone = (self.estimator_main(all_difference) * (1 - self.residual_ratio) +
                          ae_difference * self.residual_ratio)

        return difficult_zone

    @staticmethod
    def calculate_difference(
            func, value_base: torch.Tensor, value1: torch.Tensor, value2: torch.Tensor
    ) -> torch.Tensor:
        difference = mean_difference(func, value_base, value1, value2)
        return F.layer_norm(difference, difference.shape[1:])


