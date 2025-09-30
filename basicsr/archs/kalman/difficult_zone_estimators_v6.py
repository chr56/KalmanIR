import torch
from torch import nn as nn

from basicsr.archs.kalman.utils import mean_difference, cal_ae, cal_bce, cal_cs


def build_difficult_zone_estimator_for_v6(variant, dim: int, **kwargs) -> nn.Module:
    if variant == "original_v6":
        return DifficultZoneEstimatorV6(channel=dim)
    else:
        if variant:
            import warnings
            warnings.warn(f"Unknown difficult zone estimator variant `{variant}`, use default instead!")
        return DifficultZoneEstimatorV6(channel=dim)


class DifficultZoneEstimatorV6(nn.Module):
    def __init__(
            self,
            channel: int,
            residual_ratio: float = 0.4,
            norm_affine: bool = True,
    ):
        super(DifficultZoneEstimatorV6, self).__init__()

        self.channel = channel

        from basicsr.archs.kalman.utils import LayerNorm2d
        self.layer_norm_ae = LayerNorm2d(channel, elementwise_affine=norm_affine)
        self.layer_norm_kl = LayerNorm2d(channel, elementwise_affine=norm_affine)
        self.layer_norm_bce = LayerNorm2d(channel, elementwise_affine=norm_affine)
        self.layer_norm_cs = LayerNorm2d(1, elementwise_affine=norm_affine)

        self.channel_stacked = 3 * channel + 1
        self.estimator_main = nn.Sequential(
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            LayerNorm2d(self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            LayerNorm2d(self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, channel, kernel_size=3, padding='same'),
        )
        self.residual_ratio = residual_ratio

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

        ae_difference = self.layer_norm_ae(
            mean_difference(cal_ae, b, a, c)
        )  # [B, C, H, W]
        # kl_difference = self.layer_norm_kl(
        #     mean_difference(cal_kl, b, a, c)
        # )  # [B, C, H, W]
        bce_difference = self.layer_norm_bce(
            mean_difference(cal_bce, b, a, c)
        )  # [B, C, H, W]
        cs_difference = self.layer_norm_cs(
            mean_difference(cal_cs, b, a, c)
        )  # [B, 1, H, W]

        all_difference = torch.cat((b, cs_difference, ae_difference, bce_difference), dim=1)  # [B, C', H, W]

        ##################################
        #### Difficult Zone Estimation ###
        ##################################

        # ([B, C', H, W] -> [B, C, H, W]) * (1-r) + [B, C, H, W] * r
        difficult_zone = (self.estimator_main(all_difference) * (1 - self.residual_ratio) +
                          ae_difference * self.residual_ratio)

        return difficult_zone
