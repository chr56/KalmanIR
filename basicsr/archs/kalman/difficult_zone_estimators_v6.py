import torch
from torch import nn as nn

from basicsr.archs.kalman.utils import mean_difference, cal_ae, cal_bce, cal_cs, cal_kl


def build_difficult_zone_estimator_for_v6(variant, dim: int, **kwargs) -> nn.Module:
    if variant == "original_v6":
        return DifficultZoneEstimatorV6(channel=dim)
    elif variant == "deep_convolutional_v1":
        merge_ratio = kwargs.get('variant_difficult_zone_merge_ratio', 0.5)
        return DeepConvolutionalV1(channel=dim, merge_ratio=merge_ratio)
    elif variant == "multi_convolutional_v1":
        return MultiConvolutionalV1(channel=dim, hidden_channels=dim // 3)
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


class MultiConvolutionalV1(nn.Module):
    def __init__(self, channel: int, hidden_channels: int):
        super(MultiConvolutionalV1, self).__init__()

        self.channel = channel

        from .convolutional_res_block import ConvolutionalResBlockLayerNorm

        self.conv_ae = ConvolutionalResBlockLayerNorm(channel, hidden_channels, activation_type='relu')
        self.conv_kl = ConvolutionalResBlockLayerNorm(channel, hidden_channels, activation_type='relu')
        self.conv_bce = ConvolutionalResBlockLayerNorm(channel, hidden_channels, activation_type='relu')

        self.conv_cs = ConvolutionalResBlockLayerNorm(channels=1, out_channels=1, activation_type='relu')

        self.channel_stacked = channel + 3 * hidden_channels + 1
        self.conv_middle = ConvolutionalResBlockLayerNorm(self.channel_stacked, activation_type='silu')
        self.conv_out = ConvolutionalResBlockLayerNorm(self.channel_stacked, channel, activation_type='silu')

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        ##################################
        ##### Calculate Difference #######
        ##################################

        ae_difference = mean_difference(cal_ae, b, a, c)  # [B, C, H, W]
        kl_difference = mean_difference(cal_kl, b, a, c)  # [B, C, H, W]
        bce_difference = mean_difference(cal_bce, b, a, c)  # [B, C, H, W]
        cs_difference = mean_difference(cal_cs, b, a, c)  # [B, 1, H, W]

        ##################################
        #### Difficult Zone Estimation ###
        ##################################

        in_ae = self.conv_ae(ae_difference)
        in_kl = self.conv_kl(kl_difference)
        in_bce = self.conv_bce(bce_difference)
        in_cs = self.conv_cs(cs_difference)

        del ae_difference, bce_difference, cs_difference, kl_difference

        all_difference = torch.cat((b, in_ae, in_kl, in_bce, in_cs), dim=1)  # [B, C', H, W]

        middle = self.conv_middle(all_difference)
        difficult_zone = self.conv_out(middle)

        return difficult_zone


class DeepConvolutionalV1(nn.Module):
    def __init__(
            self,
            channel: int,
            merge_ratio: float,
            norm_affine: bool = True,
    ):
        super(DeepConvolutionalV1, self).__init__()

        self.channel = channel

        self.merge_ratio = merge_ratio
        self.keep_ratio = 1 - self.merge_ratio

        from .utils import LayerNorm2d
        self.layer_norm_ae = LayerNorm2d(channel, elementwise_affine=norm_affine)
        # self.layer_norm_kl = LayerNorm2d(channel, elementwise_affine=norm_affine)
        self.layer_norm_bce = LayerNorm2d(channel, elementwise_affine=norm_affine)
        self.layer_norm_cs = LayerNorm2d(1, elementwise_affine=norm_affine)

        from .convolutional_res_block import ConvolutionalResBlockLayerNorm
        self.channel_stacked = 3 * channel + 1
        self.block1 = ConvolutionalResBlockLayerNorm(
            channels=self.channel_stacked, out_channels=channel, activation_type='leaky_relu',
        )
        self.block2 = ConvolutionalResBlockLayerNorm(
            channels=channel, out_channels=channel, activation_type='silu',
        )

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

        # [B, C', H, W] -> [B, C, H, W] * ratio
        x = all_difference
        x = self.block1(x)
        x = self.block2(x)
        x = self.merge_ratio * x

        # [B, C, H, W] * (1 - ratio)
        y = self.keep_ratio * ae_difference

        difficult_zone = x + y
        return difficult_zone
