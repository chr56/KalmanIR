from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY


@REFINEMENT_ARCH_REGISTRY.register()
class MEPS(nn.Module):
    """
    MEPS-Net without expert
    from paper 'Restoring Spatially-Heterogeneous Distortions using Mixture of Experts Network'
    """

    def __init__(
            self,
            branch: int,
            channels: int,
            features_size: int = 64,
            norm_image: bool = False,
            img_range: float = 1.0,
            **kwargs):
        from basicsr.archs.modules.residual_conv_block import ResidualConvBlock
        super().__init__()
        self.branch = branch
        self.channels = channels

        self.features_size = features_size
        self.feature_extractor = ResidualConvBlock(
            in_channels=self.branch * self.channels,
            out_channels=self.branch * self.features_size,
            num_layers=4, activation_type='sigmoid',
        )
        self.feature_fusion = FeatureFus(self.features_size, self.branch)
        self.reconstruction = Recon(self.features_size, self.branch)

        # rgb norm
        self.norm_image = norm_image
        if self.norm_image:
            self.sub_mean = MeanShift(rgb_range=img_range)
            self.add_mean = MeanShift(rgb_range=img_range, sign=1)

    def forward(self, images: List[torch.Tensor]) -> dict:
        # Input shape L * [B, C, H, W]
        if self.norm_image:
            images = [self.sub_mean(i) for i in images]

        x = torch.cat(images, dim=1)  # [B, L * C, H, W]

        x = self.feature_extractor(x)  # [B, L * C', H, W]
        x = self.feature_fusion(x)
        x = self.reconstruction(x)  # Output shape [B, C, H, W]

        if self.norm_image:
            x = self.add_mean(x)

        return {
            'sr_refined': x,  # [B, C, H, W]
        }

    def model_output_format(self):
        return {
            'sr_refined': 'I',
        }

    def primary_output(self):
        return 'sr_refined'


class Recon(nn.Module):
    def __init__(self, in_channels=64, num_experts=3):
        super(Recon, self).__init__()
        channel_size = in_channels * num_experts
        self.reconst = nn.Sequential(
            nn.Conv2d(channel_size, channel_size // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_size // 2, 3, 3, 1, 1)
        )

    def forward(self, x):
        y = self.reconst(x)
        return y


class FeatureFus(nn.Module):
    def __init__(self, feature_size=64, num_experts=3):
        super(FeatureFus, self).__init__()
        channel_size = feature_size * num_experts

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeeze = nn.Conv2d(channel_size, channel_size // 2, 1, 1, 0)
        self.excitation = nn.Conv2d(channel_size // 2, channel_size, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.relu(self.squeeze(y))
        y = self.sigmoid(self.excitation(y))
        y = x * y
        return y


class MeanShift(nn.Conv2d):
    def __init__(
            self,
            rgb_range,
            sign=-1,
            rgb_std=(1.0, 1.0, 1.0),
            rgb_mean=(0.4488, 0.4371, 0.4040),
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
