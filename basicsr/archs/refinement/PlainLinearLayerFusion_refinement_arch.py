from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY


@REFINEMENT_ARCH_REGISTRY.register()
class PlainLinearLayerFusion(nn.Module):
    def __init__(self, branch: int, channels: int, **kwargs):
        super(PlainLinearLayerFusion, self).__init__()
        self.branch = branch
        self.channels = channels
        self.conv = nn.Conv2d(
            branch * channels, channels, kernel_size=1, stride=1, padding='same'
        )

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        # Input shape n * [B, C, H, W]
        x = torch.cat(images, dim=1)
        x = self.conv(x)
        x = F.sigmoid(x)
        # Output shape [B, C, H, W]
        return x
