from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY


@REFINEMENT_ARCH_REGISTRY.register()
class AverageFusion(nn.Module):
    def __init__(self, **kwargs):
        super(AverageFusion, self).__init__()

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        # Input shape n * [B, C, H, W]
        stacked = torch.stack(images) # [n, B, C, H, W]
        averaged = torch.mean(stacked, dim=0)
        # Output shape [B, C, H, W]
        return averaged