from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY


@REFINEMENT_ARCH_REGISTRY.register()
class Skipped(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, images: List[torch.Tensor]):
        return None