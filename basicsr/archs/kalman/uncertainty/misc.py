import torch
from torch import nn


class DummyUncertaintyEstimator(nn.Module):
    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        return difficult_zone
