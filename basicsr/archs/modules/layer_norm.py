import torch
from torch import nn
from torch.nn import functional as F


def layer_norm_2d(
        tensor: torch.Tensor,
        normalized_shape,
        weight=None,
        bias=None,
        eps=1e-5
) -> torch.Tensor:
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = F.layer_norm(tensor, normalized_shape, weight, bias, eps)
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


class LayerNorm2d(nn.LayerNorm):
    """ shape [B, C, H, W] """

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
