from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.modules.layer_norm import LayerNorm2d
from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY


@REFINEMENT_ARCH_REGISTRY.register()
class DichotomyIR(nn.Module):
    def __init__(
            self,
            branch: int,
            channels: int,
            mamba_d_state: int = 16,
            mamba_d_expand: float = 2.,
            mamba_dt_rank='auto',
            **kwargs):
        super(DichotomyIR, self).__init__()
        self.branch = branch
        self.channels = channels * 8

        self.uem = UncertaintyEliminationMambaBlock(
            channel=self.channels,
            mamba_d_state=mamba_d_state,
            mamba_d_expand=mamba_d_expand,
            mamba_dt_rank=mamba_dt_rank,
        )

    def forward(self, images: List[torch.Tensor]) -> dict:
        # Input shape n * [B, C, H, W]

        # Convert
        images = [decimal_to_binary(image) for image in images]

        # Init
        refined = images[0]
        bias = images[1] - images[0]

        # Refine iteratively
        for i in range(self.branch):
            refined, bias = self.uem(img_current=refined, img_previous=images[i], bias=bias)

        # Recover
        refined = binary_to_decimal(refined)

        # Output shape [B, C, H, W]
        return {
            'sr_refined': refined,  # [B, C, H, W]
        }

    def model_output_format(self):
        return {
            'sr_refined': 'I',
        }

    def primary_output(self):
        return 'sr_refined'


class UncertaintyEliminationMambaBlock(nn.Module):
    def __init__(
            self,
            channel: int = 3,
            mamba_d_state: int = 16,
            mamba_d_expand: float = 2.,
            mamba_dt_rank='auto',
            **kwargs,
    ):
        super().__init__()
        from basicsr.archs.modules.modules_ss2d import SS2DChanelFirst
        self.mamba_sigma = SS2DChanelFirst(
            d_model=channel,
            d_state=mamba_d_state,
            expand=mamba_d_expand,
            dt_rank=mamba_dt_rank,
            **kwargs,
        )
        self.mamba_bias = SS2DChanelFirst(
            d_model=channel,
            d_state=mamba_d_state,
            expand=mamba_d_expand,
            dt_rank=mamba_dt_rank,
            **kwargs,
        )
        self.norm_img = LayerNorm2d(channel)
        self.norm_bias = LayerNorm2d(channel)

    def forward(self, img_current: torch.Tensor, img_previous: torch.Tensor, bias: torch.Tensor):
        img_previous_norm = self.norm_img(img_previous)
        img_current_norm = self.norm_img(img_current)
        bias_norm = self.norm_bias(bias)

        kld = _cal_kl(img_current_norm, img_previous_norm)
        sigma = self.mamba_sigma(kld)
        bias_next = self.mamba_bias(bias_norm)
        img_refined = (img_current / torch.exp(-sigma)) + bias_next

        return img_refined, bias_next


def _cal_kl(pred, target):
    p = torch.log_softmax(pred, dim=1)
    q = torch.softmax(target, dim=1)
    return F.kl_div(p, q, reduction='none')


def binary_to_decimal(
        binary_tensors: torch.Tensor,
        value_range: float = 255.,
) -> torch.Tensor:
    """Convert binary tensors [b, 8C, h, w] to decimal tensors [b, C, h, w]."""

    no_batch_dimension = len(binary_tensors.shape) == 3
    if no_batch_dimension:
        binary_tensors = binary_tensors.unsqueeze(0)

    b, c_binary, h, w = binary_tensors.shape
    assert c_binary % 8 == 0, f"Channel must be divisible by 8, got {binary_tensors.shape}."
    c = c_binary // 8  # real channel in decimal

    #########################

    square = torch.tensor(
        [1., 2., 4., 8., 16., 32., 64., 128.],
        device=binary_tensors.device, dtype=binary_tensors.dtype
    ).reshape(1, 8, 1, 1)

    binary_reshaped = binary_tensors.reshape(b, c, 8, h, w)

    decimal_tensors = torch.sum(square * binary_reshaped, dim=2)

    #########################

    if value_range > 0:
        decimal_tensors = decimal_tensors / value_range

    if no_batch_dimension:
        decimal_tensors = decimal_tensors.squeeze(0)

    return decimal_tensors


def decimal_to_binary(
        decimal_tensors: torch.Tensor,
        value_range: float = 255.,
) -> torch.Tensor:
    """Convert decimal tensors [b, C, h, w] (values 0-255) to binary tensors [b, 8C, h, w]."""

    no_batch_dimension = len(decimal_tensors.shape) == 3
    if no_batch_dimension:
        decimal_tensors = decimal_tensors.unsqueeze(0)

    if value_range > 0:
        decimal_tensors = decimal_tensors * value_range

    b, c, h, w = decimal_tensors.shape

    binary_reshaped = torch.zeros(
        (b, c, 8, h, w),
        dtype=decimal_tensors.dtype,
        device=decimal_tensors.device
    )

    for i in range(8):
        bit_value = (decimal_tensors.to(torch.uint8) >> i) & 1
        binary_reshaped[:, :, i, :, :] = bit_value.to(decimal_tensors.dtype)

    output_tensor = binary_reshaped.reshape(b, c * 8, h, w)

    if no_batch_dimension:
        output_tensor = output_tensor.squeeze(0)

    return output_tensor
