import torch
from torch import nn
from torch.nn import functional as F


def build_uncertainty_estimator_for_v6(variant, dim: int, seq_length: int) -> nn.Module:
    if variant == "mamba_recursive_state_adjustment_v1":
        return MambaRecursiveStateAdjustmentV1(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v2":
        return MambaRecursiveStateAdjustmentV2(seq_length, dim)
    else:
        raise ValueError(f"Unsupported variant: {variant}")


class MambaRecursiveStateAdjustmentV1(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(MambaRecursiveStateAdjustmentV1, self).__init__()
        self.channel = channel
        self.iteration = length

        from .utils import cal_kl, LayerNorm2d
        self.cal_kl = cal_kl
        self.norm_image = LayerNorm2d(channel)
        self.norm_difficult_zone = LayerNorm2d(channel)

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=channel, **kwargs)

    def _forward_one_step(self, current_state, previous_state, image):
        sigma = self.mamba_sigma(self.cal_kl(current_state, previous_state))
        bias = self.mamba_bias(self.norm_image(image))
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        previous = self.norm_difficult_zone(difficult_zone)  # Initial value
        current = previous / torch.exp(-sigma)  # Initial value
        for i in range(self.iteration):
            image = image_sequence[:, i, ...]
            current, previous = self._forward_one_step(
                current_state=current,
                previous_state=previous,
                image=image,
            )
        return current


class MambaRecursiveStateAdjustmentV2(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(MambaRecursiveStateAdjustmentV2, self).__init__()
        self.channel = channel
        self.iteration = length

        from .utils import cal_kl, LayerNorm2d
        self.cal_kl = cal_kl
        self.norm_difficult_zone = LayerNorm2d(channel)
        self.norm_image = LayerNorm2d(channel)

        self.conv_init = nn.Conv2d(channel, channel, kernel_size=3, padding='same')

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma_state = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_sigma_image = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=channel, **kwargs)

    def _forward_one_step(self, current_state, previous_state, image):
        sigma_image = self.mamba_sigma_image(self.norm_image(image))
        sigma_state = self.mamba_sigma_state(self.cal_kl(current_state, previous_state))
        bias = self.mamba_bias(current_state)
        all_sigma = torch.exp(-sigma_state) * torch.exp(-sigma_image)
        next_state = (current_state / all_sigma) + bias
        return next_state, current_state

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        previous = self.norm_difficult_zone(difficult_zone)  # Initial value
        current = previous / torch.exp(-self.conv_init(sigma))  # Initial value
        for i in range(self.iteration):
            image = image_sequence[:, i, ...]
            current, previous = self._forward_one_step(
                current_state=current,
                previous_state=previous,
                image=image,
            )
        return current