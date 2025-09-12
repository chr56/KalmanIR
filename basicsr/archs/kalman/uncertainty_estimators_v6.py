import torch
from torch import nn
from torch.nn import functional as F


def build_uncertainty_estimator_for_v6(variant, dim: int, seq_length: int) -> nn.Module:
    if variant == "mamba_recursive_state_adjustment_v1":
        return MambaRecursiveStateAdjustmentV1(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v2":
        return MambaRecursiveStateAdjustmentV2(seq_length, dim)
    elif variant == "mamba_recursive_state_adjustment_v3":
        return MambaRecursiveStateAdjustmentV3(seq_length, dim)
    elif variant == "recursive_convolutional_v1":
        return RecursiveConvolutionalV1(seq_length, dim)
    elif variant == "recursive_convolutional_v2":
        return RecursiveConvolutionalV2(seq_length, dim)
    elif variant == "recursive_convolutional_v3":
        return RecursiveConvolutionalV3(seq_length, dim)
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


class MambaRecursiveStateAdjustmentV3(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(MambaRecursiveStateAdjustmentV3, self).__init__()
        self.channel = channel
        self.iteration = length

        from .utils import cal_kl, LayerNorm2d
        self.cal_kl = cal_kl
        self.norm_difficult_zone = LayerNorm2d(channel)
        self.norm_image = LayerNorm2d(channel)

        self.conv_init = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding='same', groups=channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same', groups=channel),
            nn.ReLU(inplace=True),
        )

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma_state = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_sigma_image = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=channel, **kwargs)

    def _initial_state(self, difficult_zone: torch.Tensor):
        previous = self.norm_difficult_zone(difficult_zone)
        sigma = self.conv_init(previous)
        bias = self.mamba_bias(previous)
        current = previous / torch.exp(sigma) + bias
        return current, previous

    def _forward_one_step(self, current_state: torch.Tensor, previous_state: torch.Tensor, image: torch.Tensor):
        sigma_image = self.mamba_sigma_image(self.norm_image(image))
        sigma_state = self.mamba_sigma_state(self.cal_kl(current_state, previous_state))
        bias = self.mamba_bias(current_state)
        all_sigma = torch.exp(-sigma_state) * torch.exp(-sigma_image)
        next_state = (current_state / all_sigma) + bias
        return next_state, current_state

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        current, previous = self._initial_state(difficult_zone)
        for i in range(self.iteration):
            image = image_sequence[:, i, ...]
            current, previous = self._forward_one_step(
                current_state=current,
                previous_state=previous,
                image=image,
            )
        return current


class RecursiveConvolutionalV1(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(RecursiveConvolutionalV1, self).__init__()
        self.iteration = length
        self.channel = channel
        self.residual_rate = 0.2
        from .utils import LayerNorm2d
        from .convolutional_res_block import ConvolutionalResBlock
        self.norm_image = LayerNorm2d(channel)
        self.conv_block = ConvolutionalResBlock(
            3 * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        assert image_sequence.shape[1] == self.iteration

        uncertainty = difficult_zone
        for i in range(self.iteration):
            image = self.norm_image(image_sequence[:, i, ...])
            block = torch.cat((uncertainty, image, sigma), dim=1)
            uncertainty = self.conv_block(block)

        uncertainty = uncertainty * (1 - self.residual_rate) + difficult_zone * self.residual_rate

        return uncertainty


class RecursiveConvolutionalV2(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(RecursiveConvolutionalV2, self).__init__()
        self.iteration = length
        self.channel = channel
        self.residual_rate = 0.2

        self.norm_image = nn.GroupNorm(3, channel)

        self.conv_block_main = nn.Sequential(
            nn.GroupNorm(2, 2 * channel),
            nn.Conv2d(2 * channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
        )
        self.conv_block_side = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        assert image_sequence.shape[1] == self.iteration

        uncertainty = difficult_zone
        for i in range(self.iteration):
            image = self.norm_image(image_sequence[:, i, ...])
            block = torch.cat((uncertainty, image), dim=1)
            uncertainty = self.conv_block_main(block) + self.conv_block_side(block)

        uncertainty = uncertainty * (1 - self.residual_rate) + difficult_zone * self.residual_rate

        return uncertainty


class RecursiveConvolutionalV3(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(RecursiveConvolutionalV3, self).__init__()
        self.iteration = length
        self.channel = channel

        self.norm_image = nn.GroupNorm(3, channel)

        self.conv_block_main = nn.Sequential(
            nn.Conv2d(2 * channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, channel),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
        )
        self.conv_block_side = nn.Sequential(
            nn.Conv2d(2 * channel, 2 * channel, kernel_size=3, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.GroupNorm(2, 2 * channel),
            nn.Conv2d(2 * channel, channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(inplace=True),
        )

        self.merge_rate = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        assert image_sequence.shape[1] == self.iteration

        state = difficult_zone
        for i in range(self.iteration):
            image = self.norm_image(image_sequence[:, i, ...])
            block = torch.cat((state, image), dim=1)
            state = self.conv_block_main(block) + self.conv_block_side(block)

        uncertainty = difficult_zone + state * F.sigmoid(self.merge_rate)
        uncertainty = F.sigmoid(uncertainty)

        return uncertainty
