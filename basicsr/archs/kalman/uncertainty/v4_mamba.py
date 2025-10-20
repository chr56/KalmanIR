import torch
from torch import nn


class UncertaintyEstimatorIterativeNarrowMambaBlock(nn.Module):
    def __init__(self, length: int, channel: int, ):
        super(UncertaintyEstimatorIterativeNarrowMambaBlock, self).__init__()
        self.length = length
        from basicsr.archs.modules_mamba import SS2D, Mlp, VSSBlockFabric
        self.vss_block = VSSBlockFabric(
            dim=channel,
            ssm_block=SS2D(d_model=channel),
            mlp_block=Mlp(channel),
            post_norm=False,
        )
        from ..convolutional_res_block import ConvolutionalResBlock
        self.channel_compressor = ConvolutionalResBlock(
            length * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )
        from ..utils import LayerNorm2d
        self.norm = LayerNorm2d(channel)

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        length = image_sequence.shape[1]
        assert length == self.length, ValueError(f"Expected length of {length} but got {length}")

        uncertainty = []
        for i in range(length):
            x = torch.cat((difficult_zone, self.norm(image_sequence[:, i, ...]), sigma), dim=1)
            x = self.channel_compressor(x)
            x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            x = self.vss_block(x)
            x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            uncertainty.append(x)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorIterativeWideMambaBlock(nn.Module):
    def __init__(self, length: int, channel: int, ):
        super(UncertaintyEstimatorIterativeWideMambaBlock, self).__init__()
        self.length = length
        from basicsr.archs.modules_mamba import SS2D, Mlp, VSSBlockFabric
        self.vss_block = VSSBlockFabric(
            dim=length * channel,
            ssm_block=SS2D(d_model=length * channel),
            mlp_block=Mlp(length * channel),
            post_norm=False,
        )
        from ..convolutional_res_block import ConvolutionalResBlock
        self.channel_compressor = ConvolutionalResBlock(
            length * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )
        from ..utils import LayerNorm2d
        self.norm = LayerNorm2d(channel)

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        length = image_sequence.shape[1]
        assert length == self.length, ValueError(f"Expected length of {length} but got {length}")

        uncertainty = []
        for i in range(length):
            x = torch.cat((difficult_zone, self.norm(image_sequence[:, i, ...]), sigma), dim=1)
            x = x.permute(0, 2, 3, 1)  # [B, C', H, W] -> [B, H, W, C']
            x = self.vss_block(x)
            x = x.permute(0, 3, 1, 2)  # [B, H, W, C'] -> [B, C', H, W]
            x = self.channel_compressor(x)
            uncertainty.append(x)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorIterativeMambaErrorEstimation(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(UncertaintyEstimatorIterativeMambaErrorEstimation, self).__init__()
        self.channel = channel
        self.iteration = length

        from basicsr.archs.kalman.utils import cal_kl
        self.cal_kl = cal_kl

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=channel, **kwargs)

        self.merger = nn.Sequential(
            nn.GroupNorm(2, 2 * channel, eps=1e-6),
            nn.Conv2d(2 * channel, channel, kernel_size=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
            nn.Sigmoid(),
        )
        self.merger_residual_rate = nn.Parameter(torch.ones(1))

    def _forward_one_step(self, current_state, previous_state, image):
        sigma = self.mamba_sigma(self.cal_kl(current_state, previous_state))
        bias = self.mamba_bias(image)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        uncertainty = []
        current = difficult_zone / torch.exp(-sigma)  # Initial value
        previous = difficult_zone  # Initial value
        for i in range(self.iteration):
            image = image_sequence[:, i, ...]
            current, previous = self._forward_one_step(
                current_state=current,
                previous_state=previous,
                image=image,
            )
            u = self.merger(torch.cat((current, image), dim=1))  # [B, 2C, H, W] -> [B, C, H, W]
            u = self.merger_residual_rate * current + (1 - self.merger_residual_rate) * u
            uncertainty.append(u)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return uncertainty


class UncertaintyEstimatorIterativeMambaErrorEstimationV2(nn.Module):
    def __init__(self, length: int, channel: int, **kwargs):
        super(UncertaintyEstimatorIterativeMambaErrorEstimationV2, self).__init__()
        self.channel = channel
        self.iteration = length

        from ..utils import cal_kl
        self.cal_kl = cal_kl

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=channel, **kwargs)

        from ..utils import LayerNorm2d
        self.feat_dim = 36
        self.merger = nn.Sequential(
            LayerNorm2d(2 * channel),
            nn.Conv2d(2 * channel, self.feat_dim, kernel_size=3, padding='same'),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.feat_dim, channel, kernel_size=1, padding='same'),
            LayerNorm2d(channel),
        )
        self.mamba_adjust = SS2DChanelFirst(d_model=channel, **kwargs)

    def _forward_one_step_error_estimate(self, current_state, previous_state, image):
        sigma = self.mamba_sigma(self.cal_kl(current_state, previous_state))
        bias = self.mamba_bias(image)
        next_state = (current_state / torch.exp(-sigma)) + bias
        return next_state, current_state

    def _forward_uncertainty_estimate(self, state, image):
        features = self.merger(torch.cat((state, image), dim=1))
        weights = torch.sigmoid_(self.mamba_adjust(features)) * 2
        return weights * state

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        uncertainty = []
        current = difficult_zone / torch.exp(-sigma)  # Initial value
        previous = difficult_zone  # Initial value
        for i in range(self.iteration):
            image = image_sequence[:, i, ...]
            current, previous = self._forward_one_step_error_estimate(
                current_state=current,
                previous_state=previous,
                image=image,
            )
            estimated = self._forward_uncertainty_estimate(current, image)
            uncertainty.append(estimated)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return uncertainty