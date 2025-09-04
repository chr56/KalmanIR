import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from .utils import unpatch_and_unpad, pad_and_patch


def build_uncertainty_estimator_for_v4(mode, dim, seq_length) -> nn.Module:
    if mode == "iterative_convolutional":
        return UncertaintyEstimatorIterativeConvolutional(dim)
    elif mode == "iterative_narrow_mamba_block":
        return UncertaintyEstimatorIterativeNarrowMambaBlock(seq_length, dim)
    elif mode == "iterative_wide_mamba_block":
        return UncertaintyEstimatorIterativeWideMambaBlock(seq_length, dim)
    elif mode == "one_decoder_layer":
        return UncertaintyEstimatorOneDecoderLayer(seq_length, dim, seq_length)
    elif mode == "iterative_decoder_layer":
        return UncertaintyEstimatorIterativeDecoderLayer(dim, seq_length)
    elif mode == "one_cross_attention":
        return UncertaintyEstimatorOneCrossAttention(seq_length, dim, seq_length)
    elif mode == "iterative_cross_attention":
        return UncertaintyEstimatorIterativeCrossAttention(dim, 3)
    elif mode == "iterative_convolutional_cross_attention":
        return UncertaintyEstimatorIterativeConvolutionalCrossAttention(dim, 3)
    elif mode == "iterative_mamba_error_estimation":
        return UncertaintyEstimatorIterativeMambaErrorEstimation(seq_length, dim)
    elif mode == "iterative_mamba_error_estimation_v2":
        return UncertaintyEstimatorIterativeMambaErrorEstimationV2(seq_length, dim)
    else:
        if mode:
            import warnings
            warnings.warn(f"Unknown uncertainty estimator mode `{mode}`, using default!")
        return UncertaintyEstimatorIterativeConvolutional(dim)


class UncertaintyEstimatorIterativeConvolutional(nn.Module):
    def __init__(self, channel: int, ):
        super(UncertaintyEstimatorIterativeConvolutional, self).__init__()
        from .convolutional_res_block import ConvolutionalResBlock
        self.block = ConvolutionalResBlock(
            3 * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        length = image_sequence.shape[1]

        uncertainty = []
        for i in range(length):
            x = torch.cat((difficult_zone, sigma, image_sequence[:, i, ...]), dim=1)
            x = self.block(x)
            uncertainty.append(x)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


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
        from .convolutional_res_block import ConvolutionalResBlock
        self.channel_compressor = ConvolutionalResBlock(
            length * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )
        from .utils import LayerNorm2d
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
        from .convolutional_res_block import ConvolutionalResBlock
        self.channel_compressor = ConvolutionalResBlock(
            length * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )
        from .utils import LayerNorm2d
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


class UncertaintyEstimatorIterativeDecoderLayer(nn.Module):
    def __init__(self, channel: int, head: int, patch_size: int = 16):
        super(UncertaintyEstimatorIterativeDecoderLayer, self).__init__()
        self.patch_size = patch_size

        self.decoder_sigma_merge = nn.TransformerDecoderLayer(
            d_model=channel, nhead=head, dim_feedforward=channel * 4,
            batch_first=True, norm_first=True, dropout=0.04
        )

        self.decoder_difficult_zone_merge = nn.TransformerDecoderLayer(
            d_model=channel, nhead=head, dim_feedforward=channel * 4,
            batch_first=True, norm_first=True, dropout=0.04,
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        batch, length, channel, height, width = image_sequence.shape

        difficult_zone, original, padding = pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, D]
        sigma, _, _ = pad_and_patch(sigma, self.patch_size)  # [N, P^2, D]

        uncertainty = []
        for i in range(length):
            img_patches, _, _ = pad_and_patch(image_sequence[:, i, ...], self.patch_size)  # [N, P^2, D]
            u = self.decoder_sigma_merge(img_patches, sigma)
            u = self.decoder_difficult_zone_merge(difficult_zone, u)
            u = unpatch_and_unpad(u, original, padding, self.patch_size)
            uncertainty.append(u)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorOneDecoderLayer(nn.Module):
    def __init__(self, length: int, channel: int, head: int, patch_size: int = 16):
        super(UncertaintyEstimatorOneDecoderLayer, self).__init__()
        self.patch_size = patch_size
        self.length = length

        self.decoder_sigma_merge = nn.TransformerDecoderLayer(
            d_model=length * channel, nhead=head, dim_feedforward=length * channel * 4,
            batch_first=True, norm_first=True, dropout=0.04
        )

        self.decoder_difficult_zone_merge = nn.TransformerDecoderLayer(
            d_model=length * channel, nhead=head, dim_feedforward=length * channel * 4,
            batch_first=True, norm_first=True, dropout=0.04
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        batch, length, channel, height, width = image_sequence.shape

        image_sequence = rearrange(image_sequence, 'b l c h w -> b (l c) h w').contiguous()  # [B, LC, H, W]
        difficult_zone = repeat(difficult_zone, 'b c h w -> b (l c) h w', l=length).contiguous()  # [B, LC, H, W]
        sigma = repeat(sigma, 'b c h w -> b (l c) h w', l=length).contiguous()  # [B, LC, H, W]

        image_sequence, original, padding = pad_and_patch(image_sequence, self.patch_size)  # [N, P^2, D]
        difficult_zone, _, _ = pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, D]
        sigma, _, _ = pad_and_patch(sigma, self.patch_size)  # [N, P^2, D]

        u = self.decoder_sigma_merge(image_sequence, sigma)
        u = self.decoder_difficult_zone_merge(difficult_zone, u)

        u = unpatch_and_unpad(u, original, padding, self.patch_size)  # [B, LC, H, W]
        u = rearrange(u, 'b (l c) h w -> b l c h w', l=length).contiguous()  # [B, L, C, H, W]

        return u


class UncertaintyEstimatorIterativeConvolutionalCrossAttention(nn.Module):
    def __init__(self, channel: int, head: int, patch_size: int = 16):
        super(UncertaintyEstimatorIterativeConvolutionalCrossAttention, self).__init__()
        self.patch_size = patch_size

        self.attention_sigma_merge = nn.MultiheadAttention(
            embed_dim=channel, num_heads=head, batch_first=True,
        )
        self.norm_sigma_merge = nn.LayerNorm(channel)

        self.attention_img_merge = nn.MultiheadAttention(
            embed_dim=channel, num_heads=head, batch_first=True,
        )
        self.norm_img_merge = nn.LayerNorm(channel)

        self.cnn_post = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=5, padding='same'),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding='same'),
            nn.Sigmoid(),
        )
        self.residual_rate = nn.Parameter(torch.ones(1))

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        batch, length, channel, height, width = image_sequence.shape

        difficult_zone_patched, original, padding = pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, C]
        sigma_patched, _, _ = pad_and_patch(sigma, self.patch_size)  # [N, P^2, C]

        uncertainty = []
        for i in range(length):
            img, _, _ = pad_and_patch(image_sequence[:, i, ...], self.patch_size)  # [N, P^2, C]
            u = self.norm_sigma_merge(img)
            u, _ = self.attention_sigma_merge(u, sigma_patched, sigma_patched)  # [B, HW, C]
            u = self.norm_img_merge(u)
            u, _ = self.attention_img_merge(difficult_zone_patched, u, u)  # [B, HW, C]
            u = unpatch_and_unpad(u, original, padding, self.patch_size)  # [B, C, H, W]
            u = self.cnn_post(u) * (1 - self.residual_rate) + difficult_zone * self.residual_rate
            uncertainty.append(u)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorIterativeCrossAttention(nn.Module):
    def __init__(self, channel: int, head: int, patch_size: int = 16):
        super(UncertaintyEstimatorIterativeCrossAttention, self).__init__()
        self.patch_size = patch_size

        self.attention_sigma_merge = nn.MultiheadAttention(
            embed_dim=channel, num_heads=head, batch_first=True,
        )
        self.norm_sigma_merge = nn.LayerNorm(channel)

        self.attention_img_merge = nn.MultiheadAttention(
            embed_dim=channel, num_heads=head, batch_first=True,
        )
        self.norm_img_merge = nn.LayerNorm(channel)

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        batch, length, channel, height, width = image_sequence.shape

        difficult_zone, original, padding = pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, C]
        sigma, _, _ = pad_and_patch(sigma, self.patch_size)  # [N, P^2, C]

        uncertainty = []
        for i in range(length):
            img, _, _ = pad_and_patch(image_sequence[:, i, ...], self.patch_size)  # [N, P^2, C]
            u = self.norm_sigma_merge(img)
            u, _ = self.attention_sigma_merge(u, sigma, sigma)  # [B, HW, C]
            u = self.norm_img_merge(u)
            u, _ = self.attention_img_merge(difficult_zone, u, u)  # [B, HW, C]
            u = unpatch_and_unpad(u, original, padding, self.patch_size)  # [B, C, H, W]
            uncertainty.append(u)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorOneCrossAttention(nn.Module):
    def __init__(self, length: int, channel: int, head: int, patch_size: int = 16):
        super(UncertaintyEstimatorOneCrossAttention, self).__init__()
        self.patch_size = patch_size

        self.length = length

        self.attention_sigma_merge = nn.MultiheadAttention(
            embed_dim=length * channel, kdim=channel, vdim=channel,
            num_heads=head, batch_first=True,
        )
        self.norm_sigma_merge = nn.LayerNorm(length * channel)

        self.attention_difficult_zone_merge = nn.MultiheadAttention(
            embed_dim=length * channel, kdim=channel, vdim=channel,
            num_heads=head, batch_first=True,
        )
        self.norm_difficult_zone_merge = nn.LayerNorm(length * channel)

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        batch, length, channel, height, width = image_sequence.shape

        image_sequence = rearrange(image_sequence, 'b l c h w -> b (l c) h w').contiguous()  # [B, LC, H, W]

        image_sequence, original, padding = pad_and_patch(image_sequence, self.patch_size)  # [N, P^2, D]
        sigma, _, _ = pad_and_patch(sigma, self.patch_size)  # [N, P^2, D]
        difficult_zone, _, _ = pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, D]

        u = self.norm_sigma_merge(image_sequence)
        u, _ = self.attention_sigma_merge(u, sigma, sigma)
        u = self.norm_difficult_zone_merge(u)
        u, _ = self.attention_difficult_zone_merge(u, difficult_zone, difficult_zone)

        u = unpatch_and_unpad(u, original, padding, self.patch_size)  # [B, LC, H, W]
        u = rearrange(u, 'b (l c) h w -> b l c h w', l=length).contiguous()  # [B, L, C, H, W]

        return u


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

        from .utils import cal_kl
        self.cal_kl = cal_kl

        from basicsr.archs.modules_mamba import SS2DChanelFirst
        self.mamba_sigma = SS2DChanelFirst(d_model=channel, **kwargs)
        self.mamba_bias = SS2DChanelFirst(d_model=channel, **kwargs)

        from .utils import LayerNorm2d
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