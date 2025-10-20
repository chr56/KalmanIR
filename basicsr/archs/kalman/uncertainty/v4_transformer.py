import torch
from einops import rearrange, repeat
from torch import nn

from ..utils import unpatch_and_unpad, pad_and_patch


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
