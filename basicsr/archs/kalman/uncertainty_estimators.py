import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat


class UncertaintyEstimatorRecursiveConvolutional(nn.Module):
    def __init__(self, channel: int, ):
        super(UncertaintyEstimatorRecursiveConvolutional, self).__init__()
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


class UncertaintyEstimatorRecursiveNarrowMambaBlock(nn.Module):
    def __init__(self, length: int, channel: int, ):
        super(UncertaintyEstimatorRecursiveNarrowMambaBlock, self).__init__()
        self.length = length
        from basicsr.archs.modules_mamba import SS2D
        from .mamba_block import VSSBlockFabric, Mlp
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
        from basicsr.archs.kalman.utils import layer_norm
        self.layer_norm = layer_norm

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
            x = torch.cat((difficult_zone, self.layer_norm(image_sequence[:, i, ...]), sigma), dim=1)
            x = self.channel_compressor(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
            x = self.vss_block(x)
            x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C] -> [B, C, H, W]
            uncertainty.append(x)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorRecursiveWideMambaBlock(nn.Module):
    def __init__(self, length: int, channel: int, ):
        super(UncertaintyEstimatorRecursiveWideMambaBlock, self).__init__()
        self.length = length
        from basicsr.archs.modules_mamba import SS2D
        from .mamba_block import VSSBlockFabric, Mlp
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
        from basicsr.archs.kalman.utils import layer_norm
        self.layer_norm = layer_norm

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
            x = torch.cat((difficult_zone, self.layer_norm(image_sequence[:, i, ...]), sigma), dim=1)
            x = x.permute(0, 2, 3, 1).contiguous()  # [B, C', H, W] -> [B, H, W, C']
            x = self.vss_block(x)
            x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, W, C'] -> [B, C', H, W]
            x = self.channel_compressor(x)
            uncertainty.append(x)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorRecursiveDecoderLayer(nn.Module):
    def __init__(self, channel: int, head: int, patch_size: int = 16):
        super(UncertaintyEstimatorRecursiveDecoderLayer, self).__init__()
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

        difficult_zone, original, padding = _pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, D]
        sigma, _, _ = _pad_and_patch(sigma, self.patch_size)  # [N, P^2, D]

        uncertainty = []
        for i in range(length):
            img_patches, _, _ = _pad_and_patch(image_sequence[:, i, ...], self.patch_size)  # [N, P^2, D]
            u = self.decoder_sigma_merge(img_patches, sigma)
            u = self.decoder_difficult_zone_merge(difficult_zone, u)
            u = _unpatch_and_unpad(u, original, padding, self.patch_size)
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

        image_sequence, original, padding = _pad_and_patch(image_sequence, self.patch_size)  # [N, P^2, D]
        difficult_zone, _, _ = _pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, D]
        sigma, _, _ = _pad_and_patch(sigma, self.patch_size)  # [N, P^2, D]

        u = self.decoder_sigma_merge(image_sequence, sigma)
        u = self.decoder_difficult_zone_merge(difficult_zone, u)

        u = _unpatch_and_unpad(u, original, padding, self.patch_size)  # [B, LC, H, W]
        u = rearrange(u, 'b (l c) h w -> b l c h w', l=length).contiguous()  # [B, L, C, H, W]

        return u


class UncertaintyEstimatorRecursiveCrossAttention(nn.Module):
    def __init__(self, channel: int, head: int, patch_size: int = 16):
        super(UncertaintyEstimatorRecursiveCrossAttention, self).__init__()
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

        difficult_zone, original, padding = _pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, C]
        sigma, _, _ = _pad_and_patch(sigma, self.patch_size)  # [N, P^2, C]

        uncertainty = []
        for i in range(length):
            img, _, _ = _pad_and_patch(image_sequence[:, i, ...], self.patch_size)  # [N, P^2, C]
            u = self.norm_sigma_merge(img)
            u, _ = self.attention_sigma_merge(u, sigma, sigma)  # [B, HW, C]
            u = self.norm_img_merge(u)
            u, _ = self.attention_img_merge(difficult_zone, u, u)  # [B, HW, C]
            u = _unpatch_and_unpad(u, original, padding, self.patch_size)  # [B, C, H, W]
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

        image_sequence, original, padding = _pad_and_patch(image_sequence, self.patch_size)  # [N, P^2, D]
        sigma, _, _ = _pad_and_patch(sigma, self.patch_size)  # [N, P^2, D]
        difficult_zone, _, _ = _pad_and_patch(difficult_zone, self.patch_size)  # [N, P^2, D]

        u = self.norm_sigma_merge(image_sequence)
        u, _ = self.attention_sigma_merge(u, sigma, sigma)
        u = self.norm_difficult_zone_merge(u)
        u, _ = self.attention_difficult_zone_merge(u, difficult_zone, difficult_zone)

        u = _unpatch_and_unpad(u, original, padding, self.patch_size)  # [B, LC, H, W]
        u = rearrange(u, 'b (l c) h w -> b l c h w', l=length).contiguous()  # [B, L, C, H, W]

        return u


def _pad_and_patch(x: torch.Tensor, patch_size: int):
    """
    :param x: shape [B, D, H, W]
    :param patch_size: width and height of patch, P
    :return: shape [N, P^2, D], N = B * (H/P) * (W/P)
    """
    batch_size, _, origin_h, origin_w = x.shape

    pad_h = (patch_size - origin_h % patch_size) % patch_size
    pad_w = (patch_size - origin_w % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

    x_patched = rearrange(x,
                          'b d (hs p1) (ws p2) -> (b hs ws) (p1 p2) d',
                          p1=patch_size, p2=patch_size, )

    return x_patched.contiguous(), (origin_h, origin_w), (pad_h, pad_w)


def _unpatch_and_unpad(
        x_patched: torch.Tensor,
        original_size: tuple,
        padding: tuple, patch_size: int
):
    """
    :param x_patched: shape [N, P^2, D], N = B * (H/P) * (W/P)
    :param original_size: tuple of (H, W)
    :param padding: tuple of (pad_h, pad_w)
    :param patch_size: P
    :return: shape [B, D, H, W]
    """

    origin_h, origin_w = original_size
    pad_h, pad_w = padding

    n, _, d = x_patched.shape
    height = origin_h + pad_h
    width = origin_w + pad_w
    hs = height // patch_size
    ws = width // patch_size
    b = n // (hs * ws)

    x = rearrange(x_patched,
                  '(b hs ws) (p1 p2) d -> b d (hs p1) (ws p2)',
                  b=b, hs=hs, ws=ws, p1=patch_size, p2=patch_size, )

    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :origin_h, :origin_w]

    return x.contiguous()
