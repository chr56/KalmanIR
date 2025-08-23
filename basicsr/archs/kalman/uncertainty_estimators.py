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


class UncertaintyEstimatorRecursiveDecoderLayer(nn.Module):
    def __init__(self, channel: int, head: int):
        super(UncertaintyEstimatorRecursiveDecoderLayer, self).__init__()

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

        difficult_zone = rearrange(difficult_zone, 'b c h w -> b (h w) c').contiguous()  # [B, HW, C]
        sigma = rearrange(sigma, 'b c h w -> b (h w) c')  # [B, HW, C]

        uncertainty = []
        for i in range(length):
            img = image_sequence[:, i, ...]  # [B, C, H, W]
            img = rearrange(img, 'b c h w -> b (h w) c').contiguous()  # [B, HW, C]
            u = self.decoder_sigma_merge(img, sigma)
            u = self.decoder_difficult_zone_merge(difficult_zone, u)
            u = rearrange(u, 'b (h w) c -> b c h w', h=height).contiguous()  # [B, C, H, W]
            uncertainty.append(u)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorOneDecoderLayer(nn.Module):
    def __init__(self, length: int, channel: int, head: int):
        super(UncertaintyEstimatorOneDecoderLayer, self).__init__()

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
        :return: estimated x, shape [B, L, C, H, W]
        """

        batch, length, channel, height, width = image_sequence.shape

        image_sequence = rearrange(image_sequence, 'b l c h w -> b (h w) (l c)').contiguous()  # [B, HW, LC]
        difficult_zone = repeat(difficult_zone, 'b c h w -> b (l c) h w', l=length)
        difficult_zone = rearrange(difficult_zone, 'b d h w -> b (h w) d').contiguous()  # [B, HW, LC]
        sigma = repeat(sigma, 'b c h w -> b (l c) h w', l=length)
        sigma = rearrange(sigma, 'b d h w -> b (h w) d').contiguous()  # [B, HW, LC]

        x = self.decoder_sigma_merge(image_sequence, sigma)
        x = self.decoder_difficult_zone_merge(difficult_zone, x)

        x = rearrange(x, 'b (h w) (l c) -> b l c h w', h=height, l=length).contiguous()

        return x


class UncertaintyEstimatorRecursiveCrossAttention(nn.Module):
    def __init__(self, channel: int, head: int):
        super(UncertaintyEstimatorRecursiveCrossAttention, self).__init__()

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

        difficult_zone = rearrange(difficult_zone, 'b c h w -> b (h w) c').contiguous()  # [B, HW, C]
        sigma = rearrange(sigma, 'b c h w -> b (h w) c')  # [B, HW, C]

        uncertainty = []
        for i in range(length):
            img = image_sequence[:, i, ...]  # [B, C, H, W]
            img = rearrange(img, 'b c h w -> b (h w) c').contiguous()  # [B, HW, C]
            u = self.norm_sigma_merge(img)
            u, _ = self.attention_sigma_merge(u, sigma, sigma)  # [B, HW, C]
            u = self.norm_img_merge(u)
            u, _ = self.attention_img_merge(difficult_zone, u, u)  # [B, HW, C]
            u = rearrange(u, 'b (h w) c -> b c h w', h=height).contiguous()  # [B, C, H, W]
            uncertainty.append(u)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty


class UncertaintyEstimatorOneCrossAttention(nn.Module):
    def __init__(self, length: int, channel: int, head: int):
        super(UncertaintyEstimatorOneCrossAttention, self).__init__()

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
        :return: estimated x, shape [B, L, C, H, W]
        """

        batch, length, channel, height, width = image_sequence.shape

        image_sequence = rearrange(image_sequence, 'b l c h w -> b (h w) (l c)').contiguous()  # [B, HW, LC]
        difficult_zone = rearrange(difficult_zone, 'b c h w -> b (h w) c').contiguous()  # [B, HW, C]
        sigma = rearrange(sigma, 'b c h w -> b (h w) c').contiguous()  # [B, HW, C]

        x = self.norm_sigma_merge(image_sequence)
        x, _ = self.attention_sigma_merge(x, sigma, sigma)
        x = self.norm_difficult_zone_merge(x)
        x, _ = self.attention_difficult_zone_merge(x, difficult_zone, difficult_zone)

        x = rearrange(x, 'b (h w) (l c) -> b l c h w', h=height, l=length).contiguous()

        return x
