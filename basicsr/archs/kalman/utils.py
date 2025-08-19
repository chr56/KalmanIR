import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_difference(func, value_base: torch.Tensor, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
    return layer_norm(mean_reduce(
        func(value_base, value1),
        func(value_base, value2),
    ))


def cal_ae(value1, value2) -> torch.Tensor:
    """Calculate Absolute Error"""
    return torch.abs(torch.sub(value1, value2))


def cal_kl(value1, value2) -> torch.Tensor:
    """Calculate Kullback-Leibler divergence; result's shape remains same."""
    p = torch.log_softmax(value1, dim=1)
    q = torch.softmax(value2, dim=1)
    return F.kl_div(p, q, reduction='none')


def cal_bce(value1, value2) -> torch.Tensor:
    """Calculate Binary Cross Entropy; result's shape remains same."""
    p = torch.softmax(value1, dim=1)
    q = torch.softmax(value2, dim=1)
    return F.binary_cross_entropy(p, q, reduction='none')


def cal_cs(value1, value2, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Calculate Cosine Similarity, taking channel as dimension; result's channel reduces to 1."""
    return F.cosine_similarity(value1, value2, dim, eps).unsqueeze(dim)


def mean_reduce(value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
    """reduce two tensor by mean"""
    return torch.div(torch.add(value1, value2), 2)


def layer_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    :param tensor: shape [B, C, H, W]
    :return: normal tensor, shape [B, C, H, W]
    """
    return F.layer_norm(tensor, tensor.shape[1:])


class ChannelCompressor(nn.Module):
    def __init__(self, in_channel: int, out_channel: int = 6, norm_num_groups=None):
        super(ChannelCompressor, self).__init__()
        num_groups = 1 if norm_num_groups is None else norm_num_groups
        self.norm = nn.GroupNorm(num_groups, in_channel, eps=1e-6)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(num_groups, in_channel, eps=1e-6),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(num_groups, in_channel, eps=1e-6),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding='same'),
        )
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.main_branch(x) + self.residual_branch(x)
        return x
