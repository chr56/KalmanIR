import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_reduce(value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
    """reduce two tensor by mean"""
    return torch.div(torch.add(value1, value2), 2)


def mean_difference(func, value_base: torch.Tensor, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
    return mean_reduce(
        func(value_base, value1),
        func(value_base, value2),
    )


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


def pad_and_patch(x: torch.Tensor, patch_size: int):
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

    from einops import rearrange
    x_patched = rearrange(x,
                          'b d (hs p1) (ws p2) -> (b hs ws) (p1 p2) d',
                          p1=patch_size, p2=patch_size, )

    return x_patched.contiguous(), (origin_h, origin_w), (pad_h, pad_w)


def unpatch_and_unpad(
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

    from einops import rearrange
    x = rearrange(x_patched,
                  '(b hs ws) (p1 p2) d -> b d (hs p1) (ws p2)',
                  b=b, hs=hs, ws=ws, p1=patch_size, p2=patch_size, )

    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :origin_h, :origin_w]

    return x.contiguous()


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
