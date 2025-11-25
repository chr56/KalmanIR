import torch
import torch.nn.functional as F


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


def cal_bce_sigmoid(value1, value2) -> torch.Tensor:
    """Calculate Binary Cross Entropy; result's shape remains same."""
    p = torch.sigmoid(value1)
    q = torch.sigmoid(value2)
    return F.binary_cross_entropy(p, q, reduction='none')


def cal_cs(value1, value2, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Calculate Cosine Similarity, taking channel as dimension; result's channel reduces to 1."""
    return F.cosine_similarity(value1, value2, dim, eps).unsqueeze(dim)


def mean_reduce(value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
    """reduce two tensor by mean"""
    return torch.div(torch.add(value1, value2), 2)


def mean_difference(func, value_base: torch.Tensor, value1: torch.Tensor, value2: torch.Tensor) -> torch.Tensor:
    return mean_reduce(
        func(value_base, value1),
        func(value_base, value2),
    )
