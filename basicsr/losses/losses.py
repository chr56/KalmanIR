from typing import Union, Tuple, Sequence, Dict, Any

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils import binary_to_decimal, decimal_to_binary
from basicsr.utils.registry import LOSS_REGISTRY
from .perceptual_losses import PerceptualLoss
from .gan_losses import GANLoss
from .primitive_losses import (
    bce_loss, bce_with_logits_loss,
    L1Loss, CharbonnierLoss, MSELoss,
)

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class FourierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', use_charbonnier=False):
        super(FourierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        if use_charbonnier:
            self.l1 = CharbonnierLoss()
        else:
            self.l1 = L1Loss()

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * self.l1(torch.fft.rfft2(pred, dim=(-2, -1)), torch.fft.rfft2(target, dim=(-2, -1)))


@LOSS_REGISTRY.register()
class BCELoss(nn.Module):
    def __init__(self, with_sigmoid: bool = False, loss_weight=1.0, reduction='mean'):
        super(BCELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.with_sigmoid = with_sigmoid

    def forward(self, pred, target, weight=None, **kwargs):
        if self.with_sigmoid:
            return self.loss_weight * bce_with_logits_loss(pred, target, reduction=self.reduction)
        else:
            return self.loss_weight * bce_loss(pred, target, reduction=self.reduction)


@LOSS_REGISTRY.register()
class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean', eps=1e-9):
        """
        二分类 Focal Loss
        :param gamma: 调节因子 (让模型更关注困难样本)
        :param alpha: 类别平衡因子 (用于调整正负样本权重)
        :param reduction: 'mean', 'sum' or 'none'
        """
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        self.eps = eps

        self.i = 0

    def forward(self, predict, target):
        """
        :param predict: 预测的 logits (未经过 sigmoid)，形状 (B, C, H, W)
        :param target: 真实标签 (0 或 1)，形状 (B, C, H, W)
        """
        pt = torch.clamp(predict, min=self.eps, max=1.0 - self.eps)  # 限制范围防止数值问题 log(0)

        # self.dump(
        #     binary_to_decimal(torch.logit(pt)),
        #     binary_to_decimal(target),
        # )

        # 计算 Focal Loss
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        # 选择 reduction 方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def dump(self, sr, hr):
        from basicsr.utils.img_util import dump_images
        self.i = self.i + 1
        if self.i % 100 == 0:
            dump_images(sr, hr, save_directory="results/visualization/focal_loss")


class FourierWrapper(nn.Module):
    def __init__(self,
                 wrapped_loss: nn.Module,
                 require_transform_weight: bool = False,
                 ):
        super(FourierWrapper, self).__init__()
        self.wrapped_loss = wrapped_loss
        self.require_transform_weight = require_transform_weight

    def forward(self, pred, target, weight=None, **kwargs):
        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1))
        target_freq = torch.fft.rfft2(target, dim=(-2, -1))
        if self.require_transform_weight:
            weight_freq = torch.fft.rfft2(weight, dim=(-2, -1))
            return self.wrapped_loss(pred_freq, target_freq, weight_freq, **kwargs)
        else:
            return self.wrapped_loss(pred_freq, target_freq, weight, **kwargs)


@LOSS_REGISTRY.register()
class DifficultZoneReconstructionNoiseLoss(nn.Module):
    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 mode: str = 'l1',
                 preprocess: str = 'none',
                 **kwargs,
                 ):
        super(DifficultZoneReconstructionNoiseLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.mode = mode
        if mode == 'l1':
            self.loss_fn = L1Loss(reduction=self.reduction)
        elif mode == 'l2' or mode == 'mse':
            self.loss_fn = MSELoss(reduction=self.reduction)
        elif mode == 'bce':
            self.loss_fn = BCELoss(reduction=self.reduction, with_sigmoid=True)
        elif mode == 'fourier':
            self.loss_fn = FourierLoss(reduction=self.reduction)
        else:
            raise ValueError(f'Unsupported mode: {mode}.')

        self.preprocess = preprocess
        assert preprocess in _SUPPORTED_PREPROCESS, f"Unsupported preprocess: {self.preprocess}."

    def preprocess_residual(self, item: torch.Tensor):
        if self.preprocess == 'none':
            return item
        elif self.preprocess == 'sin':
            return torch.sin(item)
        elif self.preprocess == 'sigmoid':
            return torch.sigmoid(item)
        elif self.preprocess == 'tanh':
            return torch.tanh(item)
        elif self.preprocess == 'layer_norm':
            from basicsr.archs.modules.layer_norm import layer_norm_2d
            return layer_norm_2d(item, (item.shape[1],))
        else:
            raise ValueError(f"Unsupported preprocess: {self.preprocess}.")

    def forward(self, bundle: Dict[str, Any], **kwargs):
        difficult_zone = bundle['difficult_zone']
        image_sr = bundle['sr_refined'].detach()
        image_hr = bundle['gt']

        residual = torch.abs(image_sr - image_hr)
        residual = self.preprocess_residual(residual)
        loss = self.loss_fn(difficult_zone, residual)

        return self.loss_weight * loss


_SUPPORTED_PREPROCESS = ['none', 'sigmoid', 'tanh', 'sin', 'layer_norm']
