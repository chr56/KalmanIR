from typing import Union, Tuple, Sequence

import torch
from torch import nn as nn

from basicsr.utils import binary_to_decimal, decimal_to_binary
from basicsr.utils.img_util import dump_images
from basicsr.utils.registry import LOSS_REGISTRY
from .perceptual_losses import PerceptualLoss
from .gan_losses import GANLoss
from .primitive_losses import (
    bce_loss, bce_with_logits_loss,
    L1Loss, CharbonnierLoss,
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
        return self.l1(torch.fft.rfft2(pred, dim=(-2, -1)), torch.fft.rfft2(target, dim=(-2, -1)))


@LOSS_REGISTRY.register()
class BCELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(BCELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * bce_loss(pred, target, weight, reduction=self.reduction)


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

        self.dump(
            binary_to_decimal(torch.logit(pt)),
            binary_to_decimal(target),
        )

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
        self.i = self.i + 1
        if self.i % 100 == 0:
            dump_images(sr, hr, save_directory="/data1/hsw/visualization/focal_loss")


@LOSS_REGISTRY.register()
class V0_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', use_focal=False):
        super(V0_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.main_loss_fn = V1_Loss()
        if use_focal:
            self.secondary_loss_fn = BCEFocalLoss()
        else:
            self.secondary_loss_fn = BCELoss()

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        target_decimal = target
        x_decimal_2, x_decimal_1, x_binary_1 = pred
        target_binary = decimal_to_binary((target_decimal * 255.))
        pred = [x_decimal_2, x_decimal_1]
        loss_main = self.main_loss_fn(pred, target_decimal)
        loss_secondary = self.secondary_loss_fn(x_binary_1, target_binary)
        return loss_main + 0.02 * loss_secondary


@LOSS_REGISTRY.register()
class V1_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', use_charbonnier=False):
        super(V1_Loss, self).__init__()
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
        target_decimal = target
        pred_decimal_2, pred_decimal_1, pred_binary_1 = pred
        target_binary = decimal_to_binary((target_decimal * 255.))
        bce = bce_with_logits_loss(pred_binary_1, target_binary)
        l1_1 = self.l1(pred_decimal_1, target_decimal)
        l1_2 = self.l1(pred_decimal_2, target_decimal)
        return l1_2 + 0.2 * l1_1 + 0.1 * bce


@LOSS_REGISTRY.register()
class V2_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', use_charbonnier=False):
        super(V2_Loss, self).__init__()
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
        target_decimal = target
        pred_decimal_2, pred_decimal_1, pred_binary_1 = pred
        target_binary = decimal_to_binary((target_decimal * 255.))
        l1_1_2 = self.l1(pred_binary_1, target_binary)
        l1_1 = self.l1(torch.fft.rfft2(pred_decimal_1, dim=(-2, -1)),
                       torch.fft.rfft2(target_decimal, dim=(-2, -1)))
        l1_2 = self.l1(pred_decimal_2, target_decimal)
        return l1_2 + 0.2 * l1_1 + 0.1 * l1_1_2


@LOSS_REGISTRY.register()
class V3_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', use_charbonnier=False):
        super(V3_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

        if use_charbonnier:
            self.l1 = CharbonnierLoss()
        else:
            self.l1 = L1Loss()

        self.fourier = FourierLoss()

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        x_refine, x_L1, x_Fourier = pred
        loss1_L1 = self.l1(x_L1, target)
        loss1_fourier = self.fourier(x_Fourier, target)
        loss2_refine = self.l1(x_refine, target)
        return loss2_refine + 0.2 * loss1_L1 + 0.1 * loss1_fourier


@LOSS_REGISTRY.register()
class V5_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', use_charbonnier=False):
        super(V5_Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        if use_charbonnier:
            self.l1 = CharbonnierLoss()
        else:
            self.l1 = L1Loss()
        self.fourier = FourierLoss()

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        x_refine, x_L1, x_Fourier = pred
        loss1_L1 = self.l1(x_L1, target)
        loss1_fourier = self.fourier(x_Fourier, target)
        loss2_refine = self.l1(x_refine, target)
        return loss2_refine + 0.2 * loss1_L1 + 0.1 * loss1_fourier


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
class L1FourierGAN_MixedLoss(nn.Module):
    """
    Mixed loss:
    - L1 Loss
    - Fourier L1 Loss
    - GAN Loss
    """

    def __init__(self,
                 weights,
                 binaries,
                 l1_reduction: str = 'mean',
                 fourier_l1_reduction: str = 'mean',
                 gan_type: str = 'vanilla',
                 gan_real_label_val=1.0,
                 gan_fake_label_val=0.0,
                 ):
        super(L1FourierGAN_MixedLoss, self).__init__()

        self.l1_weights = weights[0]
        self.fourier_l1_weights = weights[1]
        self.gan_weights = weights[2]

        assert not isinstance(binaries, str), "`binaries` must not be a string"
        assert hasattr(binaries, "__getitem__"), "`binaries` must be a sequence"
        assert len(binaries) > 0 and type(binaries[0]) is bool, "content of `binaries` must be boolean"
        if len(binaries) == 3:
            self.three_way_prediction_output = True
            self.should_convert_binaries = binaries
        elif len(binaries) == 1:
            self.three_way_prediction_output = False
            self.should_convert_binary = binaries[0]
        else:
            raise ValueError(f"`binaries` must has length 1 or 3 but got {len(binaries)}")

        self.l1_reduction = l1_reduction
        self.fourier_l1_reduction = fourier_l1_reduction
        self.gan_type = gan_type
        self.gan_real_label_val = gan_real_label_val
        self.gan_fake_label_val = gan_fake_label_val

        self.l1 = L1Loss(reduction=l1_reduction)
        self.fourier_l1 = FourierWrapper(L1Loss(reduction=fourier_l1_reduction))
        self.gan_loss = GANLoss(gan_type, gan_real_label_val, gan_fake_label_val)

    def forward(self,
                pred: Union[Sequence[torch.Tensor], torch.Tensor],
                target: torch.Tensor,
                weight=None, **kwargs):
        """
        :param pred: Predicted results, one or multiple, of shape (N, C, H, W)
        :param target: Ground truth, of shape (N, C, H, W)
        :param weight: Element-wise weights, of shape (N, C, H, W).
        :param kwargs: (unused)
        :return: mixed loss
        """

        if self.three_way_prediction_output:
            r, a, b = pred
            if self.should_convert_binaries[0]: r = binary_to_decimal(r)
            if self.should_convert_binaries[1]: a = binary_to_decimal(a)
            if self.should_convert_binaries[2]: b = binary_to_decimal(b)
            loss_l1 = self.l1(r, target, weight)
            loss_fourier_l1 = self.fourier_l1(a, target)
            loss_gan = self.gan_loss(b, False)  # + self.gan_loss(target, True)
        else:
            if self.should_convert_binary: pred = binary_to_decimal(pred)
            loss_l1 = self.l1(pred, target, weight)
            loss_fourier_l1 = self.fourier_l1(pred, target)
            loss_gan = self.gan_loss(pred, False)  # + self.gan_loss(target, True)
            pass

        loss_total = (
                self.l1_weights * loss_l1 +
                self.fourier_l1_weights * loss_fourier_l1 +
                self.gan_weights * loss_gan
        )
        return loss_total

@LOSS_REGISTRY.register()
class LFBG_MixedLoss(nn.Module):
    """
    Mixed loss:
    - L1 Loss
    - Fourier L1 Loss
    - BCE Loss with GAN Loss
    """

    def __init__(self,
                 weights,
                 binaries,
                 l1_reduction: str = 'mean',
                 fourier_l1_reduction: str = 'mean',
                 bce_reduction: str = 'mean',
                 gan_type: str = 'vanilla',
                 gan_real_label_val=1.0,
                 gan_fake_label_val=0.0,
                 ):
        super(LFBG_MixedLoss, self).__init__()

        self.l1_weights = weights[0]
        self.fourier_l1_weights = weights[1]
        self.bce_weights = weights[2]
        self.gan_weights = weights[3]

        assert not isinstance(binaries, str), "`binaries` must not be a string"
        assert hasattr(binaries, "__getitem__"), "`binaries` must be a sequence"
        assert len(binaries) > 0 and type(binaries[0]) is bool, "content of `binaries` must be boolean"
        if len(binaries) == 3:
            self.three_way_prediction_output = True
            self.should_convert_binaries = binaries
        elif len(binaries) == 1:
            self.three_way_prediction_output = False
            self.should_convert_binary = binaries[0]
        else:
            raise ValueError(f"`binaries` must has length 1 or 3 but got {len(binaries)}")

        self.l1_reduction = l1_reduction
        self.fourier_l1_reduction = fourier_l1_reduction
        self.gan_type = gan_type
        self.gan_real_label_val = gan_real_label_val
        self.gan_fake_label_val = gan_fake_label_val

        self.l1 = L1Loss(reduction=l1_reduction)
        self.fourier_l1 = FourierWrapper(L1Loss(reduction=fourier_l1_reduction))

        self.bce = BCELoss(reduction=bce_reduction)
        self.gan_loss = GANLoss(gan_type, gan_real_label_val, gan_fake_label_val)

    def forward(self,
                pred: Union[Sequence[torch.Tensor], torch.Tensor],
                target: torch.Tensor,
                weight=None, **kwargs):
        """
        :param pred: Predicted results, one or multiple, of shape (N, C, H, W)
        :param target: Ground truth, of shape (N, C, H, W)
        :param weight: Element-wise weights, of shape (N, C, H, W).
        :param kwargs: (unused)
        :return: mixed loss
        """

        if self.three_way_prediction_output:
            r, a, b = pred
            if self.should_convert_binaries[0]: r = binary_to_decimal(r)
            if self.should_convert_binaries[1]: a = binary_to_decimal(a)
            if self.should_convert_binaries[2]: b = binary_to_decimal(b)
            loss_l1 = self.l1(r, target, weight)
            loss_fourier_l1 = self.fourier_l1(a, target)
            loss_bce = self.bce(b, target)
            loss_gan = self.gan_loss(b, True)
        else:
            if self.should_convert_binary: pred = binary_to_decimal(pred)
            loss_l1 = self.l1(pred, target, weight)
            loss_fourier_l1 = self.fourier_l1(pred, target)
            loss_bce = self.bce(pred, target)
            loss_gan = self.gan_loss(pred, True)
            pass

        loss_total = (
                self.l1_weights * loss_l1 +
                self.fourier_l1_weights * loss_fourier_l1 +
                self.bce_weights * loss_bce +
                self.gan_weights * loss_gan
        )
        return loss_total


@LOSS_REGISTRY.register()
class LFP_MixedLoss(nn.Module):
    """
    Mixed loss:
    - L1 Loss
    - Fourier L1 Loss
    - Perceptual Loss
    """

    def __init__(self,
                 weights,
                 binaries,
                 l1_reduction: str = 'mean',
                 fourier_l1_reduction: str = 'mean',
                 perceptual_vgg_arch: str = 'vgg19',
                 perceptual_layer: str = 'relu3_3',
                 perceptual_criterion: str = 'l1',
                 ):
        super(LFP_MixedLoss, self).__init__()

        self.l1_weights = weights[0]
        self.fourier_l1_weights = weights[1]
        self.gan_weights = weights[2]

        assert not isinstance(binaries, str), "`binaries` must not be a string"
        assert hasattr(binaries, "__getitem__"), "`binaries` must be a sequence"
        assert len(binaries) > 0 and type(binaries[0]) is bool, "content of `binaries` must be boolean"
        if len(binaries) == 3:
            self.three_way_prediction_output = True
            self.should_convert_binaries = binaries
        elif len(binaries) == 1:
            self.three_way_prediction_output = False
            self.should_convert_binary = binaries[0]
        else:
            raise ValueError(f"`binaries` must has length 1 or 3 but got {len(binaries)}")

        self.l1_reduction = l1_reduction
        self.fourier_l1_reduction = fourier_l1_reduction

        self.perceptual_vgg_arch = perceptual_vgg_arch
        self.perceptual_layer = perceptual_layer
        self.perceptual_criterion = perceptual_criterion

        self.l1 = L1Loss(reduction=l1_reduction)
        self.fourier_l1 = FourierWrapper(L1Loss(reduction=fourier_l1_reduction))
        self.perceptual = PerceptualLoss(
            layer_weights={perceptual_layer: 1},
            vgg_type=perceptual_vgg_arch,
            criterion=perceptual_criterion,
            perceptual_weight=1.0,
            style_weight=0., # disable stype loss
        )

    def forward(self,
                pred: Union[Sequence[torch.Tensor], torch.Tensor],
                target: torch.Tensor,
                weight=None, **kwargs):
        """
        :param pred: Predicted results, one or multiple, of shape (N, C, H, W)
        :param target: Ground truth, of shape (N, C, H, W)
        :param weight: Element-wise weights, of shape (N, C, H, W).
        :param kwargs: (unused)
        :return: mixed loss
        """

        if self.three_way_prediction_output:
            r, a, b = pred
            if self.should_convert_binaries[0]: r = binary_to_decimal(r)
            if self.should_convert_binaries[1]: a = binary_to_decimal(a)
            if self.should_convert_binaries[2]: b = binary_to_decimal(b)
            loss_l1 = self.l1(r, target, weight)
            loss_fourier_l1 = self.fourier_l1(a, target)
            loss_perceptual, _ = self.perceptual(b, target)
        else:
            if self.should_convert_binary: pred = binary_to_decimal(pred)
            loss_l1 = self.l1(pred, target, weight)
            loss_fourier_l1 = self.fourier_l1(pred, target)
            loss_perceptual, _ = self.perceptual(pred, target)
            pass

        loss_total = (
                self.l1_weights * loss_l1 +
                self.fourier_l1_weights * loss_fourier_l1 +
                self.gan_weights * loss_perceptual
        )
        return loss_total
