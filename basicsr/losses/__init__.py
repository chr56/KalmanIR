from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .losses import (
    FourierLoss, BCELoss, BCEFocalLoss,
    V0_Loss, V1_Loss, V2_Loss, V3_Loss, V5_Loss,
    L1FourierGAN_MixedLoss, LFBG_MixedLoss,
)
from .primitive_losses import (
    L1Loss, MSELoss, CharbonnierLoss, WeightedTVLoss,
    PerceptualLoss,
    GANLoss, GANFeatLoss, MultiScaleGANLoss,
    g_path_regularize, gradient_penalty_loss, r1_penalty
)

__all__ = [
    # Original
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss',
    'PerceptualLoss',
    'GANLoss', 'GANFeatLoss', 'MultiScaleGANLoss',
    'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize',
    # Customs
    'FourierLoss', 'BCEFocalLoss', 'BCELoss',
    'V0_Loss', 'V1_Loss', 'V2_Loss', 'V3_Loss', 'V5_Loss',
    'L1FourierGAN_MixedLoss', 'LFBG_MixedLoss'
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
