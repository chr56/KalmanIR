from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY

# basic
from .primitive_losses import (
    L1Loss, MSELoss, CharbonnierLoss, WeightedTVLoss,
)
from .gan_losses import (
    GANLoss, GANFeatLoss, MultiScaleGANLoss,
    r1_penalty, g_path_regularize, gradient_penalty_loss
)
from .perceptual_losses import (
    PerceptualLoss
)
from .vae_losses import (
    VAELoss
)
# complex
from .losses import (
    FourierLoss, BCELoss, BCEFocalLoss,
    V0_Loss, V1_Loss, V2_Loss, V3_Loss, V5_Loss,
    L1FourierGAN_MixedLoss, LFBG_MixedLoss, LFP_MixedLoss
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
    'L1FourierGAN_MixedLoss', 'LFBG_MixedLoss', 'LFP_MixedLoss',
    'VAELoss'
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
