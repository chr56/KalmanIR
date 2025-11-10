import math

import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

from .gan_losses import GANLoss


@LOSS_REGISTRY.register()
class IntegratedGANLoss(nn.Module):
    def __init__(
            self,
            gan_type,
            discriminator: dict,
            discriminator_weights: str,
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1.0,
    ):
        super(IntegratedGANLoss, self).__init__()
        self.discriminator = _load_discriminator(discriminator, discriminator_weights)
        self.loss_fn = GANLoss(gan_type, real_label_val, fake_label_val, loss_weight)

    def move_to_device(self, device):
        self.discriminator = self.discriminator.to(device)

    def parallel(self, distributed: bool, data_parallel: bool, find_unused_parameters: bool = False):
        if distributed:
            self.discriminator = nn.parallel.DistributedDataParallel(
                self.discriminator,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters
            )
        elif data_parallel:
            self.discriminator = nn.DataParallel(self.discriminator)
        else:
            if isinstance(self.discriminator, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.discriminator = self.discriminator.module

    def forward(self, output, target_is_real=True):
        d_out = self.discriminator(output)
        loss = self.loss_fn(d_out, target_is_real)
        return loss, d_out


def _load_discriminator(discriminator_opt: dict, weights_path: str, param_key: str = 'params') -> nn.Module:
    from copy import deepcopy
    from basicsr.archs import build_network

    # build discriminator from options dict
    discriminator = build_network(discriminator_opt)

    # read weights
    load_net = torch.load(weights_path, weights_only=True, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
        load_net = load_net[param_key]

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)

    # load weights
    discriminator.load_state_dict(load_net, strict=True)
    for p in discriminator.parameters():
        p.requires_grad = False

    return discriminator
