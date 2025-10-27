import torch
from torch import nn as nn
from typing_extensions import Literal

from basicsr.utils.registry import LOSS_REGISTRY

ReconType = Literal['l1', 'l2', 'mse', 'bce']


@LOSS_REGISTRY.register()
class VAELoss(nn.Module):
    def __init__(
            self,
            loss_weight=1,
            reduction='mean',
            recon_type: ReconType = 'mse',
            kl_weight=0.00005,
    ):
        super(VAELoss, self).__init__()
        self.loss_weight = loss_weight

        if recon_type == 'mse' or recon_type == 'l2':
            self.cri_re = nn.MSELoss(reduction=reduction)
        elif recon_type == 'bce':
            self.cri_re = nn.BCELoss(reduction=reduction)
        elif recon_type == 'l1':
            self.cri_re = nn.L1Loss(reduction=reduction)
        else:
            raise NotImplementedError(f"recon_type {recon_type} not implemented")

        self.kl_weight = kl_weight

    def forward(self, output, mu, var, gt):

        re_loss = self.cri_re(output, gt)
        kl_loss = 0.5 * torch.sum(-1 - var + mu.pow(2) + var.exp())

        return self.loss_weight * (re_loss + self.kl_weight * kl_loss)
