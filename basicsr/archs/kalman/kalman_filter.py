from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm
from einops import rearrange


class KalmanFilter(nn.Module):
    """
    Perform a Kalman filter.
    (borrow from KEEP: https://github.com/jnjaby/KEEP)
    """

    def __init__(self,
                 emb_dim: int,
                 image_patch: int = 1,
                 uncertainty_estimator: nn.Module = None,
                 ):
        super().__init__()

        assert image_patch >= 1, "Image patch must be at least 1"
        self.image_patch = image_patch

        self.uncertainty_estimator = uncertainty_estimator

        self.kalman_gain_calculator = nn.Sequential(
            CNNResBlock(emb_dim, emb_dim),
            CNNResBlock(emb_dim, emb_dim),
            CNNResBlock(emb_dim, emb_dim),
            nn.Conv2d(emb_dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.predictor = nn.Sequential(
            CNNResBlock(emb_dim, emb_dim),
            CNNResBlock(emb_dim, emb_dim),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def predict(self, z_hat):
        """
        Predict the next state based on the current state
        :param z_hat: Shape [Batch, Channel, Height, Weight]
        :return: Shape [Batch, Channel, Height, Weight]
        """
        z_prime = self.predictor(z_hat)
        return z_prime

    def update(self, z_code, z_prime, kalman_gain):
        """
        Update the state and uncertainty based on the measurement and Kalman gain
        :param z_code: original z, Shape [Batch, Channel, Height, Weight]
        :param z_prime: delta z, Shape [Batch, Channel, Height, Weight]
        :param kalman_gain: calculated Kalman gain, Shape [Batch, Channel, Height, Weight]
        :return: refined z, Shape [Batch, Channel, Height, Weight]
        """
        z_hat = (1 - kalman_gain) * z_code + kalman_gain * z_prime
        return z_hat

    def calc_gain(self, z_codes: torch.Tensor) -> torch.Tensor:
        """
        :param z_codes: Shape [Batch, Sequence, Channel, Height, Weight]
        :return: Shape [Batch, Sequence, Channel, Height, Weight]
        """
        assert z_codes.dim() == 5, f"Expected z_codes has 5 dimension but got {z_codes.shape}"

        image_sequence_length = z_codes.shape[1]
        height, width = z_codes.shape[3:5]

        assert height % self.image_patch == 0 and width % self.image_patch == 0, \
            f"Height ({height}) and width ({width}) must be divisible by {self.image_patch}"

        ################# Uncertainty Estimation #################

        #### reshape
        if self.image_patch > 1:
            z_reshaped = rearrange(
                z_codes,
                "b f c (h ph) (w pw) -> (b f) (h w) (c ph pw)", ph=self.image_patch, pw=self.image_patch)
        else:
            z_reshaped = rearrange(
                z_codes,
                "b f c h w -> (b f) (h w) c"
            )
            pass
        h_codes = z_reshaped

        if self.uncertainty_estimator is not None:
            # uncertainty_estimator takes [n d c] and image_sequence_length as input
            h_codes = self.uncertainty_estimator(h_codes, image_sequence_length=image_sequence_length)

        ### reshape
        if self.image_patch > 1:
            patch_height = height // self.image_patch
            h_codes = rearrange(
                h_codes,
                "n (h w) (c ph pw) -> n c (h ph) (w pw)", ph=self.image_patch, pw=self.image_patch, h=patch_height,
            )
        else:
            h_codes = rearrange(
                h_codes,
                "n (h w) c -> n c h w", h=height
            )

        ################# Kalman Gain Calculation #################

        w_codes = self.kalman_gain_calculator(h_codes)
        w_codes = rearrange(w_codes, "(b f) c h w -> b f c h w", f=image_sequence_length)

        return w_codes


class CNNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(CNNResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(num_groups=in_channels // 4, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=in_channels // 4, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)
