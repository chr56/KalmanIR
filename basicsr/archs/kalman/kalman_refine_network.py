from typing import Tuple

import torch
import torch.nn as nn

from .kalman_filter import KalmanFilter
from .uncertainty_estimator import UncertaintyEstimator


class KalmanRefineNetV0(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(self, dim: int):
        super().__init__()

        image_patch = 8
        dim_stacked = dim * image_patch * image_patch

        uncertainty_estimator = UncertaintyEstimator(
            dim=dim_stacked,
            num_attention_heads=2,
            attention_head_dim=12,
            num_uncertainty_layers=8
        )

        self.kalman_filter = KalmanFilter(
            emb_dim=dim,
            image_patch=image_patch,
            uncertainty_estimator=uncertainty_estimator
        )

    # noinspection PyPep8Naming
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: Shape [Batch, Sequence, Channel, Height, Weight]
        """
        assert z.dim() == 5, f"Expected 5 dimension but got {z.shape}"
        B, L, C, H, W = z.shape

        kalman_gain = self.kalman_filter.calc_gain(z)

        z_hat = None
        previous_z = None
        for i in range(L):
            if i == 0:
                z_hat = z[:, i, ...]  # initialize Z_hat with first z
            else:
                z_prime = self.kalman_filter.predict(previous_z.detach())
                z_hat = self.kalman_filter.update(z[:, i, ...], z_prime, kalman_gain[:, i, ...])

            previous_z = z_hat
            pass

        return z_hat
