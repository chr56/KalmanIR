import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .convolutional_res_block import ConvolutionalResBlock
from .kalman_filter import KalmanFilter

from .sparse_causal_transformer import SparseCausalTransformerBlock


class KalmanRefineNetV0(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(
            self,
            dim: int,
    ):
        super(KalmanRefineNetV0, self).__init__()
        self.image_patch = 8
        self.dim_stacked = dim * self.image_patch * self.image_patch
        # self.uncertainty_estimator = UncertaintyEstimator(
        #     dim=self.dim_stacked,
        #     num_attention_heads=2,
        #     attention_head_dim=12,
        #     num_uncertainty_layers=8
        # )
        self.uncertainty_estimator = None

        kalman_gain_calculator = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            ConvolutionalResBlock(dim, dim),
            ConvolutionalResBlock(dim, dim),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        predictor = nn.Sequential(
            ConvolutionalResBlock(dim, dim),
            ConvolutionalResBlock(dim, dim),
            nn.Sigmoid(),
        )

        self.kalman_filter = KalmanFilter(
            kalman_gain_calculator=kalman_gain_calculator,
            predictor=predictor,
        )

    # noinspection PyPep8Naming
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: Shape [Batch, Sequence, Channel, Height, Weight]
        """
        assert z.dim() == 5, f"Expected 5 dimension but got {z.shape}"
        B, L, C, H, W = z.shape

        uncertainty = self.calculate_uncertainty(z)
        kalman_gain = self.kalman_filter.calc_gain(uncertainty, B)

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

    def calculate_uncertainty(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: input, shape [B, L, C, H, W]
        :return: Stacked uncertainty, shape [B*L, C, H, W]
        """

        _, image_sequence_length, _, height, width = z.shape

        if self.uncertainty_estimator is None:
            return rearrange(z, "b f c h w -> (b f) c h w")

        ### reshape
        if self.image_patch > 1:
            assert height % self.image_patch == 0 and width % self.image_patch == 0, \
                f"Height ({height}) and width ({width}) must be divisible by {self.image_patch}"
            z = rearrange(
                z,
                "b f c (h ph) (w pw) -> (b f) (h w) (c ph pw)", ph=self.image_patch, pw=self.image_patch)
        else:
            z = rearrange(
                z,
                "b f c h w -> (b f) (h w) c"
            )
            pass

        # uncertainty_estimator takes [n d c] and image_sequence_length as input
        uncertainty: torch.Tensor = self.uncertainty_estimator(z, image_sequence_length=image_sequence_length)

        ### reshape
        if self.image_patch > 1:
            patch_height = height // self.image_patch
            uncertainty = rearrange(
                uncertainty,
                "n (h w) (c ph pw) -> n c (h ph) (w pw)", ph=self.image_patch, pw=self.image_patch, h=patch_height,
            )
        else:
            uncertainty = rearrange(
                uncertainty,
                "n (h w) c -> n c h w", h=height
            )

        return uncertainty


class UncertaintyEstimator(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int = 2,
            attention_head_dim: int = 12,
            num_uncertainty_layers: int = 8,
    ):
        super(UncertaintyEstimator, self).__init__()
        self.layers = nn.ModuleList(
            [
                SparseCausalTransformerBlock(
                    dim,
                    num_attention_heads,
                    attention_head_dim,
                )
                for d in range(num_uncertainty_layers)
            ]
        )

    def forward(self, x: torch.Tensor, image_sequence_length: int) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, image_sequence_length=image_sequence_length)
        return x

