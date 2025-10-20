from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import AdaLayerNorm, FeedForward, CrossAttention
from einops import rearrange

from basicsr.archs.arch_util import init_weights
from .convolutional_res_block import ConvolutionalResBlock
from .kalman_filter import KalmanFilter


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

        self.apply(init_weights)

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


# borrow from KEEP: https://github.com/jnjaby/KEEP
class SparseCausalTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # # Cross-Attn
        # if cross_attention_dim is not None:
        #     self.attn2 = CrossAttention(
        #         query_dim=dim,
        #         cross_attention_dim=cross_attention_dim,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )
        # else:
        #     self.attn2 = None

        # if cross_attention_dim is not None:
        #     self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        # else:
        #     self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout,
                              activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(
            dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def forward(
            self, hidden_states, image_sequence_length, encoder_hidden_states=None, timestep=None, attention_mask=None
    ):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = self.attn1(
                norm_hidden_states, encoder_hidden_states,
                attention_mask=attention_mask, image_sequence_length=image_sequence_length
            ) + hidden_states

        else:
            hidden_states = self.attn1(
                norm_hidden_states,
                attention_mask=attention_mask, image_sequence_length=image_sequence_length
            ) + hidden_states

        # if self.attn2 is not None:
        #     # Cross-Attention
        #     norm_hidden_states = (
        #         self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        #     )
        #     hidden_states = (
        #         self.attn2(
        #             norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        #         )
        #         + hidden_states
        #     )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=image_sequence_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep)
            if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


# borrow from KEEP: https://github.com/jnjaby/KEEP
class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, image_sequence_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(image_sequence_length) - 1
        former_frame_index[0] = 0

        # d = h*w
        key = rearrange(key, "(b f) d c -> b f d c", f=image_sequence_length)
        key = torch.cat([key[:, [0] * image_sequence_length],
                         key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=image_sequence_length)
        value = torch.cat([value[:, [0] * image_sequence_length],
                           value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(
                    attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(
                    self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(
                query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(
                    query, key, value, attention_mask
                )
            else:
                hidden_states = self._sliced_attention(
                    query, key, value, sequence_length, dim, attention_mask
                )

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
