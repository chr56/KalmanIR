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
                 num_attention_heads: int = 2,
                 attention_head_dim: int = 12,
                 num_uncertainty_layers: int = 8,
                 ):
        super().__init__()

        assert image_patch >= 1, "Image patch must be at least 1"
        self.image_patch = image_patch

        transformer_dim = emb_dim * image_patch * image_patch
        self.uncertainty_estimator = nn.ModuleList(
            [
                BasicTransformerBlock(
                    transformer_dim,
                    num_attention_heads,
                    attention_head_dim,
                )
                for d in range(num_uncertainty_layers)
            ]
        )
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
        h_codes = z_reshaped  # uncertainty_estimator takes [n d c] as input

        #### Pass Uncertainty Estimator
        for block in self.uncertainty_estimator:
            h_codes = block(h_codes, sequence_length=image_sequence_length)

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


class BasicTransformerBlock(nn.Module):
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
            self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, sequence_length=None
    ):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = self.attn1(
                norm_hidden_states, encoder_hidden_states,
                attention_mask=attention_mask, image_sequence_length=sequence_length
            ) + hidden_states

        else:
            hidden_states = self.attn1(
                norm_hidden_states,
                attention_mask=attention_mask, image_sequence_length=sequence_length
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
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=sequence_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep)
            if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


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
