import torch
import torch.nn as nn
from einops import rearrange

from .convolutional_res_block import ConvolutionalResBlock
from .kalman_filter import KalmanFilter


class KalmanRefineNetV2(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(self, dim: int):
        super(KalmanRefineNetV2, self).__init__()

        self.difficult_zone_estimator = DifficultZoneEstimator(dim)

        self.uncertainty_estimator = UncertaintyEstimator(dim)

        kalman_gain_calculator = nn.Sequential(
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

    # noinspection PyPep8Naming
    def forward(
            self,
            refining: torch.Tensor,
            raw_a: torch.Tensor,
            raw_b: torch.Tensor,
            sigma: torch.Tensor,
    ) -> torch.Tensor:

        #####################################
        ##### Difficult Zone Estimation #####
        #####################################

        difference = torch.div(torch.abs(refining - raw_a) + torch.abs(refining - raw_a), 2)

        difficult_zone_mask = self.difficult_zone_estimator(difference, refining, raw_a, raw_b)

        #####################################
        #### Uncertainty & KG Estimation ####
        #####################################

        sequence = torch.stack((raw_b, raw_a, refining), dim=1).contiguous()
        B, L, C, H, W = sequence.shape

        uncertainty = self.uncertainty_estimator(sequence, difficult_zone_mask, sigma)
        kalman_gain = self.kalman_filter.calc_gain(uncertainty, B)

        #####################################
        ########### Kalman Filter ###########
        #####################################

        z_hat = None
        previous_z = None
        for i in range(L):
            if i == 0:
                z_hat = sequence[:, i, ...]  # initialize Z_hat with first z
            else:
                z_prime = self.kalman_filter.predict(previous_z.detach())
                z_hat = self.kalman_filter.update(
                    sequence[:, i, ...],
                    z_prime,
                    kalman_gain[:, i, ...]
                )

            previous_z = z_hat
            pass

        #####################################
        ############### Merge ###############
        #####################################

        difference = torch.tanh(difference)
        refined = (1 - difference) * refining + difference * z_hat

        return refined


class DifficultZoneEstimator(nn.Module):
    def __init__(self, channel: int):
        super(DifficultZoneEstimator, self).__init__()

        self.channel = channel

        self.compressed_channel = 6
        self.channel_compressor = ChannelCompressor(in_channel=channel, out_channel=self.compressed_channel)

        self.channel_stacked = 3 * self.compressed_channel + channel
        self.hidden_estimator_channel = self.channel_stacked // 2
        self.estimator_main = nn.Sequential(
            nn.GroupNorm(1, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, self.hidden_estimator_channel, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, self.hidden_estimator_channel, eps=1e-6),
            nn.Conv2d(self.hidden_estimator_channel, 1, kernel_size=3, padding='same'),
        )
        self.estimator_residual = nn.Sequential(
            nn.GroupNorm(1, channel, eps=1e-6),
            nn.Conv2d(channel, channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, channel, eps=1e-6),
            nn.Conv2d(channel, 1, kernel_size=3, padding='same'),
        )

    def forward(self, difference: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param difference: Mean difference, shape [B, C, H, W]
        :param a: shape [B, C, H, W]
        :param b: shape [B, C, H, W]
        :param c: shape [B, C, H, W]
        :return: Difficult Zone Mask, shape [B, 1, H, W]
        """

        ##################################
        ###### Compress Channels #########
        ##################################

        a_compressed = self.channel_compressor(a)
        b_compressed = self.channel_compressor(b)
        c_compressed = self.channel_compressor(c)

        stacked = torch.cat((difference, a_compressed, b_compressed, c_compressed), dim=1)  # [B, 3C' + C, H, W]

        del a_compressed, b_compressed, c_compressed

        ##################################
        #### Difficult Zone Estimation ###
        ##################################

        # [B, 3C' + C, H, W] -> [B, 1, H, W]
        # [B, C, H, W] -> [B, 1, H, W]
        difficult_zone_mask = self.estimator_main(stacked) + self.estimator_residual(difference)

        return difficult_zone_mask

class UncertaintyEstimator(nn.Module):
    def __init__(
            self, channel: int,
    ):
        super(UncertaintyEstimator, self).__init__()

        self.compressed_sigma_channel = 2
        self.sigma_channel_compressor = ChannelCompressor(in_channel=channel, out_channel=self.compressed_sigma_channel)

        self.block = ConvolutionalResBlock(
            channel + self.compressed_sigma_channel + 1, channel,
            norm_num_groups_1=3, norm_num_groups_2=channel // 4,
        )

    def forward(
            self,
            sequence: torch.Tensor,
            difficult_zone_mask: torch.Tensor,
            sigma: torch.Tensor,
    ):
        """
        :param sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone_mask: Difficult Zone mask, shape [B, 1, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B*L, C, H, W]
        """
        _, sequence_length, _, _, _ = sequence.shape

        sigma = self.sigma_channel_compressor(sigma)  # [B, C, H, W] -> [B, C_compressed, H, W]

        merged = []
        for l in range(sequence_length):
            stacked = torch.cat((sequence[:, l, ...], difficult_zone_mask, sigma), dim=1)  # -> [B, C', H, W]
            merged.append(
                self.block(stacked)  # [B, C', H, W] -> [B, C, H, W]
            )
        uncertainty = torch.cat(merged, dim=1)  # L * [B, C, H, W] -> [B, L * C, H, W]

        uncertainty = rearrange(uncertainty, "b (l c) h w -> (b l) c h w", l=sequence_length)
        return uncertainty


class ChannelCompressor(nn.Module):
    def __init__(self, in_channel: int, out_channel: int = 6, norm_num_groups=None):
        super(ChannelCompressor, self).__init__()
        num_groups = 1 if norm_num_groups is None else norm_num_groups
        self.channel_compressor = nn.Sequential(
            nn.GroupNorm(num_groups, in_channel, eps=1e-6),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(num_groups, in_channel, eps=1e-6),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding='same'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_compressor(x)
