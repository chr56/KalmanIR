import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from .convolutional_res_block import ConvolutionalResBlock
from .kalman_filter import KalmanFilter


class KalmanRefineNetV3(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(self, dim: int):
        super(KalmanRefineNetV3, self).__init__()

        self.difficult_zone_estimator = DifficultZoneEstimator(dim)

        self.uncertainty_estimator = PerImageUncertaintyEstimator(dim)

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

        difficult_zone = self.difficult_zone_estimator(refining, raw_a, raw_b)

        #####################################
        #### Uncertainty & KG Estimation ####
        #####################################

        sequence = torch.stack((raw_b, raw_a, refining), dim=1).contiguous()
        B, L, C, H, W = sequence.shape

        uncertainty = []
        for i in range(L):
            img_uncertainty = self.uncertainty_estimator(sequence[:, i, ...], difficult_zone, sigma)
            uncertainty.append(img_uncertainty)
        uncertainty = torch.cat(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L * C, H, W]
        uncertainty = rearrange(uncertainty, "b (l c) h w -> (b l) c h w", l=L)

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

        difficult_zone = torch.tanh(difficult_zone)
        refined = (1 - difficult_zone) * refining + difficult_zone * z_hat

        return refined


class DifficultZoneEstimator(nn.Module):
    def __init__(self, channel: int):
        super(DifficultZoneEstimator, self).__init__()

        self.channel = channel

        self.channel_stacked = 2 * channel
        self.estimator_main = nn.Sequential(
            nn.GroupNorm(2, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(2, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, channel, kernel_size=3, padding='same'),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        :param a: shape [B, C, H, W]
        :param b: shape [B, C, H, W]
        :param c: shape [B, C, H, W]
        :return: Difficult Zone, shape [B, C, H, W]
        """

        ##################################
        ###### Compress Channels #########
        ##################################

        def mean_reduce(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
            return torch.div(d1 + d2, 2)

        abs_difference = mean_reduce(torch.abs(a - b), torch.abs(b - c))
        kl_difference = mean_reduce(cal_kl(a, b), cal_kl(b, c))
        # ce_difference = mean_reduce(cal_bce(a, b), cal_bce(b, c))

        all_difference = torch.cat((abs_difference, kl_difference), dim=1)  # [B, 2C, H, W]

        ##################################
        #### Difficult Zone Estimation ###
        ##################################

        # [B, 2C, H, W] -> [B, C, H, W]
        difficult_zone = self.estimator_main(all_difference)

        return difficult_zone


class PerImageUncertaintyEstimator(nn.Module):
    def __init__(self, channel: int, ):
        super(PerImageUncertaintyEstimator, self).__init__()
        self.block = ConvolutionalResBlock(
            3 * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )

    def forward(self, img: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param img: Image, shape [B, C, H, W]
        :param difficult_zone: Difficult Zone mask, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, C, H, W]
        """
        uncertainty = self.block(torch.cat((difficult_zone, sigma, img), dim=1))
        return uncertainty


class ChannelCompressor(nn.Module):
    def __init__(self, in_channel: int, out_channel: int = 6, norm_num_groups=None):
        super(ChannelCompressor, self).__init__()
        num_groups = 1 if norm_num_groups is None else norm_num_groups
        self.norm = nn.GroupNorm(num_groups, in_channel, eps=1e-6)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(num_groups, in_channel, eps=1e-6),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(num_groups, in_channel, eps=1e-6),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding='same'),
        )
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding='same'),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.main_branch(x) + self.residual_branch(x)
        return x


def cal_kl(value1, value2) -> torch.Tensor:
    """Calculate Kullback-Leibler divergence; result's shape remains same."""
    p = torch.log_softmax(value1, dim=1)
    q = torch.softmax(value2, dim=1)
    return F.kl_div(p, q, reduction='none')


def cal_bce(value1, value2) -> torch.Tensor:
    """Calculate Binary Cross Entropy; result's shape remains same."""
    p = torch.softmax(value1, dim=1)
    q = torch.softmax(value2, dim=1)
    return F.binary_cross_entropy(p, q, reduction='none')


def cal_cs(value1, value2, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Calculate Cosine Similarity, taking channel as dimension; result's channel reduces to 1."""
    return F.cosine_similarity(value1, value2, dim, eps).unsqueeze(dim)
