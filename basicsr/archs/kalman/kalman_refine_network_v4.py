import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from .convolutional_res_block import ConvolutionalResBlock
from .kalman_filter import KalmanFilter
from .kalman_gain_calulators import KalmanGainCalculatorV0
from .predictors import KalmanPredictorV0
from .utils import calculate_difference, cal_ae, cal_bce, cal_cs


class KalmanRefineNetV4(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(self, dim: int):
        super(KalmanRefineNetV4, self).__init__()

        self.difficult_zone_estimator = DifficultZoneEstimator(dim)

        self.uncertainty_estimator = UncertaintyEstimator(dim)

        self.kalman_filter = KalmanFilter(
            kalman_gain_calculator=KalmanGainCalculatorV0(dim),
            predictor=KalmanPredictorV0(dim),
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
        image_sequence = torch.stack([raw_b, raw_a, refining], dim=1)
        B, L, C, H, W = image_sequence.shape

        #####################################
        ##### Difficult Zone Estimation #####
        #####################################

        difficult_zone = self.difficult_zone_estimator(refining, raw_a, raw_b)

        #####################################
        #### Uncertainty & KG Estimation ####
        #####################################

        uncertainty = self.uncertainty_estimator(
            image_sequence,
            difficult_zone,
            sigma,
        )  # [B, L, C, H, W]

        uncertainty = rearrange(uncertainty, "b l c h w -> (b l) c h w")
        kalman_gain = self.kalman_filter.calc_gain(uncertainty, B)

        #####################################
        ########### Kalman Filter ###########
        #####################################

        del uncertainty
        refined_with_kf = self._perform_kalman_filter(kalman_gain, image_sequence)

        #####################################
        ############### Merge ###############
        #####################################

        difficult_zone = torch.sigmoid(difficult_zone)
        refined = (1 - difficult_zone) * refining + difficult_zone * refined_with_kf

        return refined

    def _perform_kalman_filter(self, kalman_gain: torch.Tensor, image_sequence: torch.Tensor):
        """
        :param kalman_gain: pre-calculated kalman gain, shape [B, L, C, H, W]
        :param image_sequence: images in sequence, shape [B, L, C, H, W]
        :return: refined result, shape [B, C, H, W]
        """
        z_hat = None
        previous_z = None
        image_sequence_length = image_sequence.shape[1]
        for i in range(image_sequence_length):
            if i == 0:
                z_hat = image_sequence[:, i, ...]  # initialize Z_hat with first z
            else:
                z_prime = self.kalman_filter.predict(previous_z.detach())
                z_hat = self.kalman_filter.update(
                    image_sequence[:, i, ...],
                    z_prime,
                    kalman_gain[:, i, ...]
                )

            previous_z = z_hat
            pass
        return z_hat


class DifficultZoneEstimator(nn.Module):
    def __init__(self, channel: int):
        super(DifficultZoneEstimator, self).__init__()

        self.channel = channel

        self.channel_stacked = 3 * channel + 1
        self.estimator_main = nn.Sequential(
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, self.channel_stacked, kernel_size=3, padding='same'),
            nn.SiLU(inplace=True),
            nn.GroupNorm(1, self.channel_stacked, eps=1e-6),
            nn.Conv2d(self.channel_stacked, channel, kernel_size=3, padding='same'),
        )
        self.residual_ratio = 0.4

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

        ae_difference = calculate_difference(cal_ae, b, a, c)  # [B, C, H, W]
        # kl_difference = calculate_deference(cal_kl, b, a, c)  # [B, C, H, W]
        bce_difference = calculate_difference(cal_bce, b, a, c)  # [B, C, H, W]
        cs_difference = calculate_difference(cal_cs, b, a, c)  # [B, 1, H, W]

        all_difference = torch.cat((b, cs_difference, ae_difference, bce_difference), dim=1)  # [B, C', H, W]

        ##################################
        #### Difficult Zone Estimation ###
        ##################################

        # ([B, C', H, W] -> [B, C, H, W]) * (1-r) + [B, C, H, W] * r
        difficult_zone = (self.estimator_main(all_difference) * (1 - self.residual_ratio) +
                          ae_difference * self.residual_ratio)

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


class UncertaintyEstimator(nn.Module):
    def __init__(self, channel: int, ):
        super(UncertaintyEstimator, self).__init__()
        self.block = ConvolutionalResBlock(
            3 * channel, channel,
            norm_num_groups_1=1, norm_num_groups_2=1,
        )

    def forward(self, image_sequence: torch.Tensor, difficult_zone: torch.Tensor, sigma: torch.Tensor):
        """
        :param image_sequence: Image sequence, shape [B, L, C, H, W]
        :param difficult_zone: Difficult Zone, shape [B, C, H, W]
        :param sigma: variance, shape [B, C, H, W]
        :return: estimated uncertainty, shape [B, L, C, H, W]
        """

        length = image_sequence.shape[1]

        uncertainty = []
        for i in range(length):
            x = torch.cat((difficult_zone, sigma, image_sequence[:, i, ...]), dim=1)
            x = self.block(x)
            uncertainty.append(x)

        uncertainty = torch.stack(uncertainty, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]

        return uncertainty
