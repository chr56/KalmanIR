import torch
import torch.nn as nn
from einops import rearrange

from basicsr.archs.arch_util import init_weights
from .difficult_zone_estimators_v4 import DifficultZoneEstimatorV4
from .kalman_filter import KalmanFilter
from .kalman_gain_calulators_v4 import build_gain_calculator_for_v4
from .kalman_predictors import build_predictor
from .uncertainty_estimators_v4 import build_uncertainty_estimator_for_v4


class KalmanRefineNetV4(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(
            self,
            dim: int,
            img_seq: int = 3,
            uncertainty_estimation_mode: str = '',
            gain_calculation_mode: str = '',
            with_difficult_zone_affine: bool = False,
    ):
        super(KalmanRefineNetV4, self).__init__()

        self.difficult_zone_estimator = DifficultZoneEstimatorV4(dim)

        self.uncertainty_estimator = build_uncertainty_estimator_for_v4(
            mode=uncertainty_estimation_mode, seq_length=img_seq, dim=dim
        )

        kalman_gain_calculator = build_gain_calculator_for_v4(mode=gain_calculation_mode, dim=dim)
        predictor = build_predictor('convolutional', dim=dim, seq_length=img_seq)

        self.kalman_filter = KalmanFilter(kalman_gain_calculator=kalman_gain_calculator, predictor=predictor)

        self.with_difficult_zone_affine = with_difficult_zone_affine  #
        if self.with_difficult_zone_affine:
            self.difficult_zone_weight = nn.Parameter(torch.ones(dim))
            self.difficult_zone_bias = nn.Parameter(torch.zeros(dim))

        self.apply(init_weights)

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
        refined_with_kf = self.kalman_filter.perform_filtering(image_sequence, kalman_gain)

        #####################################
        ############### Merge ###############
        #####################################

        if self.with_difficult_zone_affine:
            df_weight = self.difficult_zone_weight.view(-1, 1, 1).expand(-1, H, W)  # [C] -> [C, H, W]
            df_bias = self.difficult_zone_bias.view(-1, 1, 1).expand(-1, H, W)  # [C] -> [C, H, W]
            difficult_zone = df_weight * difficult_zone + df_bias

        difficult_zone = torch.sigmoid(difficult_zone)
        refined = (1 - difficult_zone) * refining + difficult_zone * refined_with_kf

        return refined
