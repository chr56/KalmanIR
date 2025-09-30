import torch
import torch.nn as nn

from basicsr.archs.arch_util import init_weights
from .difficult_zone_estimators_v6 import build_difficult_zone_estimator_for_v6
from .kalman_filter_flexible import FlexibleKalmanFilter
from .kalman_gain_calulators_v6 import build_gain_calculator_for_v6
from .kalman_predictors import build_predictor
from .uncertainty_estimators_v6 import build_uncertainty_estimator_for_v6


class KalmanRefineNetV6(nn.Module):
    """
    Refine results with Kalman filter
    - No latent space
    - No post-processing after updating
    """

    def __init__(
            self,
            dim: int,
            img_seq: int = 3,
            variant_difficult_zone_estimator: str = '',
            variant_uncertainty_estimation: str = '',
            variant_gain_calculation: str = '',
            variant_preditor: str = '',
            with_difficult_zone_affine: bool = False,
    ):
        super(KalmanRefineNetV6, self).__init__()

        self.difficult_zone_estimator = build_difficult_zone_estimator_for_v6(
            variant=variant_difficult_zone_estimator, dim=dim, seq_length=img_seq
        )

        self.uncertainty_estimator = build_uncertainty_estimator_for_v6(
            variant=variant_uncertainty_estimation, dim=dim, seq_length=img_seq
        )

        self.kalman_gain_calculator = build_gain_calculator_for_v6(
            variant=variant_gain_calculation, dim=dim, seq_length=img_seq
        )

        self.kalman_preditor = build_predictor(variant_preditor, dim=dim, seq_length=img_seq)

        self.kalman_filter = FlexibleKalmanFilter()

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

        uncertainty = self.uncertainty_estimator(image_sequence, difficult_zone, sigma)  # [B, C, H, W]

        kalman_gain = self.kalman_gain_calculator(uncertainty, image_sequence)  # [B, L, C, H, W]

        #####################################
        ########### Kalman Filter ###########
        #####################################

        del uncertainty
        refined_with_kf = self.kalman_filter.perform_filtering(
            image_sequence,
            kalman_gain,
            self.kalman_preditor.__call__,
        )

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
