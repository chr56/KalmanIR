from typing import List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.modules import build_module
from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY


@REFINEMENT_ARCH_REGISTRY.register()
class KFRN(nn.Module):
    def __init__(
            self,
            branch: int,
            channels: int,
            difficult_zone_estimator: Dict[str, Any],
            kalman_gain_calculator: Dict[str, Any],
            kalman_predictor: Dict[str, Any],
            **kwargs,
    ):
        super(KFRN, self).__init__()
        difficult_zone_estimator.update(channels=channels, num_images=branch)
        kalman_gain_calculator.update(channels=channels, num_images=branch)
        kalman_predictor.update(channels=channels, num_images=branch)
        self.channels = channels
        self.branch = branch

        self.difficult_zone_estimator: nn.Module = build_module(difficult_zone_estimator)
        self.kalman_gain_calculator: nn.Module = build_module(kalman_gain_calculator)
        self.kalman_predictor: nn.Module = build_module(kalman_predictor)

    def forward(self, images: List[torch.Tensor]) -> torch.Tensor:
        # Input shape L * [B, C, H, W]
        images = [torch.sin(image) for image in images]
        images = torch.stack(images, dim=1) # [B, L, C, H, W]

        difficult_zone = self.difficult_zone_estimator(images) # [B, C, H, W]

        kalman_gains = self.kalman_gain_calculator(difficult_zone=difficult_zone, images=images) # [B, L, C, H, W]

        refined = perform_kalman_filtering(
            images, kalman_gains, self.kalman_predictor
        )

        # Output shape [B, C, H, W]
        return refined


def perform_kalman_filtering(
        image_sequence: torch.Tensor,
        kalman_gain: torch.Tensor,
        predictor: Callable[[torch.Tensor], torch.Tensor],
):
    """ Performs Kalman filtering on an image sequence.
    :param image_sequence: images in sequence, shape [Batch, Sequence, Channel, Height, Weight]
    :param kalman_gain: pre-calculated kalman gain, shape [Batch, Sequence, Channel, Height, Weight]
    :param predictor: function to predict the next state based on the current state
    :return: refined result, shape [Batch, Channel, Height, Weight]
    """
    current_hat = None
    previous = None
    image_sequence_length = image_sequence.shape[1]
    for i in range(image_sequence_length):
        input_image = image_sequence[:, i, ...]
        input_gain = kalman_gain[:, i, ...]
        current = input_image
        if i == 0:
            current_hat = current
        else:
            current_prime = predictor(previous.detach())  # Predict
            current_hat = (1 - input_gain) * current + input_gain * current_prime  # Update
        previous = current_hat

    return current_hat
