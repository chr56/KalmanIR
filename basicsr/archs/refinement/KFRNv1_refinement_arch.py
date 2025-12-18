from typing import List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ImageNormalization
from basicsr.archs.modules import build_module
from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY

SUPPORTED_PREPROCESS = ['none', 'sigmoid', 'tanh', 'sin']


@REFINEMENT_ARCH_REGISTRY.register()
class KFRNv1(nn.Module):
    def __init__(
            self,
            branch: int,
            channels: int,
            difficult_zone_estimator: Dict[str, Any],
            kalman_gain_calculator: Dict[str, Any],
            kalman_predictor: Dict[str, Any],
            input_order: list = None,
            preprocess: str = 'none',
            norm_image: bool = False,
            img_range: float = 1.0,
            **kwargs,
    ):
        super(KFRNv1, self).__init__()
        assert preprocess in SUPPORTED_PREPROCESS, f"unknown pre-process type: {preprocess}"
        difficult_zone_estimator.update(channels=channels, num_images=branch)
        kalman_gain_calculator.update(channels=channels, num_images=branch)
        kalman_predictor.update(channels=channels, num_images=branch)
        self.channels = channels
        self.branch = branch

        self.difficult_zone_estimator: nn.Module = build_module(difficult_zone_estimator)
        self.kalman_gain_calculator: nn.Module = build_module(kalman_gain_calculator)
        self.kalman_predictor: nn.Module = build_module(kalman_predictor)

        self.input_order = input_order

        self.preprocess = preprocess

        self.norm_image = norm_image or kwargs.get('rgb_mean_norm', False)
        if self.norm_image:
            self.image_norm = ImageNormalization(
                channel=channels,
                img_range=img_range,
                mean=kwargs.get('rgb_mean', (0.4488, 0.4371, 0.4040)),
            )

    def preprocess_images(self, images: List[torch.Tensor]):
        if self.preprocess == 'sin':
            return [torch.sin(image) for image in images]
        elif self.preprocess == 'tanh':
            return [torch.tanh(image) for image in images]
        elif self.preprocess == 'sigmoid':
            return [torch.sigmoid(image) for image in images]
        else:
            return images

    def forward(self, images: List[torch.Tensor]) -> dict:
        # Input shape L * [B, C, H, W]

        if self.norm_image:
            images = [self.image_norm(image) for image in images]

        images = torch.stack(self.preprocess_images(images), dim=1)  # [B, L, C, H, W]

        if self.input_order:
            images = images[:, self.input_order, ...]

        difficult_zone = self.difficult_zone_estimator(images)  # [B, C, H, W]

        kalman_gains = self.kalman_gain_calculator(difficult_zone=difficult_zone, images=images)  # [B, L, C, H, W]

        refined = perform_kalman_filtering(
            image_sequence=images,
            kalman_gain=kalman_gains,
            predictor=self.kalman_predictor,
        )

        if self.norm_image:
            refined = self.image_norm.recover(refined)

        return {
            'sr_refined': refined,  # [B, C, H, W]
            'difficult_zone': difficult_zone,  # [B, C, H, W]
        }

    def model_output_format(self):
        return {
            'sr_refined': 'I',
            'difficult_zone': '-',
        }

    def primary_output(self):
        return 'sr_refined'


def perform_kalman_filtering(
        image_sequence: torch.Tensor,
        kalman_gain: torch.Tensor,
        predictor: nn.Module
) -> torch.Tensor:
    """ Performs Kalman filtering on an image sequence.
    :param image_sequence: images in sequence, shape [Batch, Sequence, Channel, Height, Weight]
    :param kalman_gain: pre-calculated kalman gain, shape [Batch, Sequence, Channel, Height, Weight]
    :param predictor: module to predict the next state based on the current state
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
            # Predict Step
            current_prime = predictor(previous.detach())
            # Update Step
            current_hat = (1 - input_gain) * current + input_gain * current_prime
        previous = current_hat

    return current_hat
