from typing import List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.modules import build_module
from basicsr.archs.modules.layer_norm import LayerNorm2d
from basicsr.archs.refinement import REFINEMENT_ARCH_REGISTRY

SUPPORTED_PREPROCESS = ['none', 'sigmoid', 'tanh', 'sin', 'layer_norm']


@REFINEMENT_ARCH_REGISTRY.register()
class KFRNv1(nn.Module):
    def __init__(
            self,
            branch: int,
            channels: int,
            difficult_zone_estimator: Dict[str, Any],
            kalman_gain_calculator: Dict[str, Any],
            kalman_predictor: Dict[str, Any],
            preprocess: str = 'none',
            rgb_mean_norm: bool = False,
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

        self.preprocess = preprocess
        if self.preprocess == 'layer_norm':
            affine = kwargs.get('preprocess_layer_norm_affine', True)
            self.layer_norm = LayerNorm2d(channels, eps=1e-6, elementwise_affine=affine)

        self.rgb_mean_norm = rgb_mean_norm
        if self.rgb_mean_norm:
            rgb_mean = kwargs.get('rgb_mean', (0.4488, 0.4371, 0.4040))
            self.mean = torch.Tensor(rgb_mean).view(1, len(rgb_mean), 1, 1)

        self.kalman_predictor_argument_size = len(self.kalman_predictor.model_input_format())

    def preprocess_images(self, images: List[torch.Tensor]):
        if self.rgb_mean_norm:
            self.mean = self.mean.type_as(images[0])
            images = [(image - self.mean) for image in images]

        if self.preprocess == 'sin':
            return [torch.sin(image) for image in images]
        elif self.preprocess == 'tanh':
            return [torch.tanh(image) for image in images]
        elif self.preprocess == 'sigmoid':
            return [torch.sigmoid(image) for image in images]
        elif self.preprocess == 'layer_norm':
            return [self.layer_norm(image) for image in images]
        else:
            return images

    def forward(self, images: List[torch.Tensor]) -> dict:
        # Input shape L * [B, C, H, W]
        images = torch.stack(self.preprocess_images(images), dim=1)  # [B, L, C, H, W]

        difficult_zone = self.difficult_zone_estimator(images)  # [B, C, H, W]

        kalman_gains = self.kalman_gain_calculator(difficult_zone=difficult_zone, images=images)  # [B, L, C, H, W]

        refined = perform_kalman_filtering(
            image_sequence=images,
            kalman_gain=kalman_gains,
            predictor=self.kalman_predictor,
            predictor_input_count=self.kalman_predictor_argument_size,
        )

        if self.rgb_mean_norm:
            refined = refined + self.mean

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
        predictor: nn.Module,
        predictor_input_count: int
) -> torch.Tensor:
    """ Performs Kalman filtering on an image sequence.
    :param image_sequence: images in sequence, shape [Batch, Sequence, Channel, Height, Weight]
    :param kalman_gain: pre-calculated kalman gain, shape [Batch, Sequence, Channel, Height, Weight]
    :param predictor: module to predict the next state based on the current state
    :param predictor_input_count: predict input argument size
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
            if predictor_input_count == 2:
                current_prime = predictor(previous.detach(), current)
            else:
                current_prime = predictor(previous.detach())
            # Update Step
            current_hat = (1 - input_gain) * current + input_gain * current_prime
        previous = current_hat

    return current_hat
