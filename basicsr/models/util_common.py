from typing import Dict

import torch
from torch import nn

from basicsr.models.util_config import convert_format
from basicsr.utils.img_util import (
    calculate_and_padding_image,
    calculate_borders_for_chopping,
    recover_from_patches_and_remove_paddings
)


def patch_forward(
        network: nn.Module,
        lq: torch.Tensor,
        scale: int,
        output_formats: list,
        output_indexes_enabled: list,
) -> Dict[int, torch.Tensor]:
    """
    Forward pass through a network using partitioned LQ data.

    This function divides the input image into smaller patches, processes each
    patch through the network, and reconstructs the output image.

    :param network: The target model.
    :param lq: The target LQ image, with shape (B, C, H, W).
    :param scale: The SR scaling factor of model.
    :param output_formats: A list of output formats for each output index.
    :param output_indexes_enabled: A list of indices indicating which outputs to process.

    :return: A dictionary containing the reconstructed images for each output index enabled.
              The keys of the dictionary correspond to the indices in `output_indexes_enabled`,
              and the values are the processed images after removing paddings and scaling.

    """

    # padding image and calculate partition parameters
    img, col, row, mod_pad_h, mod_pad_w, split_h, split_w, shave_h, shave_w = calculate_and_padding_image(lq)

    # noinspection PyPep8Naming
    B, C, H, W = img.shape

    # list of partition borders
    chopping_boxes = calculate_borders_for_chopping(
        col, row, split_h, split_w, shave_h, shave_w
    )

    # list of patches / partitions
    partitioned_img = []
    for box in chopping_boxes:
        h_range, w_range = box
        partitioned_img.append(img[..., h_range, w_range])

    del chopping_boxes
    output_size = len(output_formats)
    prediction_patches = {k: [] for k in range(output_size)}

    # image processing of each partition
    for patch in partitioned_img:
        raw_outputs = network(patch)
        assert hasattr(raw_outputs, '__len__'), "model output must be iterable"
        assert len(raw_outputs) == output_size, "model image output size mismatched with settings"
        for k in output_indexes_enabled:
            patch = convert_format(
                raw_outputs[k], from_format=output_formats[k], to_format='D'
            )
            prediction_patches[k].append(patch)
            pass

    predictions = dict()
    for k in output_indexes_enabled:
        predictions[k] = recover_from_patches_and_remove_paddings(
            prediction_patches[k], col, row,
            B, C, W, H, scale,
            split_h, split_w, shave_h, shave_w,
            mod_pad_h, mod_pad_w, scale
        )

    return predictions
