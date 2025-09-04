from typing import Callable

import torch


class FlexibleKalmanFilter:
    """
    Perform a Kalman filter.
    (borrow from KEEP: https://github.com/jnjaby/KEEP)
    """

    def __init__(self):
        super().__init__()

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

    def perform_filtering(
            self,
            image_sequence: torch.Tensor,
            kalman_gain: torch.Tensor,
            fn_predict: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        :param fn_predict: function to predict the next state based on the current state
        :param image_sequence: images in sequence, shape [Batch, Sequence, Channel, Height, Weight]
        :param kalman_gain: pre-calculated kalman gain, shape [Batch, Sequence, Channel, Height, Weight]
        :return: refined result, shape [Batch, Channel, Height, Weight]
        """
        z_hat = None
        previous_z = None
        image_sequence_length = image_sequence.shape[1]
        for i in range(image_sequence_length):
            if i == 0:
                z_hat = image_sequence[:, i, ...]  # initialize Z_hat with first z
            else:
                z_prime = fn_predict(previous_z.detach())
                z_hat = self.update(
                    image_sequence[:, i, ...],
                    z_prime,
                    kalman_gain[:, i, ...]
                )

            previous_z = z_hat
            pass
        return z_hat
