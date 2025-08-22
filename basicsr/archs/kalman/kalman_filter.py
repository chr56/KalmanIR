import torch
import torch.nn as nn
from einops import rearrange


class KalmanFilter(nn.Module):
    """
    Perform a Kalman filter.
    (borrow from KEEP: https://github.com/jnjaby/KEEP)
    """

    def __init__(self,
                 kalman_gain_calculator: nn.Module,
                 predictor: nn.Module,
                 ):
        super().__init__()

        self.kalman_gain_calculator = kalman_gain_calculator

        self.predictor = predictor

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

    def predict(self, z_hat):
        """
        Predict the next state based on the current state
        :param z_hat: Shape [Batch, Channel, Height, Weight]
        :return: Shape [Batch, Channel, Height, Weight]
        """
        z_prime = self.predictor(z_hat)
        return z_prime

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

    def calc_gain(self, uncertainty: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        :param uncertainty: Shape [Batch * Sequence, Channel, Height, Weight]
        :param batch_size: batch size
        :return: Shape [Batch, Sequence, Channel, Height, Weight]
        """
        assert uncertainty.dim() == 4, f"Expected z_codes has 4 dimension but got {uncertainty.shape}"

        w_codes = self.kalman_gain_calculator(uncertainty)
        w_codes = rearrange(w_codes, "(b f) c h w -> b f c h w", b=batch_size)

        return w_codes

    def perform_filtering(self, image_sequence: torch.Tensor, kalman_gain: torch.Tensor):
        """
        :param image_sequence: images in sequence, shape [Batch, Sequence, Channel, Height, Weight]
        :param kalman_gain: pre-calculated kalman gain, shape [Batch, Sequence, Channel, Height, Weight]
        :return: refined result, shape [B, C, H, W]
        """
        z_hat = None
        previous_z = None
        image_sequence_length = image_sequence.shape[1]
        for i in range(image_sequence_length):
            if i == 0:
                z_hat = image_sequence[:, i, ...]  # initialize Z_hat with first z
            else:
                z_prime = self.predict(previous_z.detach())
                z_hat = self.update(
                    image_sequence[:, i, ...],
                    z_prime,
                    kalman_gain[:, i, ...]
                )

            previous_z = z_hat
            pass
        return z_hat
