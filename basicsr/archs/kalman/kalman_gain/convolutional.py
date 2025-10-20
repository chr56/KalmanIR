import torch
from torch import nn as nn


class DeepConvolutionalMultipleChannelsV1(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV1, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm

        self.block1 = ConvolutionalResBlockGroupNorm(
            channels=channel * 2, out_channels=channel, norm_group=6, activation_type='leaky_relu', kernel_size=3,
        )
        self.block2 = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='sigmoid', kernel_size=5,
        )
        self.block3 = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=1, activation_type='sigmoid', kernel_size=3,
        )

        self.out_linear = nn.Conv2d(channel, channel, kernel_size=1, padding='same')

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_image(self, uncertainty: torch.Tensor, image: torch.Tensor):
        x = torch.cat((uncertainty, image), dim=1)  # [B, 2C, H, W]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        gain = self.out_linear(gain)
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image = image_sequence[:, i, ...]
            gain = self._forward_one_image(uncertainty, image)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class DeepConvolutionalMultipleChannelsV2(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV2, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm
        self.block1 = ConvolutionalResBlockGroupNorm(
            channels=channel * 2, out_channels=channel, norm_group=6, activation_type='leaky_relu', kernel_size=3,
        )
        self.block2 = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='leaky_relu', kernel_size=3,
        )
        self.block3 = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=1, activation_type='sigmoid', kernel_size=3,
        )

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_image(self, uncertainty: torch.Tensor, image: torch.Tensor):
        x = torch.cat((uncertainty, image), dim=1)  # [B, 2C, H, W]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image = image_sequence[:, i, ...]
            gain = self._forward_one_image(uncertainty, image)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class DeepConvolutionalMultipleChannelsV3a(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV3a, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm

        self.image_block = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='sigmoid'
        )

        self.merge_block1 = ConvolutionalResBlockGroupNorm(
            channels=channel * 2, out_channels=channel, norm_group=6, activation_type='leaky_relu'
        )
        self.merge_block2 = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='sigmoid'
        )

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_image(self, uncertainty: torch.Tensor, image: torch.Tensor):
        image_feature = self.image_block(image)
        x = torch.cat((uncertainty, image_feature), dim=1)  # [B, 2C, H, W]
        x = self.merge_block1(x)
        x = self.merge_block2(x)
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image = image_sequence[:, i, ...]
            gain = self._forward_one_image(uncertainty, image)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class DeepConvolutionalMultipleChannelsV3b(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV3b, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm

        self.image_block1 = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='leaky_relu'
        )
        self.image_block2 = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='sigmoid'
        )
        self.merge_block = ConvolutionalResBlockGroupNorm(
            channels=channel * 2, out_channels=channel, norm_group=6, activation_type='sigmoid'
        )

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_image(self, uncertainty: torch.Tensor, image: torch.Tensor):
        y = image
        y = self.image_block1(y)
        y = self.image_block2(y)
        x = torch.cat((uncertainty, y), dim=1)  # [B, 2C, H, W]
        x = self.merge_block(x)  # [B, 2C, H, W] -> [B, C, H, W]
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image = image_sequence[:, i, ...]
            gain = self._forward_one_image(uncertainty, image)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class DeepConvolutionalMultipleChannelsV3c(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV3c, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm

        self.image_block = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='leaky_relu'
        )
        self.uncertainty_block = ConvolutionalResBlockGroupNorm(
            channels=channel, out_channels=channel, norm_group=3, activation_type='sigmoid'
        )
        self.merge_block = ConvolutionalResBlockGroupNorm(
            channels=channel * 2, out_channels=channel, norm_group=6, activation_type='sigmoid'
        )

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_image(self, uncertainty: torch.Tensor, image: torch.Tensor):
        y = self.image_block(image)
        u = self.uncertainty_block(uncertainty)
        x = torch.cat((u, y), dim=1)  # [B, 2C, H, W]
        x = self.merge_block(x)  # [B, 2C, H, W] -> [B, C, H, W]
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image = image_sequence[:, i, ...]
            gain = self._forward_one_image(uncertainty, image)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class DeepConvolutionalMultipleChannelsV4(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV4, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm
        self.image_blocks = nn.ModuleList(
            [
                ConvolutionalResBlockGroupNorm(
                    channels=channel, out_channels=channel, norm_group=3, activation_type='leaky_relu'
                )
                for _ in range(seq_length)
            ]
        )
        self.merge_block = ConvolutionalResBlockGroupNorm(
            channels=channel * 2, out_channels=channel, norm_group=6, activation_type='sigmoid'
        )

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_step(self, uncertainty: torch.Tensor, image_feature: torch.Tensor):
        x = torch.cat((uncertainty, image_feature), dim=1)  # [B, 2C, H, W]
        x = self.merge_block(x)  # [B, 2C, H, W] -> [B, C, H, W]
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image_feature = self.image_blocks[i](image_sequence[:, i, ...])
            gain = self._forward_one_step(uncertainty, image_feature)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class DeepConvolutionalMultipleChannelsV5(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV5, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ConvolutionalResBlockGroupNorm
        self.image_input_block = ConvolutionalResBlockGroupNorm(
            channels=channel, norm_group=3, activation_type='leaky_relu'
        )
        self.image_features_blocks = nn.ModuleList(
            [
                ConvolutionalResBlockGroupNorm(
                    channels=channel, out_channels=channel, norm_group=3, activation_type='leaky_relu'
                )
                for _ in range(seq_length)
            ]
        )
        self.merge_block = ConvolutionalResBlockGroupNorm(
            channels=channel * 2, out_channels=channel, norm_group=6, activation_type='sigmoid'
        )

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_step(self, uncertainty: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((uncertainty, image_features), dim=1)  # [B, 2C, H, W]
        x = self.merge_block(x)  # [B, 2C, H, W] -> [B, C, H, W]
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image_features = self.image_input_block(image_sequence[:, i, ...])
            image_features = self.image_features_blocks[i](image_features)
            gain = self._forward_one_step(uncertainty, image_features)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains


class DeepConvolutionalMultipleChannelsV5e(nn.Module):
    def __init__(self, channel: int, seq_length: int, uncertainty_update_ratio: float):
        super(DeepConvolutionalMultipleChannelsV5e, self).__init__()
        self.channel = channel
        self.seq_length = seq_length

        from ..convolutional_res_block import ResidualConvBlock
        self.image_input_block = ResidualConvBlock(
            channel, norm_type='group', norm_group=3, activation_type='leaky_relu'
        )
        self.image_features_blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    channel,
                    norm_type='group', norm_group=3,
                    num_layers=3, activation_type=['leaky_relu', 'leaky_relu', 'silu']
                )
                for _ in range(seq_length)
            ]
        )
        self.merge_block = ResidualConvBlock(
            in_channels=channel * 2, out_channels=channel,
            norm_type='group', norm_group=6, activation_type='silu'
        )

        self.update_ratio = uncertainty_update_ratio
        self.remain_ratio = 1 - self.update_ratio

    def _forward_one_step(self, uncertainty: torch.Tensor, image_features: torch.Tensor):
        x = torch.cat((uncertainty, image_features), dim=1)  # [B, 2C, H, W]
        x = self.merge_block(x)  # [B, 2C, H, W] -> [B, C, H, W]
        gain = self.remain_ratio * uncertainty + self.update_ratio * x
        return gain

    def forward(self, uncertainty: torch.Tensor, image_sequence: torch.Tensor) -> torch.Tensor:
        kalman_gains = []
        for i in range(self.seq_length):
            image_features = self.image_input_block(image_sequence[:, i, ...])
            image_features = self.image_features_blocks[i](image_features)
            gain = self._forward_one_step(uncertainty, image_features)
            kalman_gains.append(gain)
        kalman_gains = torch.stack(kalman_gains, dim=1)  # L * [B, C, H, W] -> [B, L, C, H, W]
        return kalman_gains