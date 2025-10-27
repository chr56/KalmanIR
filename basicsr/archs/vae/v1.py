import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layer_norm: bool = True,
                 ):
        super().__init__()

        if layer_norm:
            from basicsr.archs.kalman.utils import LayerNorm2d
            norm_layer = LayerNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = norm_layer(out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = norm_layer(out_channels, eps=1e-6)
        self.act = nn.LeakyReLU(inplace=True)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        y = self.conv1(x_in)
        y = self.norm1(y)
        y = self.act(y)

        y = self.conv2(y)
        y = self.norm2(y)

        if self.residual_conv is not None:
            x_in = self.residual_conv(x_in)
        y += x_in

        return self.act(y)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x)
        return self.pool(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.res_block = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.res_block(x)


class Encoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            num_layers: int,
            latent_dim: int,
            input_image_size: Tuple[int, int]
    ):
        super().__init__()

        channels = [input_channels]
        for i in range(num_layers):
            channels.append(min(64 * (2 ** i), 512))

        self.initial_conv = nn.Conv2d(input_channels, channels[1], 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(channels[1])
        self.initial_relu = nn.ReLU(inplace=True)

        self.down_blocks = nn.ModuleList()
        for i in range(1, num_layers):
            self.down_blocks.append(DownSampleBlock(channels[i], channels[i + 1]))

        # Calculate feature map size dynamically
        with torch.no_grad():
            r = torch.zeros(1, input_channels, *input_image_size)
            r = self.initial_conv(r)
            for block in self.down_blocks:
                r = block(r)
            self.final_size = r.size()[2:]
        final_channels = channels[-1]
        final_features = final_channels * self.final_size[0] * self.final_size[1]

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(final_features, latent_dim)
        self.fc_log_var = nn.Linear(final_features, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))

        for block in self.down_blocks:
            x = block(x)

        x_flat = self.flatten(x)
        mu = self.fc_mu(x_flat)
        log_var = self.fc_log_var(x_flat)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            num_layers: int,
            output_channels: int,
            input_image_size: Tuple[int, int]
    ):
        super().__init__()

        channels = [min(64 * (2 ** i), 512) for i in range(num_layers + 1)]
        channels = channels[::-1]

        # Calculate feature map size
        with torch.no_grad():
            encoder = Encoder(output_channels, num_layers, latent_dim, input_image_size)
            self.initial_size = encoder.final_size
            final_channels = channels[0]
        initial_features = final_channels * self.initial_size[0] * self.initial_size[1]

        self.fc = nn.Linear(latent_dim, initial_features)

        self.up_blocks = nn.ModuleList()
        for i in range(num_layers - 1):
            self.up_blocks.append(UpSampleBlock(channels[i], channels[i + 1]))

        self.final_conv = nn.Conv2d(channels[-2], output_channels, kernel_size=3, padding=1)
        self.final_activation = nn.Sigmoid()

        self.final_size = self.initial_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, x.size(1) // (self.final_size[0] * self.final_size[1]), self.final_size[0], self.final_size[1])

        for block in self.up_blocks:
            x = block(x)

        x = self.final_conv(x)
        return self.final_activation(x)


class VariationalAutoEncoderV1(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int, int],
            input_channels: int,
            latent_dim: int,
            num_layers: int,
            **kwargs,
    ):
        super().__init__()

        self.encoder = Encoder(input_channels, num_layers, latent_dim, input_size)
        self.decoder = Decoder(latent_dim, num_layers, input_channels, input_size)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var
