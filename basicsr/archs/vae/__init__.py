from torch import nn


def build_vae(
        variant,
        scaled_size: int,
        channels: int,
        **kwargs
) -> nn.Module:
    vae_latent_dim = kwargs.get('vae_latent_dim', 1024)
    vae_layers = kwargs.get('vae_layers', 4)
    if variant == 'vae_v1':
        from .v1 import VariationalAutoEncoderV1
        return VariationalAutoEncoderV1(
            input_channels=channels,
            input_size=(scaled_size, scaled_size),
            latent_dim=vae_latent_dim,
            num_layers=vae_layers,
        )
    else:
        raise NotImplementedError(f'{variant} is not implemented.')
