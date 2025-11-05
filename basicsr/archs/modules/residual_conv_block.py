from torch import nn

from .utils import ActivationFunction, NormLayerType, OneOrMany


class ResidualConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int = -1,
            num_layers: int = 2,
            kernel_size: OneOrMany[int] = 3,
            norm_type: OneOrMany[NormLayerType] = 'layer',
            norm_group: OneOrMany[int] = 3,
            activation_type: OneOrMany[ActivationFunction] = 'relu',
            norm_after_conv: bool = False
    ):
        super().__init__()
        from .utils import get_activation_function, get_norm_layer

        out_channels = out_channels if out_channels > 0 else in_channels

        if isinstance(activation_type, str):
            activation_type = [activation_type] * num_layers
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        if isinstance(norm_type, str):
            norm_type = [norm_type] * num_layers
        if isinstance(norm_group, int):
            norm_group = [norm_group] * num_layers

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.post_norm = norm_after_conv

        self.norm_layers = nn.ModuleList()
        self.activations = nn.ModuleList([get_activation_function(t) for t in activation_type])
        self.conv_layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            out_ch = out_channels
            norm_ch = in_ch if i == 0 and not self.post_norm else out_ch

            conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size[i], padding='same')
            self.conv_layers.append(conv)

            norm = get_norm_layer(norm_type[i], norm_ch, norm_group[i])
            self.norm_layers.append(norm)

        if in_channels != out_channels:
            self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')

    def _forward_layer(self, x, layer):
        if not self.post_norm:
            x = self.norm_layers[layer](x)
        x = self.conv_layers[layer](x)
        x = self.activations[layer](x)
        if self.post_norm:
            x = self.norm_layers[layer](x)
        return x


    def forward(self, x_in):
        x = x_in

        for i in range(self.num_layers):
            x = self._forward_layer(x, layer=i)

        if self.in_channels != self.out_channels:
            x_in = self.conv_residual(x_in)

        return x + x_in
