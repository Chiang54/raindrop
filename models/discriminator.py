import torch
import torch.nn as nn


class ConditionalPatchDiscriminator(nn.Module):
    """PatchGAN discriminator conditioned on the raindrop input (pix2pix-style).

    Pushes the generator toward realistic high-frequency texture that pure
    pixel/perceptual losses tend to over-smooth.
    """

    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()

        def sn_conv(i, o, k, s, p):
            return nn.utils.spectral_norm(nn.Conv2d(i, o, k, s, p))

        self.net = nn.Sequential(
            sn_conv(in_channels, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels, base_channels * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels * 2, base_channels * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels * 4, base_channels * 8, 4, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels * 8, 1, 4, 1, 1),
        )

    def forward(self, img, condition):
        x = torch.cat([img, condition], dim=1)
        return self.net(x)
