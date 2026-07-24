import torch
import torch.nn as nn

from .blocks import NAFBlock, TransformerBlock


class DeepUNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=96,
                 num_bottleneck_naf=4, num_bottleneck_transformer=4, transformer_heads=8):
        super().__init__()
        # Encoder
        self.inc = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.enc1 = nn.Sequential(NAFBlock(base_channels), NAFBlock(base_channels))

        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 2, 2)
        self.enc2 = nn.Sequential(NAFBlock(base_channels * 2), NAFBlock(base_channels * 2))

        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 2, 2)
        self.enc3 = nn.Sequential(NAFBlock(base_channels * 4), NAFBlock(base_channels * 4))

        # Bottleneck: deepest / cheapest resolution, so this is where we spend the
        # extra depth -- stacked NAFBlocks for local detail, then transformer blocks
        # for global context (borrowing texture from unoccluded regions).
        bottleneck_layers = [NAFBlock(base_channels * 4) for _ in range(num_bottleneck_naf)]
        bottleneck_layers += [TransformerBlock(base_channels * 4, transformer_heads)
                               for _ in range(num_bottleneck_transformer)]
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.reduce2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.dec2 = nn.Sequential(NAFBlock(base_channels * 2), NAFBlock(base_channels * 2))

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.reduce1 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.dec1 = nn.Sequential(NAFBlock(base_channels), NAFBlock(base_channels))

        self.outc = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(self.inc(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(e3)

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(self.reduce2(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(self.reduce1(d1))

        return self.outc(d1)


class UltimateTwoStageNet(nn.Module):
    """Drop -> (Blur, raindrop-attention mask) -> Clear.

    Stage 1 additionally predicts a soft raindrop-attention mask (supervised against
    |Drop - Blur|), which is fed into Stage 2 so it knows *where* to focus the
    heavier restoration effort -- mirrors the attention-recurrent idea from
    AttentiveGAN-style raindrop removal papers.
    """

    def __init__(self, base_channels=96):
        super().__init__()
        self.stage1 = DeepUNetStage(in_channels=3, out_channels=4, base_channels=base_channels)
        self.stage2 = DeepUNetStage(in_channels=7, out_channels=3, base_channels=base_channels)

    def forward(self, x_drop):
        stage1_out = self.stage1(x_drop)
        residual_rgb = stage1_out[:, :3]
        mask_logit = stage1_out[:, 3:4]

        pred_blur = x_drop + residual_rgb
        pred_mask = torch.sigmoid(mask_logit)

        stage2_input = torch.cat([x_drop, pred_blur, pred_mask], dim=1)
        stage2_residual = self.stage2(stage2_input)
        pred_clear = pred_blur + stage2_residual

        return pred_blur, pred_clear, pred_mask
