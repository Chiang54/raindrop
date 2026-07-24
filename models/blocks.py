import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Local feature extractor (depthwise conv + simplified channel attention)."""

    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel)
        self.sg1 = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0)
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)

        ffn_channel = c * FFN_Expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

    def forward(self, x):
        res = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.conv1(x_norm)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = res + y

        res = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.conv4(x_norm)
        y = self.sg2(y)
        y = self.conv5(y)
        x = res + y

        return x


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed (channel) Attention -- Restormer-style global context.

    Attention is computed over channels (C x C), not spatial (HW x HW), so it stays
    cheap even at large bottleneck resolutions -- needed to let the network borrow
    texture from far-away clean regions to inpaint occluded raindrop areas.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network paired with MDTA."""

    def __init__(self, dim, expand=2.66):
        super().__init__()
        hidden = int(dim * expand)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x
