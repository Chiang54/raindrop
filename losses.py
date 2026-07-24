import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + (self.eps * self.eps)))


class FFTLoss(nn.Module):
    def forward(self, x, y):
        x_fft = torch.fft.rfft2(x, norm='ortho')
        y_fft = torch.fft.rfft2(y, norm='ortho')
        return F.l1_loss(x_fft.real, y_fft.real) + F.l1_loss(x_fft.imag, y_fft.imag)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = (y - mean) / std
        h_x1, h_y1 = self.slice1(x), self.slice1(y)
        h_x2, h_y2 = self.slice2(h_x1), self.slice2(h_y1)
        h_x3, h_y3 = self.slice3(h_x2), self.slice3(h_y2)
        return F.l1_loss(h_x1, h_y1) + F.l1_loss(h_x2, h_y2) + F.l1_loss(h_x3, h_y3)


class EdgeGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer('kernel_y', kernel_y.repeat(3, 1, 1, 1))

    def forward(self, x, y):
        grad_x_pred = F.conv2d(x, self.kernel_x, padding=1, groups=3)
        grad_y_pred = F.conv2d(x, self.kernel_y, padding=1, groups=3)
        grad_x_gt = F.conv2d(y, self.kernel_x, padding=1, groups=3)
        grad_y_gt = F.conv2d(y, self.kernel_y, padding=1, groups=3)
        return F.l1_loss(grad_x_pred, grad_x_gt) + F.l1_loss(grad_y_pred, grad_y_gt)


def _gaussian_window(window_size, sigma, channels):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer('window', _gaussian_window(window_size, sigma, channels))

    def forward(self, x, y):
        window = self.window.to(device=x.device, dtype=x.dtype)
        pad = self.window_size // 2
        mu_x = F.conv2d(x, window, padding=pad, groups=self.channels)
        mu_y = F.conv2d(y, window, padding=pad, groups=self.channels)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=self.channels) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=self.channels) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=self.channels) - mu_xy

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
        return 1.0 - ssim_map.mean()


def hinge_d_loss(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()
