import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TripletRaindropDataset, build_scene_split, pad_to_multiple, unpad
from losses import (CharbonnierLoss, EdgeGradientLoss, FFTLoss, SSIMLoss,
                     VGGPerceptualLoss, hinge_d_loss, hinge_g_loss)
from models.discriminator import ConditionalPatchDiscriminator
from models.generator import UltimateTwoStageNet

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# ==============================================================================
# CONFIG -- switch DOMAIN to 'night' to train the night specialist separately.
# Day and night raindrop degradation (glare, low light noise, dynamic range) are
# different enough that one specialist model per domain beats a single unified one.
# ==============================================================================
DOMAIN = 'day'  # 'day' or 'night'
DATA_DIR = f'./dataset/{"Day" if DOMAIN == "day" else "Night"}RainDrop_Train'
SAVE_DIR = f'./checkpoints/{DOMAIN}'

TOTAL_EPOCHS = 300
NUM_EPOCHS_PER_RUN = 9
BASE_LR = 2e-4
BASE_CHANNELS = 96
EMA_DECAY = 0.999
GRAD_CLIP = 1.0
VAL_RATIO = 0.1
SPLIT_SEED = 42

# Progressive resolution schedule. Effective batch size (batch * accum) is kept
# roughly constant (~8) so the optimizer sees a similar gradient-noise level across
# phases even as VRAM per-sample cost grows with crop size.
PHASES = [
    {'end_epoch': 60, 'crop': 256, 'batch': 8, 'accum': 1},
    {'end_epoch': 140, 'crop': 384, 'batch': 4, 'accum': 2},
    {'end_epoch': 220, 'crop': 512, 'batch': 2, 'accum': 4},
    {'end_epoch': 300, 'crop': 640, 'batch': 1, 'accum': 8},
]

# Loss curriculum: perceptual loss ramps in early (avoids it dominating raw pixel
# loss before the network has learned basic structure); adversarial loss only
# turns on once the generator already produces plausible output, otherwise the
# discriminator overwhelms training before there's anything reasonable to refine.
VGG_RAMP_EPOCHS = 15
VGG_MAX_WEIGHT = 0.08
ADV_START_EPOCH = 40
ADV_RAMP_EPOCHS = 30
ADV_MAX_WEIGHT = 0.05

FFT_WEIGHT = 0.10
GRAD_WEIGHT = 0.10
SSIM_WEIGHT = 0.20
MASK_WEIGHT = 0.05
STAGE1_WEIGHT = 0.5
# ==============================================================================


def get_phase(epoch):
    for p in PHASES:
        if epoch <= p['end_epoch']:
            return p
    return PHASES[-1]


def ramp(epoch, start_epoch, ramp_epochs, max_val):
    if epoch < start_epoch:
        return 0.0
    t = min(1.0, (epoch - start_epoch) / max(1, ramp_epochs))
    return max_val * t


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def build_loader(scenes, root_dir, crop_size, batch_size, use_cuda):
    ds = TripletRaindropDataset(root_dir, scenes, crop_size=crop_size, is_train=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                       num_workers=8 if use_cuda else 0, pin_memory=use_cuda,
                       prefetch_factor=4 if use_cuda else None)


def compute_mask_gt(drop, blur, scale=4.0):
    return torch.clamp(torch.mean(torch.abs(drop - blur), dim=1, keepdim=True) * scale, 0.0, 1.0)


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


def train_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = device.type == 'cuda'
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Domain={DOMAIN} Device={device}")

    train_scenes, val_scenes = build_scene_split(DATA_DIR, VAL_RATIO, SPLIT_SEED)
    print(f"Scene split -> train: {len(train_scenes)} scenes | val: {len(val_scenes)} scenes (held out, no leakage)")
    if not train_scenes:
        print(f"錯誤：於路徑 `{DATA_DIR}` 未找到訓練資料集！")
        return

    val_dataset = TripletRaindropDataset(DATA_DIR, val_scenes, crop_size=None, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2 if use_cuda else 0)

    model = UltimateTwoStageNet(base_channels=BASE_CHANNELS).to(device)
    ema_model = copy.deepcopy(model).eval()
    set_requires_grad(ema_model, False)

    discriminator = ConditionalPatchDiscriminator(in_channels=6).to(device)

    criterion_pixel = CharbonnierLoss().to(device)
    criterion_vgg = VGGPerceptualLoss().to(device)
    criterion_fft = FFTLoss().to(device)
    criterion_grad = EdgeGradientLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)

    opt_g = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=BASE_LR, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    latest_ckpt_path = os.path.join(SAVE_DIR, 'latest_model.pth')
    start_epoch = 1
    best_psnr = 0.0

    if os.path.exists(latest_ckpt_path):
        print(f"--> 偵測到既有權重，嘗試讀取: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
            opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
            sched_g.load_state_dict(checkpoint['sched_g_state_dict'])
            sched_d.load_state_dict(checkpoint['sched_d_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"--> 成功恢復進度！從 Epoch [{start_epoch}] 開始執行。")
        except Exception as e:
            print(f"--> 權重不相容 ({e})，從 Epoch [1] 重新開始。")

    end_epoch = min(start_epoch + NUM_EPOCHS_PER_RUN - 1, TOTAL_EPOCHS)
    current_phase_crop = None
    train_loader = None

    for epoch in range(start_epoch, end_epoch + 1):
        phase = get_phase(epoch)
        if phase['crop'] != current_phase_crop:
            current_phase_crop = phase['crop']
            train_loader = build_loader(train_scenes, DATA_DIR, phase['crop'], phase['batch'], use_cuda)
            print(f"--> 進入新解析度階段: crop={phase['crop']} batch={phase['batch']} accum={phase['accum']}")

        model.train()
        discriminator.train()
        epoch_loss = 0.0
        start_time = time.time()
        accum = phase['accum']

        vgg_w = ramp(epoch, 1, VGG_RAMP_EPOCHS, VGG_MAX_WEIGHT)
        adv_w = ramp(epoch, ADV_START_EPOCH, ADV_RAMP_EPOCHS, ADV_MAX_WEIGHT)
        use_adv = adv_w > 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{TOTAL_EPOCHS:03d}] crop={phase['crop']}",
                           leave=False, ncols=110)
        opt_g.zero_grad()
        opt_d.zero_grad()

        for step, (drop, blur, clear) in enumerate(train_pbar):
            drop, blur, clear = drop.to(device), blur.to(device), clear.to(device)

            with torch.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.bfloat16):
                pred_blur, pred_clear, pred_mask = model(drop)

            # ---- Discriminator step (frozen generator output) ----
            if use_adv:
                set_requires_grad(discriminator, True)
                with torch.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.bfloat16):
                    d_real = discriminator(clear, drop)
                    d_fake = discriminator(pred_clear.detach(), drop)
                    d_loss = hinge_d_loss(d_real, d_fake) / accum
                d_loss.backward()
                if (step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
                    opt_d.step()
                    opt_d.zero_grad()
                set_requires_grad(discriminator, False)

            # ---- Generator step ----
            with torch.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.bfloat16):
                mask_gt = compute_mask_gt(drop, blur)
                loss_s1 = criterion_pixel(pred_blur, blur)
                loss_mask = F.l1_loss(pred_mask, mask_gt)

                loss_s2_pixel = criterion_pixel(pred_clear, clear)
                loss_s2_vgg = criterion_vgg(pred_clear, clear)
                loss_s2_fft = criterion_fft(pred_clear, clear)
                loss_s2_grad = criterion_grad(pred_clear, clear)
                loss_s2_ssim = criterion_ssim(pred_clear, clear)

                g_loss = (STAGE1_WEIGHT * loss_s1 + MASK_WEIGHT * loss_mask +
                          loss_s2_pixel + vgg_w * loss_s2_vgg + FFT_WEIGHT * loss_s2_fft +
                          GRAD_WEIGHT * loss_s2_grad + SSIM_WEIGHT * loss_s2_ssim)

                if use_adv:
                    # discriminator.requires_grad_ is False here, so this forward pass
                    # only routes gradient back into the generator, not into D's weights.
                    g_fake = discriminator(pred_clear, drop)
                    g_loss = g_loss + adv_w * hinge_g_loss(g_fake)

                g_loss_scaled = g_loss / accum

            g_loss_scaled.backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt_g.step()
                opt_g.zero_grad()
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.mul_(EMA_DECAY).add_(p.detach(), alpha=1 - EMA_DECAY)
                    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
                        ema_b.copy_(b)

            epoch_loss += g_loss.item()
            train_pbar.set_postfix({"Loss": f"{g_loss.item():.4f}", "vgg_w": f"{vgg_w:.3f}", "adv_w": f"{adv_w:.3f}"})

        sched_g.step()
        sched_d.step()
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        # ---- Validation at native resolution (reflect-padded, no cropping) using EMA weights ----
        ema_model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for drop, _, clear in val_loader:
                drop, clear = drop.to(device), clear.to(device)
                padded, (h, w) = pad_to_multiple(drop, 16)
                with torch.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.bfloat16):
                    _, pred_clear, _ = ema_model(padded)
                pred_clear = unpad(pred_clear, h, w)
                pred_clear = torch.clamp(pred_clear, 0.0, 1.0).float()
                val_psnr += calculate_psnr(pred_clear, clear)
                val_ssim += (1.0 - criterion_ssim(pred_clear, clear).item())

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        print(f"Epoch [{epoch:03d}/{TOTAL_EPOCHS:03d}] | Loss: {avg_loss:.4f} | "
              f"Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f} | Time: {elapsed:.1f}s")

        is_best = avg_psnr > best_psnr
        if is_best:
            best_psnr = avg_psnr

        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'opt_g_state_dict': opt_g.state_dict(),
            'opt_d_state_dict': opt_d.state_dict(),
            'sched_g_state_dict': sched_g.state_dict(),
            'sched_d_state_dict': sched_d.state_dict(),
            'best_psnr': best_psnr,
        }

        torch.save(save_dict, latest_ckpt_path)
        if is_best:
            torch.save(save_dict, os.path.join(SAVE_DIR, 'best_model.pth'))
            print(f"  --> 最佳權重已更新！(Best PSNR: {best_psnr:.2f} dB)")


if __name__ == '__main__':
    train_model()
