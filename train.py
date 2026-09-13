import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision.models as models
from tqdm import tqdm

# 開啟 TensorCore 最高效能模式 (RTX 4090 必開)
torch.set_float32_matmul_precision('high')

# ==============================================================================
# 1. 核心輕量化特徵提取模組 (NAFBlock)
# ==============================================================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
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

# ==============================================================================
# 2. 獨立深層 U-Net 骨幹網路
# ==============================================================================
class DeepUNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        # Encoder
        self.inc = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.enc1 = nn.Sequential(NAFBlock(base_channels), NAFBlock(base_channels))

        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 2, 2)
        self.enc2 = nn.Sequential(NAFBlock(base_channels * 2), NAFBlock(base_channels * 2))

        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 2, 2)
        self.enc3 = nn.Sequential(NAFBlock(base_channels * 4), NAFBlock(base_channels * 4))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            NAFBlock(base_channels * 4), NAFBlock(base_channels * 4)
        )

        # Decoder
        # 修正 checkerboard artifact：原本的 nn.ConvTranspose2d(k=2, s=2) 改為
        # 「先最近鄰上採樣、再用一般卷積平滑」(resize-convolution)，
        # 這是 Odena et al., 2016《Deconvolution and Checkerboard Artifacts》
        # (https://distill.pub/2016/deconv-checkerboard/) 提出的標準解法。
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1)
        )
        self.reduce2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.dec2 = nn.Sequential(NAFBlock(base_channels * 2), NAFBlock(base_channels * 2))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1)
        )
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

        # 這裡不加原圖 x，因為可能是從 6 channel 壓回 3 channel
        output = self.outc(d1)
        return output

# ==============================================================================
# 3. 雙階段究極網路 (Drop -> Blur -> Clear)
# ==============================================================================
class UltimateTwoStageNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        # Stage 1: 輸入 Drop (3 ch)，輸出預測的 Blur (3 ch)
        self.stage1 = DeepUNetStage(in_channels=3, out_channels=3, base_channels=base_channels)

        # Stage 2: 輸入 Drop + 預測的 Blur (6 ch)，輸出最終清晰圖 Clear (3 ch)
        self.stage2 = DeepUNetStage(in_channels=6, out_channels=3, base_channels=base_channels)

    def forward(self, x_drop):
        # 階段一：去除雨滴，生成模糊背景
        stage1_residual = self.stage1(x_drop)
        pred_blur = x_drop + stage1_residual  # 殘差連線

        # 階段二：將原始雨滴圖與去雨滴後的圖融合，進行去模糊與幾何銳化
        stage2_input = torch.cat([x_drop, pred_blur], dim=1)
        stage2_residual = self.stage2(stage2_input)
        pred_clear = pred_blur + stage2_residual # 基於 Blur 進行清晰化還原

        return pred_blur, pred_clear

# ==============================================================================
# 4. 複合損失函數群組
# ==============================================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + (self.eps * self.eps)))

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
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

class SSIMMetric(nn.Module):
    """
    標準單尺度 SSIM（11x11 高斯窗），純 torch 實作、不需額外套件。
    用於訓練時的驗證指標（PSNR 容易偏好「平滑模糊但像素誤差小」的結果，
    SSIM 對結構/銳利度更敏感，兩者一起看才不會選到偏模糊的權重）。
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('_window_cache', self._create_window(window_size, sigma, 3), persistent=False)
        self._cached_channels = 3

    def _gaussian(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, sigma, channel):
        g_1d = self._gaussian(window_size, sigma).unsqueeze(1)
        g_2d = g_1d.mm(g_1d.t()).unsqueeze(0).unsqueeze(0)
        return g_2d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        channel = img1.size(1)
        if channel != self._cached_channels or self._window_cache.dtype != img1.dtype:
            self._window_cache = self._create_window(self.window_size, self.sigma, channel).to(img1.device, img1.dtype)
            self._cached_channels = channel
        window = self._window_cache
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

# ==============================================================================
# 5. 三檔案聯合載入 Dataset (Drop, Blur, Clear 同步增強)
# ==============================================================================
class TripletRaindropDataset(Dataset):
    def __init__(self, root_dir, crop_size=384, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train

        self.drop_dir = os.path.join(root_dir, 'Drop')
        self.blur_dir = os.path.join(root_dir, 'Blur')
        self.clear_dir = os.path.join(root_dir, 'Clear')

        self.image_triplets = []
        if os.path.exists(self.drop_dir):
            scene_folders = sorted(os.listdir(self.drop_dir))
            for scene in scene_folders:
                scene_drop_path = os.path.join(self.drop_dir, scene)
                if os.path.isdir(scene_drop_path):
                    for img_name in sorted(os.listdir(scene_drop_path)):
                        drop_path = os.path.join(self.drop_dir, scene, img_name)
                        blur_path = os.path.join(self.blur_dir, scene, img_name)
                        clear_path = os.path.join(self.clear_dir, scene, img_name)
                        if os.path.exists(clear_path) and os.path.exists(blur_path):
                            self.image_triplets.append((drop_path, blur_path, clear_path))

    def __len__(self):
        return len(self.image_triplets)

    def __getitem__(self, idx):
        drop_path, blur_path, clear_path = self.image_triplets[idx]

        img_drop = Image.open(drop_path).convert('RGB')
        img_blur = Image.open(blur_path).convert('RGB')
        img_clear = Image.open(clear_path).convert('RGB')

        tensor_drop = TF.to_tensor(img_drop)
        tensor_blur = TF.to_tensor(img_blur)
        tensor_clear = TF.to_tensor(img_clear)

        if self.is_train:
            # 1. 智能防護：確保原圖的長寬至少大於等於 crop_size (若太小則等比例放大)
            _, h, w = tensor_drop.shape
            if h < self.crop_size or w < self.crop_size:
                new_h = max(h, self.crop_size)
                new_w = max(w, self.crop_size)
                tensor_drop = TF.resize(tensor_drop, [new_h, new_w], antialias=True)
                tensor_blur = TF.resize(tensor_blur, [new_h, new_w], antialias=True)
                tensor_clear = TF.resize(tensor_clear, [new_h, new_w], antialias=True)

            # 2. 隨機多尺度縮放 (確保縮放後的尺寸依然大於等於 crop_size)
            if torch.rand(1) > 0.5:
                scale_factor = torch.empty(1).uniform_(0.75, 1.0).item()
                _, h, w = tensor_drop.shape
                new_h = max(int(h * scale_factor), self.crop_size)
                new_w = max(int(w * scale_factor), self.crop_size)
                tensor_drop = TF.resize(tensor_drop, [new_h, new_w], antialias=True)
                tensor_blur = TF.resize(tensor_blur, [new_h, new_w], antialias=True)
                tensor_clear = TF.resize(tensor_clear, [new_h, new_w], antialias=True)

            # 3. 強制執行 RandomCrop (此時保證長寬絕對足夠，不會再有跳過裁切的問題)
            i, j, th, tw = T.RandomCrop.get_params(
                tensor_drop, output_size=(self.crop_size, self.crop_size)
            )
            tensor_drop = TF.crop(tensor_drop, i, j, th, tw)
            tensor_blur = TF.crop(tensor_blur, i, j, th, tw)
            tensor_clear = TF.crop(tensor_clear, i, j, th, tw)

            # 4. 隨機水平翻轉
            if torch.rand(1) > 0.5:
                tensor_drop = TF.hflip(tensor_drop)
                tensor_blur = TF.hflip(tensor_blur)
                tensor_clear = TF.hflip(tensor_clear)
        else:
            # 驗證階段：確保長寬是 16 的倍數，以免 U-Net 多層下採樣時維度無法整除而報錯
            _, h, w = tensor_drop.shape
            new_h = (h // 16) * 16
            new_w = (w // 16) * 16
            tensor_drop = TF.crop(tensor_drop, 0, 0, new_h, new_w)
            tensor_blur = TF.crop(tensor_blur, 0, 0, new_h, new_w)
            tensor_clear = TF.crop(tensor_clear, 0, 0, new_h, new_w)

        return tensor_drop, tensor_blur, tensor_clear

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# ==============================================================================
# 6. 主訓練流程 (RTX 4090 優化版)
# ==============================================================================
def train_model():
    # ----------------------------------------------------------------------
    # 資料集模式切換：'day' | 'night' | 'both'
    # 用來跑論文需要的消融實驗（單日間 vs 單夜間 vs 日夜合併），
    # 可用環境變數覆寫，例如：DATASET_MODE=night python3 train.py
    # ----------------------------------------------------------------------
    DATASET_MODE = os.environ.get('DATASET_MODE', 'both').strip().lower()
    assert DATASET_MODE in ('day', 'night', 'both'), f"DATASET_MODE 必須是 day/night/both，收到: {DATASET_MODE}"

    day_dir = './dataset/DayRainDrop_Train'
    night_dir = './dataset/NightRainDrop_Train'

    # day-only 沿用原本的 ./checkpoints（既有的 Epoch150 訓練成果，等同這次消融實驗的
    # 「day-only」那組數據，不用重跑），night / both 各自存到獨立資料夾，避免互相覆蓋。
    save_dir = './checkpoints' if DATASET_MODE == 'day' else f'./checkpoints_{DATASET_MODE}'

    # RTX 4090 設定：兩套 Deep U-Net 很吃顯存，4 Batch 搭配 384 視野為最佳平衡
    batch_size = 4
    crop_size = 384
    total_target_epochs = 150
    num_epochs_per_run = 9
    lr = 2e-4

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = (device.type == 'cuda')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 啟動『究極雙階段』訓練，裝置: {device}，資料集模式: {DATASET_MODE}，權重存放: {save_dir}")

    # ----------------------------------------------------------------------
    # 依 DATASET_MODE 組出訓練資料集（day / night / 兩者合併 ConcatDataset）
    # ----------------------------------------------------------------------
    sub_datasets = []
    if DATASET_MODE in ('day', 'both'):
        d = TripletRaindropDataset(day_dir, crop_size=crop_size, is_train=True)
        print(f"  --> Day 資料集：{len(d)} 筆（{day_dir}）")
        if len(d) > 0:
            sub_datasets.append(d)
    if DATASET_MODE in ('night', 'both'):
        d = TripletRaindropDataset(night_dir, crop_size=crop_size, is_train=True)
        print(f"  --> Night 資料集：{len(d)} 筆（{night_dir}）")
        if len(d) > 0:
            sub_datasets.append(d)

    if len(sub_datasets) == 0:
        print(f"錯誤：找不到訓練資料集！請確認 DATASET_MODE={DATASET_MODE} 對應的資料夾存在。")
        return

    dataset = sub_datasets[0] if len(sub_datasets) == 1 else ConcatDataset(sub_datasets)
    print(f"  --> 合併後總計：{len(dataset)} 筆訓練樣本")

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8 if use_cuda else 0, pin_memory=use_cuda, prefetch_factor=2 if use_cuda else None
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2 if use_cuda else 0)

    # 實例化究極雙階段網路
    model = UltimateTwoStageNet(base_channels=64).to(device)

    criterion_pixel = CharbonnierLoss().to(device)
    criterion_vgg = VGGPerceptualLoss().to(device)
    criterion_fft = FFTLoss().to(device)
    criterion_grad = EdgeGradientLoss().to(device)
    metric_ssim = SSIMMetric().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_target_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    latest_ckpt_path = os.path.join(save_dir, 'latest_model.pth')
    start_epoch = 1
    best_psnr = 0.0
    best_ssim = 0.0
    best_combined = 0.0

    if os.path.exists(latest_ckpt_path):
        print(f"--> 偵測到既有權重，嘗試讀取: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and use_cuda:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            best_ssim = checkpoint.get('best_ssim', 0.0)
            best_combined = checkpoint.get('best_combined', 0.0)
            print(f"--> 成功恢復進度！從 Epoch [{start_epoch}] 開始執行。")
        except Exception:
            print("因模型架構或輸入資料改變（例如新加入 Night 資料集），將清空歷史記錄並從 Epoch [1] 重新開始。")

    end_epoch = min(start_epoch + num_epochs_per_run - 1, total_target_epochs)

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch:03d}/{total_target_epochs:03d}]", leave=False, ncols=100)

        for drop, blur, clear in train_pbar:
            drop, blur, clear = drop.to(device), blur.to(device), clear.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.float16):
                # 同時取得兩階段輸出
                pred_blur, pred_clear = model(drop)

                # Stage 1: 原本只用 pixel loss，容易讓小雨滴/細紋殘留（只要像素數值大致
                # 對上就過關，對局部小面積的雨滴痕跡懲罰不夠）。加入輕量的感知/梯度損失，
                # 逼 Stage 1 把雨滴的「結構」也清乾淨，而不只是顏色數值對。
                loss_s1_pixel = criterion_pixel(pred_blur, blur)
                loss_s1_vgg = criterion_vgg(pred_blur, blur)
                loss_s1_grad = criterion_grad(pred_blur, blur)
                loss_s1 = loss_s1_pixel + 0.05 * loss_s1_vgg + 0.05 * loss_s1_grad

                # Stage 2: 拉高 VGG 感知損失與梯度損失的權重（原本 0.08 / 0.10 太小，
                # 幾乎被 pixel loss 主導，這是輸出整體偏模糊的主因——純像素回歸損失
                # 對不確定的高頻細節會傾向「取平均」而不是「猜一個銳利的答案」）。
                loss_s2_pixel = criterion_pixel(pred_clear, clear)
                loss_s2_vgg = criterion_vgg(pred_clear, clear)
                loss_s2_fft = criterion_fft(pred_clear, clear)
                loss_s2_grad = criterion_grad(pred_clear, clear)

                # 聯合優化權重設定（v2：拉高感知/梯度權重，降低對純像素損失的依賴）
                total_loss = (0.5 * loss_s1) + (
                    1.0 * loss_s2_pixel + 0.25 * loss_s2_vgg + 0.10 * loss_s2_fft + 0.25 * loss_s2_grad
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            train_pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        # 驗證階段 (以第二階段輸出 Pred_Clear 與 Clear_GT 為評估基準，同時算 PSNR + SSIM)
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for drop, _, clear in val_loader:
                drop, clear = drop.to(device), clear.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.float16):
                    _, pred_clear = model(drop)
                pred_clear = torch.clamp(pred_clear, 0.0, 1.0).float()
                clear_f = clear.float()
                val_psnr += calculate_psnr(pred_clear, clear_f)
                val_ssim += metric_ssim(pred_clear, clear_f).item()

        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        # PSNR 常見上限抓 40dB 做正規化，跟 0~1 的 SSIM 各佔一半權重，
        # 避免單獨用 PSNR 選模型時，選到「模糊但像素誤差小」的權重（這正是這次遇到的問題）。
        combined_score = 0.5 * (avg_psnr / 40.0) + 0.5 * avg_ssim
        print(f"Epoch [{epoch:03d}/{total_target_epochs:03d}] | Loss: {avg_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB | Val SSIM: {avg_ssim:.4f} | Time: {elapsed:.1f}s")

        is_best = combined_score > best_combined
        if is_best:
            best_combined = combined_score
            best_psnr = avg_psnr
            best_ssim = avg_ssim

        save_dict = {
            'epoch': epoch,
            'dataset_mode': DATASET_MODE,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_cuda else None,
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_combined': best_combined,
            'last_psnr': avg_psnr,
            'last_ssim': avg_ssim,
        }

        torch.save(save_dict, latest_ckpt_path)
        if is_best:
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
            print(f"  --> 最佳權重已更新！(Best PSNR: {best_psnr:.2f} dB, Best SSIM: {best_ssim:.4f})")

if __name__ == '__main__':
    train_model()
