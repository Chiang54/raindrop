import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ==============================================================================
# 1. 核心輕量化模組 (SimpleGate & NAFBlock)
# ==============================================================================
class SimpleGate(nn.Module):
    """無激活函數閘控，顯著加快推論速度"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """輕量化高效特徵提取模組"""
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
# 2. 兩階段雨滴去除與去模糊模型 (Fast-TwoStageRaindropNet)
# ==============================================================================
class FastTwoStageRaindropNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # --- Stage 1: 去除雨滴 (Drop -> Blur) ---
        self.s1_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.s1_enc = NAFBlock(base_channels)
        self.s1_bottleneck = NAFBlock(base_channels)
        self.s1_dec = NAFBlock(base_channels)
        self.s1_out = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

        # --- Stage 2: 去模糊與背景還原 (Blur_feat + Stage1_out -> Clear) ---
        self.s2_in = nn.Conv2d(base_channels + in_channels, base_channels, 3, 1, 1)
        self.s2_enc = NAFBlock(base_channels)
        self.s2_bottleneck = NAFBlock(base_channels)
        self.s2_dec = NAFBlock(base_channels)
        self.s2_out = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        # Stage 1 運算
        f1 = self.s1_in(x)
        f1_enc = self.s1_enc(f1)
        f1_b = self.s1_bottleneck(f1_enc)
        f1_dec = self.s1_dec(f1_b)
        stage1_blur = x + self.s1_out(f1_dec)  # 預測 Blur 影像

        # Stage 2 運算 (融合 Stage 1 特徵)
        f2_in = torch.cat([f1_dec, stage1_blur], dim=1)
        f2 = self.s2_in(f2_in)
        f2_enc = self.s2_enc(f2)
        f2_b = self.s2_bottleneck(f2_enc)
        f2_dec = self.s2_dec(f2_b)
        stage2_clear = stage1_blur + self.s2_out(f2_dec)  # 預測 Clear 影像

        return stage1_blur, stage2_clear

# ==============================================================================
# 3. 損失函數 (Charbonnier Loss)
# ==============================================================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + (self.eps * self.eps))
        return torch.mean(loss)

# ==============================================================================
# 4. 三層資料夾 Dataset 加載器
# ==============================================================================
class TripletRaindropDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, is_train=True):
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
                    img_names = sorted(os.listdir(scene_drop_path))
                    for img_name in img_names:
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
            _, h, w = tensor_drop.shape
            if h >= self.crop_size and w >= self.crop_size:
                i, j, th, tw = T.RandomCrop.get_params(
                    tensor_drop, output_size=(self.crop_size, self.crop_size)
                )
                tensor_drop = TF.crop(tensor_drop, i, j, th, tw)
                tensor_blur = TF.crop(tensor_blur, i, j, th, tw)
                tensor_clear = TF.crop(tensor_clear, i, j, th, tw)

            if torch.rand(1) > 0.5:
                tensor_drop = TF.hflip(tensor_drop)
                tensor_blur = TF.hflip(tensor_blur)
                tensor_clear = TF.hflip(tensor_clear)

        return tensor_drop, tensor_blur, tensor_clear

# ==============================================================================
# 5. 評估指標 (PSNR)
# ==============================================================================
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# ==============================================================================
# 6. 主訓練腳本 (支援分段訓練/自動接續)
# ==============================================================================
def train_model():
    data_dir = 'D:/gitserver/python/raindrop/dataset/DayRainDrop_Train' 
    save_dir = './checkpoints'
    
    batch_size = 8
    total_target_epochs = 120    # 預計訓練總輪數
    num_epochs_per_run = 20     # 每次執行要跑幾輪
    lr = 2e-4
    crop_size = 256

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = (device.type == 'cuda')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 啟動訓練，使用硬體裝置: {device}")

    dataset = TripletRaindropDataset(data_dir, crop_size=crop_size, is_train=True)
    if len(dataset) == 0:
        print(f"錯誤：於路徑 {data_dir} 未找到訓練圖片！請檢查資料夾路徑。")
        return

    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    num_workers = 4 if use_cuda else 0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0
    )

    model = FastTwoStageRaindropNet(in_channels=3, base_channels=32).to(device)
    criterion = CharbonnierLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_target_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    # ------------------ 檢查並載入歷史訓練狀態 (Resume) ------------------
    latest_ckpt_path = os.path.join(save_dir, 'latest_model.pth')
    start_epoch = 1
    best_psnr = 0.0

    if os.path.exists(latest_ckpt_path):
        print(f"--> 偵測到歷史模型權重，正在讀取: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and use_cuda:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0.0)
        print(f"--> 成功恢復進度！將從 Epoch [{start_epoch}] 開始繼續訓練。")

    end_epoch = min(start_epoch + num_epochs_per_run - 1, total_target_epochs)

    if start_epoch > total_target_epochs:
        print(f"模型已完全訓練達到目標總輪數 ({total_target_epochs} Epochs)！無需繼續訓練。")
        return

    print(f"本次執行範圍: Epoch [{start_epoch:03d}] -> Epoch [{end_epoch:03d}] (共 {end_epoch - start_epoch + 1} 輪)")

    # ------------------ 訓練迴圈 ------------------
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch [{epoch:03d}/{total_target_epochs:03d}]", 
            leave=False,
            ncols=100
        )

        for drop, blur, clear in train_pbar:
            drop, blur, clear = drop.to(device), blur.to(device), clear.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                pred_blur, pred_clear = model(drop)
                
                loss_s1 = criterion(pred_blur, blur)
                loss_s2 = criterion(pred_clear, clear)
                total_loss = 0.5 * loss_s1 + 1.0 * loss_s2

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            train_pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        scheduler.step()
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)

        # ---------------- 驗證階段 ----------------
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for drop, _, clear in val_loader:
                drop, clear = drop.to(device), clear.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                    _, pred_clear = model(drop)
                pred_clear = torch.clamp(pred_clear, 0.0, 1.0)
                val_psnr += calculate_psnr(pred_clear, clear)

        avg_psnr = val_psnr / len(val_loader)

        print(f"Epoch [{epoch:03d}/{total_target_epochs:03d}] | Train Loss: {avg_loss:.4f} | Val PSNR: {avg_psnr:.2f} dB | Time: {elapsed:.1f}s")

        # 更新最佳 PSNR
        is_best = avg_psnr > best_psnr
        if is_best:
            best_psnr = avg_psnr

        # 儲存最新的模型進度 (給 Resume 使用)
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_cuda else None,
            'best_psnr': best_psnr,
        }
        
        # 覆蓋最新權重檔
        torch.save(save_dict, latest_ckpt_path)

        # 若創下最高分數，額外存一份 best_model.pth
        if is_best:
            torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
            print(f"  --> 創下新紀錄！最佳權重已更新 (Best PSNR: {best_psnr:.2f} dB)")

    print(f"\n[完成] 本次 20 輪訓練結束！進度已儲存至 {latest_ckpt_path}")
    print(f"下次再次執行 `python train.py` 將自動從 Epoch [{end_epoch + 1}] 繼續訓練。")

if __name__ == '__main__':
    train_model()