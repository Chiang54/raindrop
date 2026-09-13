import os
import glob
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

# ==============================================================================
# 1. 核心輕量化特徵提取模組
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
# 2. 雙階段究極網路架構 (與訓練腳本完全一致)
# ==============================================================================
class DeepUNetStage(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        self.inc = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.enc1 = nn.Sequential(NAFBlock(base_channels), NAFBlock(base_channels))
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 2, 2)
        self.enc2 = nn.Sequential(NAFBlock(base_channels * 2), NAFBlock(base_channels * 2))
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 2, 2)
        self.enc3 = nn.Sequential(NAFBlock(base_channels * 4), NAFBlock(base_channels * 4))

        self.bottleneck = nn.Sequential(NAFBlock(base_channels * 4), NAFBlock(base_channels * 4))

        # 修正 checkerboard artifact：與 train.py 同步，改為 resize-convolution
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
        return self.outc(d1)

class UltimateTwoStageNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.stage1 = DeepUNetStage(in_channels=3, out_channels=3, base_channels=base_channels)
        self.stage2 = DeepUNetStage(in_channels=6, out_channels=3, base_channels=base_channels)

    def forward(self, x_drop):
        stage1_residual = self.stage1(x_drop)
        pred_blur = x_drop + stage1_residual

        stage2_input = torch.cat([x_drop, pred_blur], dim=1)
        stage2_residual = self.stage2(stage2_input)
        pred_clear = pred_blur + stage2_residual

        return pred_blur, pred_clear

# ==============================================================================
# 3. 測試推論主程式
# ==============================================================================
def run_test():
    # 與 train.py 的 DATASET_MODE 對應：day 用原本的 ./checkpoints，
    # night / both 各自讀自己的 ./checkpoints_night、./checkpoints_both，
    # 方便對三組消融實驗的模型分別產生比對圖與量測 PSNR/SSIM。
    dataset_mode = os.environ.get('DATASET_MODE', 'day').strip().lower()
    assert dataset_mode in ('day', 'night', 'both'), f"DATASET_MODE 必須是 day/night/both，收到: {dataset_mode}"
    ckpt_dir = './checkpoints' if dataset_mode == 'day' else f'./checkpoints_{dataset_mode}'

    checkpoint_path = os.path.join(ckpt_dir, 'best_model.pth')  # 一律用 best_model.pth（依 PSNR+SSIM 綜合分數選出的最佳權重）
    test_input_dir = './testdata/DayRainDrop_Train/Drop'
    output_dir = f'./results_test_{dataset_mode}'

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的推論硬體: {device}，資料集模式: {dataset_mode}，權重: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"錯誤: 找不到模型權重檔 `{checkpoint_path}`。")
        return

    # 實例化【究極雙階段網路】
    model = UltimateTwoStageNet(base_channels=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ckpt_epoch = checkpoint.get('epoch', '?')
    ckpt_psnr = checkpoint.get('best_psnr', checkpoint.get('last_psnr', 0.0))
    ckpt_ssim = checkpoint.get('best_ssim', checkpoint.get('last_ssim', None))
    if ckpt_ssim is not None:
        print(f"成功載入究極雙階段權重！(Epoch {ckpt_epoch}) Val PSNR: {ckpt_psnr:.2f} dB, Val SSIM: {ckpt_ssim:.4f}")
    else:
        print(f"成功載入究極雙階段權重！(Epoch {ckpt_epoch}) Val PSNR: {ckpt_psnr:.2f} dB（此權重無 SSIM 紀錄，屬於舊版 checkpoint）")

    # 搜尋測試圖片
    scene_folders = sorted(os.listdir(test_input_dir))
    test_images = []

    for scene in scene_folders[:9]:
        scene_path = os.path.join(test_input_dir, scene)
        if os.path.isdir(scene_path):
            imgs = glob.glob(os.path.join(scene_path, '*.png')) + glob.glob(os.path.join(scene_path, '*.jpg'))
            test_images.extend(imgs[:5])

    if not test_images:
        print("未找到任何測試圖片！")
        return

    print(f"找到 {len(test_images)} 張測試圖片，開始進行極致去雨滴推論...")

    with torch.no_grad():
        for idx, img_path in enumerate(test_images):
            raw_img = Image.open(img_path).convert('RGB')

            # 確保圖片長寬能被 16 整除 (避免 U-Net 降採樣出錯)
            w, h = raw_img.size
            new_w = (w // 16) * 16
            new_h = (h // 16) * 16
            raw_img = raw_img.crop((0, 0, new_w, new_h))

            input_tensor = TF.to_tensor(raw_img).unsqueeze(0).to(device)

            # 模型推論：直接獲取第二階段的高清輸出
            _, pred_clear = model(input_tensor)
            pred_clear = torch.clamp(pred_clear, 0.0, 1.0)

            # 將原始雨滴圖 (Input) 與模型還原圖 (Pred Clear) 左右拼接
            compared_result = torch.cat([input_tensor, pred_clear], dim=3)

            filename = f"test_ultimate_result_{idx+1:03d}.png"
            save_path = os.path.join(output_dir, filename)
            save_image(compared_result, save_path)

            print(f"[{idx+1}/{len(test_images)}] 已生成對比圖: {save_path}")

    print(f"\n測試完成！所有對比圖已儲存至 `{output_dir}` 資料夾。")

if __name__ == '__main__':
    run_test()
