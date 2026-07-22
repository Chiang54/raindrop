import os
import glob
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

# ==============================================================================
# 1. 模型架構定義 (必須與訓練時完全一致)
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

class FastTwoStageRaindropNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.s1_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.s1_enc = NAFBlock(base_channels)
        self.s1_bottleneck = NAFBlock(base_channels)
        self.s1_dec = NAFBlock(base_channels)
        self.s1_out = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

        self.s2_in = nn.Conv2d(base_channels + in_channels, base_channels, 3, 1, 1)
        self.s2_enc = NAFBlock(base_channels)
        self.s2_bottleneck = NAFBlock(base_channels)
        self.s2_dec = NAFBlock(base_channels)
        self.s2_out = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        f1 = self.s1_in(x)
        f1_enc = self.s1_enc(f1)
        f1_b = self.s1_bottleneck(f1_enc)
        f1_dec = self.s1_dec(f1_b)
        stage1_blur = x + self.s1_out(f1_dec)

        f2_in = torch.cat([f1_dec, stage1_blur], dim=1)
        f2 = self.s2_in(f2_in)
        f2_enc = self.s2_enc(f2)
        f2_b = self.s2_bottleneck(f2_enc)
        f2_dec = self.s2_dec(f2_b)
        stage2_clear = stage1_blur + self.s2_out(f2_dec)

        return stage1_blur, stage2_clear

# ==============================================================================
# 2. 測試/推論主要邏輯
# ==============================================================================
def run_test():
    # ---------------- 參數配置 ----------------
    checkpoint_path = './checkpoints/best_model.pth'  # 最佳訓練權重檔
    test_input_dir = 'D:/gitserver/python/raindrop/testdata/DayRainDrop_Train/Drop' # 測試圖片來源目錄
    output_dir = './results_test'                       # 輸出結果對比圖資料夾
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的推論硬體: {device}")

    # 1. 載入模型結構與權重
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: 找不到模型權重檔 `{checkpoint_path}`，請確認路徑或是否已完成訓練。")
        return

    model = FastTwoStageRaindropNet(in_channels=3, base_channels=32).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"成功載入權重！此權重的 Best PSNR 為: {checkpoint.get('best_psnr', 0.0):.2f} dB")

    # 2. 搜尋測試圖片 (抓取第一個場景資料夾下的所有 png/jpg 檔進行測試)
    scene_folders = sorted(os.listdir(test_input_dir))
    test_images = []
    
    for scene in scene_folders[:9]:  # 預設抓取前 9 個場景資料夾做測試
        scene_path = os.path.join(test_input_dir, scene)
        if os.path.isdir(scene_path):
            imgs = glob.glob(os.path.join(scene_path, '*.png')) + glob.glob(os.path.join(scene_path, '*.jpg'))
            test_images.extend(imgs[:5]) # 每個場景抓取前 5 張照片展示

    if not test_images:
        print("未找到任何測試圖片！")
        return

    print(f"找到 {len(test_images)} 張測試圖片，開始進行雨滴與水痕去除推論...")

    # 3. 進行推論與儲存視覺化結果
    with torch.no_grad():
        for idx, img_path in enumerate(test_images):
            # 讀取並轉換圖片 Tensor
            raw_img = Image.open(img_path).convert('RGB')
            input_tensor = TF.to_tensor(raw_img).unsqueeze(0).to(device)

            # 模型推論 (使用全精度或 AMP)
            _, pred_clear = model(input_tensor)
            
            # 將數值限制在 0~1 的合法顏色範圍
            pred_clear = torch.clamp(pred_clear, 0.0, 1.0)

            # 將原始雨滴圖 (Input) 與模型還原圖 (Pred Clear) 並排拼接 (Horizontal Concatenation)
            compared_result = torch.cat([input_tensor, pred_clear], dim=3)

            # 儲存對比圖
            filename = f"test_result_{idx+1:03d}.png"
            save_path = os.path.join(output_dir, filename)
            save_image(compared_result, save_path)
            
            print(f"[{idx+1}/{len(test_images)}] 已生成對比圖: {save_path}")

    print(f"\n測試完成！所有兩兩對比圖（左邊：含雨滴原圖 / 右邊：模型修復結果）已儲存至 `{output_dir}` 資料夾。")

if __name__ == '__main__':
    run_test()