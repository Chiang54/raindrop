import os
import glob
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

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

class DeepRaindropUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        self.inc = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.enc1 = nn.Sequential(NAFBlock(base_channels), NAFBlock(base_channels))
        
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 2, 2)
        self.enc2 = nn.Sequential(NAFBlock(base_channels * 2), NAFBlock(base_channels * 2))
        
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 2, 2)
        self.enc3 = nn.Sequential(NAFBlock(base_channels * 4), NAFBlock(base_channels * 4))

        self.bottleneck = nn.Sequential(
            NAFBlock(base_channels * 4),
            NAFBlock(base_channels * 4),
            NAFBlock(base_channels * 4)
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.reduce2 = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.dec2 = nn.Sequential(NAFBlock(base_channels * 2), NAFBlock(base_channels * 2))

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.reduce1 = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.dec1 = nn.Sequential(NAFBlock(base_channels), NAFBlock(base_channels))

        self.outc = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

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

        output = x + self.outc(d1)
        return output

def run_test():
    checkpoint_path = './checkpoints/best_model.pth'
    test_input_dir = './testdata/DayRainDrop_Train/Drop'
    output_dir = './results_test'
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint `{checkpoint_path}` not found.")
        return

    model = DeepRaindropUNet(in_channels=3, base_channels=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded Deep UNet Model. Best PSNR: {checkpoint.get('best_psnr', 0.0):.2f} dB")

    scene_folders = sorted(os.listdir(test_input_dir))
    test_images = []
    
    for scene in scene_folders[:9]:
        scene_path = os.path.join(test_input_dir, scene)
        if os.path.isdir(scene_path):
            imgs = glob.glob(os.path.join(scene_path, '*.png')) + glob.glob(os.path.join(scene_path, '*.jpg'))
            test_images.extend(imgs[:5])

    with torch.no_grad():
        for idx, img_path in enumerate(test_images):
            raw_img = Image.open(img_path).convert('RGB')
            
            # Ensure spatial dimensions are divisible by 4 for U-Net downsampling
            w, h = raw_img.size
            new_w = (w // 4) * 4
            new_h = (h // 4) * 4
            raw_img = raw_img.crop((0, 0, new_w, new_h))

            input_tensor = TF.to_tensor(raw_img).unsqueeze(0).to(device)

            pred_clear = model(input_tensor)
            pred_clear = torch.clamp(pred_clear, 0.0, 1.0)

            compared_result = torch.cat([input_tensor, pred_clear], dim=3)

            filename = f"test_hq_result_{idx+1:03d}.png"
            save_path = os.path.join(output_dir, filename)
            save_image(compared_result, save_path)
            print(f"[{idx+1}/{len(test_images)}] Saved: {save_path}")

if __name__ == '__main__':
    run_test()