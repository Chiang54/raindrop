import glob
import os

import torch
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from dataset import pad_to_multiple, unpad
from models.generator import UltimateTwoStageNet

# ==============================================================================
DOMAIN = 'day'  # 'day' or 'night' -- must match the model you trained
CHECKPOINT_PATH = f'./checkpoints/{DOMAIN}/best_model.pth'
TEST_INPUT_DIR = f'./testdata/{"Day" if DOMAIN == "day" else "Night"}RainDrop_Train/Drop'
OUTPUT_DIR = './results_test'
BASE_CHANNELS = 96
USE_EMA = True          # EMA weights are smoother/more stable than raw weights
SELF_ENSEMBLE = True    # 8-way D4 geometric test-time augmentation (free quality boost)
# ==============================================================================


def d4_transform(x, mode):
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=[3])
    if mode == 2:
        return torch.flip(x, dims=[2])
    if mode == 3:
        return torch.flip(x, dims=[2, 3])
    if mode == 4:
        return torch.rot90(x, 1, dims=[2, 3])
    if mode == 5:
        return torch.rot90(x, 1, dims=[2, 3]).flip(dims=[3])
    if mode == 6:
        return torch.rot90(x, -1, dims=[2, 3])
    if mode == 7:
        return torch.rot90(x, -1, dims=[2, 3]).flip(dims=[3])
    raise ValueError(mode)


def d4_inverse(x, mode):
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=[3])
    if mode == 2:
        return torch.flip(x, dims=[2])
    if mode == 3:
        return torch.flip(x, dims=[2, 3])
    if mode == 4:
        return torch.rot90(x, -1, dims=[2, 3])
    if mode == 5:
        return torch.rot90(x.flip(dims=[3]), -1, dims=[2, 3])
    if mode == 6:
        return torch.rot90(x, 1, dims=[2, 3])
    if mode == 7:
        return torch.rot90(x.flip(dims=[3]), 1, dims=[2, 3])
    raise ValueError(mode)


def run_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的推論硬體: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"錯誤: 找不到模型權重檔 `{CHECKPOINT_PATH}`。")
        return

    model = UltimateTwoStageNet(base_channels=BASE_CHANNELS).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_key = 'ema_state_dict' if (USE_EMA and 'ema_state_dict' in checkpoint) else 'model_state_dict'
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    print(f"成功載入權重（{state_key}），Best PSNR: {checkpoint.get('best_psnr', 0.0):.2f} dB")

    if not os.path.isdir(TEST_INPUT_DIR):
        print(f"錯誤: 找不到測試資料夾 `{TEST_INPUT_DIR}`。")
        return

    scene_folders = sorted(os.listdir(TEST_INPUT_DIR))
    test_images = []
    for scene in scene_folders[:9]:
        scene_path = os.path.join(TEST_INPUT_DIR, scene)
        if os.path.isdir(scene_path):
            imgs = glob.glob(os.path.join(scene_path, '*.png')) + glob.glob(os.path.join(scene_path, '*.jpg'))
            test_images.extend(imgs[:5])

    if not test_images:
        print("未找到任何測試圖片！")
        return

    print(f"找到 {len(test_images)} 張測試圖片，開始推論（self-ensemble={SELF_ENSEMBLE}）...")

    with torch.no_grad():
        for idx, img_path in enumerate(test_images):
            raw_img = Image.open(img_path).convert('RGB')
            input_tensor = TF.to_tensor(raw_img).unsqueeze(0).to(device)

            # Reflect-pad instead of cropping to a multiple of 16, so no pixels at
            # the right/bottom edge are silently discarded before inference.
            padded, (h, w) = pad_to_multiple(input_tensor, 16)

            modes = range(8) if SELF_ENSEMBLE else [0]
            acc = torch.zeros_like(padded)
            for m in modes:
                aug = d4_transform(padded, m)
                _, pred_clear, _ = model(aug)
                acc += d4_inverse(pred_clear, m)
            pred_clear = acc / len(list(modes))
            pred_clear = unpad(pred_clear, h, w)
            pred_clear = torch.clamp(pred_clear, 0.0, 1.0)

            compared_result = torch.cat([input_tensor, pred_clear], dim=3)

            filename = f"test_ultimate_result_{idx+1:03d}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            save_image(compared_result, save_path)

            print(f"[{idx+1}/{len(test_images)}] 已生成對比圖: {save_path}")

    print(f"\n測試完成！所有對比圖已儲存至 `{OUTPUT_DIR}` 資料夾。")


if __name__ == '__main__':
    run_test()
