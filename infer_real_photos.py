import os, sys, glob
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from test import UltimateTwoStageNet

MAX_SIDE = 1024

def run(mode, ckpt_dir, photos, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cpu')
    model = UltimateTwoStageNet(base_channels=64).to(device)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print('SKIP', mode, 'no checkpoint at', ckpt_path)
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    epoch = ckpt.get('epoch','?')
    print('Loaded', mode, 'epoch', epoch)
    with torch.no_grad():
        for p in photos:
            img = Image.open(p).convert('RGB')
            w,h = img.size
            scale = MAX_SIDE / max(w,h)
            if scale < 1.0:
                w, h = int(w*scale), int(h*scale)
                img = img.resize((w,h), Image.LANCZOS)
            nw, nh = (w//16)*16, (h//16)*16
            img = img.crop((0,0,nw,nh))
            inp = TF.to_tensor(img).unsqueeze(0).to(device)
            print('  processing', p, 'shape', inp.shape, flush=True)
            _, pred = model(inp)
            pred = torch.clamp(pred, 0.0, 1.0)
            name = os.path.splitext(os.path.basename(p))[0]
            save_image(pred, os.path.join(out_dir, f'{name}_pred_{mode}.png'))
            save_image(torch.cat([inp, pred], dim=3), os.path.join(out_dir, f'{name}_sidebyside_{mode}.png'))
            print('  ->', name, pred.shape, flush=True)

if __name__ == '__main__':
    photos = sorted(glob.glob('testdata/S__3836314*_0.jpg'))
    print('Photos found:', photos)
    out_dir = 'testdata/real_photo_results'
    run('day', './checkpoints', photos, out_dir)
