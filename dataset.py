import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def build_scene_split(root_dir, val_ratio=0.1, seed=42):
    """Scene-level split so no frame from a held-out scene leaks into training.

    Splitting the flattened image list (as before) put near-duplicate burst frames
    from the same scene on both sides of the split, which silently inflated
    validation PSNR. Splitting by scene folder fixes that.
    """
    drop_dir = os.path.join(root_dir, 'Drop')
    scenes = [s for s in sorted(os.listdir(drop_dir)) if os.path.isdir(os.path.join(drop_dir, s))]
    rng = random.Random(seed)
    rng.shuffle(scenes)
    val_count = max(1, int(len(scenes) * val_ratio))
    val_scenes = sorted(scenes[:val_count])
    train_scenes = sorted(scenes[val_count:])
    return train_scenes, val_scenes


def pad_to_multiple(x, multiple=16):
    """Reflect-pad a NCHW tensor so H/W are divisible by `multiple`.

    Used instead of hard-cropping at inference time so no image content is
    discarded just to satisfy the U-Net's downsampling factor.
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, (h, w)


def unpad(x, h, w):
    return x[:, :, :h, :w]


class TripletRaindropDataset(Dataset):
    def __init__(self, root_dir, scene_list, crop_size=384, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train

        self.drop_dir = os.path.join(root_dir, 'Drop')
        self.blur_dir = os.path.join(root_dir, 'Blur')
        self.clear_dir = os.path.join(root_dir, 'Clear')

        self.image_triplets = []
        for scene in scene_list:
            scene_drop_path = os.path.join(self.drop_dir, scene)
            if not os.path.isdir(scene_drop_path):
                continue
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
            _, h, w = tensor_drop.shape
            if h < self.crop_size or w < self.crop_size:
                new_h = max(h, self.crop_size)
                new_w = max(w, self.crop_size)
                tensor_drop = TF.resize(tensor_drop, [new_h, new_w], antialias=True)
                tensor_blur = TF.resize(tensor_blur, [new_h, new_w], antialias=True)
                tensor_clear = TF.resize(tensor_clear, [new_h, new_w], antialias=True)

            if torch.rand(1) > 0.5:
                scale_factor = torch.empty(1).uniform_(0.75, 1.0).item()
                _, h, w = tensor_drop.shape
                new_h = max(int(h * scale_factor), self.crop_size)
                new_w = max(int(w * scale_factor), self.crop_size)
                tensor_drop = TF.resize(tensor_drop, [new_h, new_w], antialias=True)
                tensor_blur = TF.resize(tensor_blur, [new_h, new_w], antialias=True)
                tensor_clear = TF.resize(tensor_clear, [new_h, new_w], antialias=True)

            i, j, th, tw = T.RandomCrop.get_params(tensor_drop, output_size=(self.crop_size, self.crop_size))
            tensor_drop = TF.crop(tensor_drop, i, j, th, tw)
            tensor_blur = TF.crop(tensor_blur, i, j, th, tw)
            tensor_clear = TF.crop(tensor_clear, i, j, th, tw)

            if torch.rand(1) > 0.5:
                tensor_drop = TF.hflip(tensor_drop)
                tensor_blur = TF.hflip(tensor_blur)
                tensor_clear = TF.hflip(tensor_clear)

            # Same jitter applied to all three layers -- preserves the physical
            # relationship between Drop/Blur/Clear while adding lighting diversity.
            if torch.rand(1) > 0.5:
                b = torch.empty(1).uniform_(0.85, 1.15).item()
                c = torch.empty(1).uniform_(0.85, 1.15).item()
                s = torch.empty(1).uniform_(0.85, 1.15).item()
                tensor_drop = torch.clamp(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(tensor_drop, b), c), s), 0.0, 1.0)
                tensor_blur = torch.clamp(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(tensor_blur, b), c), s), 0.0, 1.0)
                tensor_clear = torch.clamp(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(tensor_clear, b), c), s), 0.0, 1.0)
        # is_train=False: return full-resolution tensors untouched; the caller pads
        # to a valid multiple (pad_to_multiple) right before feeding the model.

        return tensor_drop, tensor_blur, tensor_clear
