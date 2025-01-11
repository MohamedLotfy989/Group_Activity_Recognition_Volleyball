import os
import yaml
import torch
import random
import numpy as np
from PIL import Image, ImageEnhance


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def save_config(config, filepath):
    """Save configuration to a YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config.__dict__, f)


def load_config(filepath):
    """Load configuration from a YAML file."""
    with open(filepath, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)

def save_checkpoint(model, optimizer, epoch, save_dir, filename='checkpoint.pth'):
    """Save model and optimizer state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')


def load_checkpoint(filepath, model, optimizer=None):
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RandomScale:
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = random.uniform(*self.scale_range)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.BILINEAR)

class RandomBrightnessContrast:
    def __init__(self, brightness_range=(0.85, 1.15), contrast_range=(0.85, 1.15)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(*self.brightness_range))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(*self.contrast_range))
        return img

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# Example usage:
# save_config(config, 'config.json')
# loaded_config = load_config('config.json')
