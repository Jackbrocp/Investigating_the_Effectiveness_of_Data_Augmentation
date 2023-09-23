import torch
from PIL import ImageEnhance
import torchvision.transforms.functional as F


def color(img, magnitude):
    return ImageEnhance.Color(img).enhance(magnitude)


def contrast(img, magnitude):
    return ImageEnhance.Contrast(img).enhance(magnitude)


def brightness(img, magnitude):
    return ImageEnhance.Brightness(img).enhance(magnitude)


def sharpness(img, magnitude):
    return ImageEnhance.Sharpness(img).enhance(magnitude)


def cutout(img, pad_size, replace ):
    img = F.pil_to_tensor(img)
    _, h, w = img.shape
    center_h, center_w = torch.randint(high=h, size=(1,)), torch.randint(high=w, size=(1,))
    low_h, high_h = torch.clamp(center_h-pad_size, 0, h).item(), torch.clamp(center_h+pad_size, 0, h).item()
    low_w, high_w = torch.clamp(center_w-pad_size, 0, w).item(), torch.clamp(center_w+pad_size, 0, w).item()
    cutout_img = img.clone()
    cutout_img[:, low_h:high_h, low_w:high_w] = replace
    return F.to_pil_image(cutout_img)