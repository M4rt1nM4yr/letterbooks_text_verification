import torch
from PIL import Image
from torchvision import transforms

class ForegroundChecker(object):
    def __init__(self, foreground_is_one=True):
        self.foreground_is_one = foreground_is_one

    def __call__(self, img):
        input_type = type(img)
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        if (torch.mean(img) > 0.5 and self.foreground_is_one) or (torch.mean(img) < 0.5 and not self.foreground_is_one):
            img = 1-img
        if input_type == Image.Image:
            img = transforms.ToPILImage()(img)
        return img