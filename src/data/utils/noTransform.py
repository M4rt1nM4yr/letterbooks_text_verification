import torch
from torch import nn
from torchvision import transforms
from PIL import Image


class NoTransform(nn.Module):
    def __init__(self, return_type="unchanged", **kwargs):
        super(NoTransform, self).__init__()
        self.return_type = return_type

    def forward(self, x, **kwargs):
        if self.return_type == "unchanged":
            return x
        if self.return_type.lower() == "tensor":
            if isinstance(x, torch.Tensor):
                return x
            toTensor = transforms.ToTensor()
            return toTensor(x)
        if self.return_type.lower() == "pil":
            if isinstance(x, Image.Image):
                return x
            toPIL = transforms.ToPILImage()
            return toPIL(x)