import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast, adjust_brightness
from kornia import morphology

from src.data.sample_augmentation.utils.degrade import *

# data augmentation based on https://github.com/NVlabs/ocrodeg
class OcrodegAug(nn.Module):
    def __init__(
            self,
            p_random_vert_pad=0.2,
            p_random_hori_pad=0.2,
            p_random_squeeze_stretch=0.2,
            p_dilation=0.2,
            p_erosion=0.2,
            p_distort_with_noise=0.2,
            p_background_noise=0.2,
            p_slant_augmentation=0.2,
            background_noise_n_inks=10,
            p_contrast=0.2,
            p_brightness=0.2,
    ):
        super(OcrodegAug, self).__init__()
        self.p_random_vert_pad = p_random_vert_pad
        self.p_random_hori_pad = p_random_hori_pad
        self.p_random_squeeze_stretch = p_random_squeeze_stretch
        self.p_dilation = p_dilation
        self.p_erosion = p_erosion
        self.p_distort_with_noise = p_distort_with_noise
        self.p_background_noise = p_background_noise
        self.noise_bg = FastPrintlike(n_inks=background_noise_n_inks) if self.p_background_noise > 0 else None
        self.p_slant_augmentation = p_slant_augmentation
        self.p_contrast = p_contrast
        self.p_brightness = p_brightness

        self.toTensor = transforms.ToTensor()

    # TODO: change all of that shit below to a standardized x of the shape channels x height x width
    def __call__(self, x):
        x = np.array(x)

        x = x / (x.max() if x.max()>0 else 1)
        pad_max = np.zeros(4)
        # TODO: Fix padding for RGB
        if self.p_random_vert_pad > torch.rand(1):
            pad_max[1] = x.shape[0] // 4
        if self.p_random_hori_pad > torch.rand(1):
            pad_max[3] = x.shape[0] * 2
        if np.sum(pad_max)>0:
            x = random_pad(x, border=pad_max)

        if self.p_random_squeeze_stretch > torch.rand(1):
            fx = np.random.uniform(low=0.7,high=1.6)
            x = cv2.resize(x, None, fx=fx, fy=1, interpolation=cv2.INTER_LINEAR)

        if self.p_dilation >= torch.rand(1):
            kernel = torch.ones(tuple(torch.randint(low=3,high=6,size=(2,))))
            if len(x.shape) == 3 and x.shape[2] == 3:
                x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            else:
                x = torch.from_numpy(x).view(1, 1, x.shape[0], x.shape[1])
            x = morphology.erosion(x,kernel).permute(0,2,3,1).squeeze().numpy()

        if self.p_erosion >= torch.rand(1):
            kernel = torch.ones(2,2)
            if len(x.shape)==3 and x.shape[2]==3:
                x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
            else:
                x = torch.from_numpy(x).view(1,1,x.shape[0],x.shape[1])
            x = morphology.dilation(x,kernel).permute(0,2,3,1).squeeze().numpy()

        # TODO: fix for RGB
        for sigma in [2,5]:
            if self.p_distort_with_noise > torch.rand(1):
                noise = bounded_gaussian_noise(x.shape, sigma, 3.0)
                x = distort_with_noise(x, noise)

        x = x / (x.max() if x.max() > 0 else 1)
        # TODO: fix for RGB
        if self.p_background_noise > torch.rand(1):
            x = 1-self.noise_bg(x)

        if self.p_slant_augmentation > torch.rand(1):
            x = slant_augmentation(x)

        x = Image.fromarray((x*255).astype(np.uint8))

        if self.p_contrast > torch.rand(1):
            factor = np.random.uniform(0.2, 0.4)
            x = adjust_contrast(x, factor)

        if self.p_brightness > torch.rand(1):
            factor = np.random.uniform(0.01,
                                       0.5)
            x = adjust_brightness(x, factor)

        return x
