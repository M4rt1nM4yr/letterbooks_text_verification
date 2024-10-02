from torch import nn
from torch.nn import functional as F
from torch.nn import Module

class LightFE(Module):
    def __init__(
            self,
            in_channels=1,
            fine_grained=False,
    ):
        super(LightFE, self).__init__()
        self.act = nn.LeakyReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=(4,2))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(1,2))
        if fine_grained:
            self.conv1 = nn.Conv2d(in_channels, 8, (3, 2), stride=(2,1))
        else:
            self.conv1 = nn.Conv2d(in_channels, 8, (6, 2), stride=(4,2))
        self.conv2 = nn.Conv2d(8, 32, (6, 4), padding='same')
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding='same')
        self.out_dim = 64
        self.height_at_64px = 8 if fine_grained else 4

    def forward(self, x):
        x = F.pad(x,(1,2,1,2))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.max_pool1(x)
        x = self.act(self.conv3(x))
        x = self.max_pool2(x)
        return x


if __name__ == "__main__":
    import torch
    for i in [256,512,1024]:
        x = torch.randn(10,1,64,i)
        m = LightFE(fine_grained=True)
        print(m(x).shape)

