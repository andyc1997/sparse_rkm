import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Description: This script contains the encoder-decoder networks for the dSprites dataset.
'''


# network dimensions
c = 64  # capacity


class FeatMap(nn.Module):
    """Encoder function or feature map"""
    def __init__(self, p_f):
        super(FeatMap, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(c * 4 * 8 * 8, p_f)

    def forward(self, x):
        if x.dim() == 3:
            x = x[None, :, :, :]
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = x.reshape(x.size(0), c * 4 * 8 * 8)
        x = self.fc1(x)
        return x  # p_f


class PreImgMap(nn.Module):
    """Decoder function or preimage map"""
    def __init__(self, p_f):
        super(PreImgMap, self).__init__()
        self.fc1 = nn.Linear(in_features=p_f, out_features=c * 4 * 8 * 8)
        self.conv3 = nn.ConvTranspose2d(in_channels=c * 4, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        if x.dim() == 1:
            x = x.reshape(1, c * 4, 8, 8)
        else:
            x = x.reshape(x.size(0), c * 4, 8, 8)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = torch.sigmoid(self.conv1(x))
        return x  # N x 64 x 64
