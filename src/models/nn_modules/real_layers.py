import torch.nn as nn
from torch.nn.functional import sigmoid, interpolate, relu
import torch
import numpy as np

class ECA(nn.Module):
    def __init__(self, channels: int, b: int = 1, gamma:int = 2):
        super(ECA, self).__init__()
        self.channels = channels
        self.b = b
        self.gamma = gamma
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding="same", bias=False)
        self.sigmoid = nn.Sigmoid()    

    def kernel_size(self):
        k = int(abs((np.log2(self.channels) + self.b) / self.gamma))
        out = k if k % 2 else k + 1
        return out

    def forward(self, x):
        B, C, _, _ = x.shape
        
        y = self.pooling(x)
        y = self.conv(y.squeeze(-1).view(B, 1, C))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)

class PatchAutoencoder(nn.Module):
    
    def __init__(self):
        super(PatchAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
