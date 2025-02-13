import torch.nn as nn
from torch.nn.functional import sigmoid, interpolate, relu
import torch
from complexNN.nn import cConv1d, cAvgPool2d, cConv2d, cMaxPool2d, cRelu, cSigmoid, cTanh
import numpy as np
from complexPyTorch.complexLayers import ComplexConv2d, ComplexConvTranspose2d, ComplexReLU, ComplexMaxPool2d
from torchinfo import summary

class PhaseSigmoid(nn.Module):
    
    def __init__(self):
        super(PhaseSigmoid, self).__init__()
    
    def forward(self, x):
        magnitude = x.abs()
        return sigmoid(magnitude) * x / magnitude

class cAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super().__init__(output_size)
        self.output_size = output_size

    def forward(self, x):
        return torch.complex(super().forward(x.real), super().forward(x.imag))
    
class CVEfficientChannelAtttention(nn.Module):
    
    def __init__(self, channels: int, b: int = 1, gamma:int = 2):
        super(CVEfficientChannelAtttention, self).__init__()
        self.channels = channels
        self.b = b
        self.gamma = gamma
        
        self.pooling = cAdaptiveAvgPool2d(1)
        self.conv = cConv1d(1, 1, kernel_size=self.kernel_size(), padding="same", bias=False)
        self.sigmoid = PhaseSigmoid()
    

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

class cUpsample2d(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest', align_corners=None):
        super(cUpsample2d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        return torch.complex(interpolate(x.real, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners),
                            interpolate(x.imag, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners))

class ComplexModReLU(nn.Module):
    def __init__(self, num_channels):
        super(ComplexModReLU, self).__init__()
        self.bias = nn.Parameter(torch.randn(num_channels) * 0.02)
    
    def forward(self, x):
        magnitude = torch.abs(x) 
        biased = magnitude + self.bias.view(1, -1, 1, 1)  
        thresholded = relu(biased)  
        scale = thresholded / (magnitude + 1e-6)  
        
        return x * scale  
        
class cPatchAutoencoder(nn.Module):
    
    def __init__(self):
        super(cPatchAutoencoder, self).__init__()
    
        self.encoder = nn.Sequential(
            cConv2d(1, 64, kernel_size=3, padding="same", bias=True),
            cRelu(),
            cMaxPool2d(2, 2),
            cConv2d(64, 64, kernel_size=3, padding="same", bias=True),
            cRelu(),
            cMaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=True, dtype=torch.complex64),
            cRelu(),
            cUpsample2d(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=True, dtype=torch.complex64),
            cRelu(),
            cUpsample2d(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1, bias=True, dtype=torch.complex64),
            #cTanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
            
class cPatchAutoencoderGrouped(nn.Module):
        
    def __init__(self, num_patches):
        super(cPatchAutoencoderGrouped, self).__init__()
        self.num_patches = num_patches
        
        self.encoder = nn.Sequential(
            cConv2d(1 * num_patches, 64 * num_patches, kernel_size=3, padding="same", bias=True, groups=num_patches),
            cRelu(),
            cMaxPool2d(2, 2),
            cConv2d(64 * num_patches, 64 * num_patches, kernel_size=3, padding="same", bias=True, groups=num_patches),
            cRelu(),
            cMaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64 * num_patches, 64 * num_patches, kernel_size=3, padding=1, bias=True, groups=num_patches, dtype=torch.complex64),
            cRelu(),
            cUpsample2d(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64 * num_patches, 64 * num_patches, kernel_size=3, padding=1, bias=True, groups=num_patches, dtype=torch.complex64),
            cRelu(),
            cUpsample2d(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64 * num_patches, 1 * num_patches, kernel_size=3, padding=1, bias=True, groups=num_patches, dtype=torch.complex64),
            #cTanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

