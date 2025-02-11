import torch.nn as nn
from torch.nn.functional import sigmoid, interpolate, relu
import torch
from complexNN.nn import cConv1d, cAvgPool2d, cConv2d, cMaxPool2d, cRelu
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
            ComplexConv2d(1, 64, kernel_size=3, padding="same", bias=True),
            ComplexReLU(),
            ComplexMaxPool2d(2, 2),
            ComplexConv2d(64, 64, kernel_size=3, padding="same", bias=True),
            ComplexReLU(),
            ComplexMaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            ComplexConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=True),
            ComplexReLU(),
            cUpsample2d(scale_factor=2, mode='nearest'),
            ComplexConvTranspose2d(64, 64, kernel_size=3, padding=1, bias=True),
            ComplexReLU(),
            cUpsample2d(scale_factor=2, mode='nearest'),
            ComplexConvTranspose2d(64, 1, kernel_size=3, padding=1, bias=True),
            PhaseSigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ComplexDenoisingAutoencoder(nn.Module):
    def __init__(self, image_size, num_patches):
        super(ComplexDenoisingAutoencoder, self).__init__()
        self.image_size = image_size
        self.num_patches = num_patches
        self.patches_per_dim = int(num_patches**0.5)
        self.patch_size = image_size // self.patches_per_dim
        
        self.autoencoders = nn.ModuleList([cPatchAutoencoder() for _ in range(self.num_patches)])

    def extract_patches(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(2, 0, 1, 3, 4).contiguous().view(-1, C, self.patch_size, self.patch_size)
        return patches

    def combine_patches(self, patches):
        B = patches.shape[0] // self.num_patches
        patches = patches.view(self.patches_per_dim, self.patches_per_dim, B, -1, self.patch_size, self.patch_size)
        patches = patches.permute(2, 3, 0, 4, 1, 5).contiguous()
        combined = patches.view(B, -1, self.patches_per_dim * self.patch_size, self.patches_per_dim * self.patch_size)
        return combined

    def forward(self, x):
        patches = self.extract_patches(x)
        denoised_patches = []
        for i, autoencoder in enumerate(self.autoencoders):
            patch_group = patches[i::self.num_patches, ...]
            denoised_patch_group = autoencoder(patch_group)
            denoised_patches.append(denoised_patch_group)
        
        denoised_patches = torch.cat(denoised_patches, dim=0)
        combined = self.combine_patches(denoised_patches)
        return combined, denoised_patches
        
def test_extract_combine():
    image_size = 128
    num_patches = 4
    B, C = 256, 1  # Batch size 1, 3 channels
    x = torch.randn(B, C, image_size, image_size, dtype=torch.complex64)

    model = ComplexDenoisingAutoencoder(image_size, num_patches)

    output, patches = model(x)
    print(output.shape)
    print(patches.shape)
    
def test_cpatchautoencoder_summary():
    model = ComplexDenoisingAutoencoder(128, 4)
    summary(model, input_size=(256, 1, 128, 128), dtypes=[torch.complex64])

if __name__ == "__main__":
    test_extract_combine()