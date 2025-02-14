import torch.nn as nn
from .real_layers import ECA, PatchAutoencoder
import torch
from vit_pytorch import ViT
import matplotlib.pyplot as plt

class RealConvNet(nn.Module):
    
    def __init__(self):
        super(RealConvNet, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(2, 8, (3, 3), padding="same"),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, (3, 3), padding="same"),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (3, 3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32*16*16, 128),
            nn.ELU(),
            nn.Linear(128, 8)
        )
    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

class RealConvNetAttention(nn.Module):
    def __init__(self):
        super(RealConvNetAttention, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(2, 8, (3, 3), padding="same"),
            nn.BatchNorm2d(8),
            nn.ELU(),
            ECA(8),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, (3, 3), padding="same"),
            nn.BatchNorm2d(16),
            nn.ELU(),
            ECA(16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (3, 3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ELU(),
            ECA(32),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32*16*16, 128),
            nn.ELU(),
            nn.Linear(128, 8)
        )
    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

class RealConvNetDenoise(nn.Module):
    def __init__(self, model, autoencoder):
        super(RealConvNetDenoise, self).__init__()
        self.denoiser = autoencoder
        self.model = model
        
    def forward(self, x):
        combined_image, patches = self.denoiser(x)
        x = self.model(combined_image)
        return x, combined_image, patches

class RealDenoisingAutoencoder(nn.Module):
    def __init__(self, image_size, number_patches):
        super(RealDenoisingAutoencoder,  self).__init__()
        self.image_size = image_size
        self.num_patches = number_patches
        self.patches_per_dim = int(number_patches**0.5)
        self.patch_size = image_size // self.patches_per_dim

        self.autoencoders = nn.ModuleList([PatchAutoencoder() for _ in range(self.num_patches)])
        
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
        
        denoised_patches = torch.zeros_like(patches)
        for i in range(self.num_patches):
            patch_group = patches[i::self.num_patches, ...]
            denoised_patch_group = self.autoencoders[i](patch_group)
            denoised_patches[i::self.num_patches, ...] = denoised_patch_group

        combined = self.combine_patches(denoised_patches)
        return combined, denoised_patches
    
class RealViT(nn.Module):
    def __init__(self):
        super(RealViT, self).__init__()
        self.model = ViT(image_size=128, channels=2, patch_size=16, num_classes=8, 
                         dim=128, depth=6, heads=16, mlp_dim=256, dropout=0.3)
    
    def forward(self, x):
        x = self.model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x