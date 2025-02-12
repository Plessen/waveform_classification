from complexNN.nn import cConv2d, cLinear, cElu, cMaxPool2d, cAvgPool2d, cDropout, cBatchNorm2d
import torch.nn as nn
from .complex_layers import CVEfficientChannelAtttention, cPatchAutoencoder, cPatchAutoencoderGrouped
import torch

class ComplexConvNet(nn.Module):
    
    def __init__(self):
        super(ComplexConvNet, self).__init__()
        
        self.layers = nn.Sequential(
            cConv2d(1, 8, (3, 3), padding="same"),
            cBatchNorm2d(8),
            cElu(),
            cMaxPool2d(2, 2),
            cConv2d(8, 16, (3, 3), padding="same"),
            cBatchNorm2d(16),
            cElu(),
            cMaxPool2d(2, 2),
            cConv2d(16, 32, (3, 3), padding="same"),
            cBatchNorm2d(32),
            cElu(),
            cAvgPool2d(2, 2),
            nn.Flatten(),
            cDropout(0.5),
            cLinear(32*16*16, 128),
            cElu(),
            cLinear(128, 8)
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = x.abs()
        x = nn.functional.log_softmax(x, dim=1)
        return x

class ComplexConvNetAttention(nn.Module):
        
    def __init__(self):
        super(ComplexConvNetAttention, self).__init__()
        
        self.layers = nn.Sequential(
            cConv2d(1, 8, (3, 3), padding="same"),
            cBatchNorm2d(8),
            cElu(),
            CVEfficientChannelAtttention(8),
            cMaxPool2d(2, 2),
            cConv2d(8, 16, (3, 3), padding="same"),
            cBatchNorm2d(16),
            cElu(),
            CVEfficientChannelAtttention(16),
            cMaxPool2d(2, 2),
            cConv2d(16, 32, (3, 3), padding="same"),
            cBatchNorm2d(32),
            cElu(),
            CVEfficientChannelAtttention(32),
            cAvgPool2d(2, 2),
            nn.Flatten(),
            cDropout(0.5),
            cLinear(32*16*16, 128),
            cElu(),
            cLinear(128, 8)
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = x.abs()
        x = nn.functional.log_softmax(x, dim=1)
        return x

class ComplexConvNetDenoise(nn.Module):
    def __init__(self, model, autoencoder):
        super(ComplexConvNetDenoise, self).__init__()
        self.denoiser = autoencoder
        self.model = model
        
    def forward(self, x):
        combined_image, patches = self.denoiser(x)
        x = self.model(combined_image)
        return x, combined_image, patches

class ComplexDenoisingAutoencoder(nn.Module):
    def __init__(self, image_size, number_patches):
        super(ComplexDenoisingAutoencoder, self).__init__()
        self.image_size = image_size
        self.num_patches = number_patches
        self.patches_per_dim = int(number_patches**0.5)
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
    
class ComplexDenoisingAutoencoderGrouped(nn.Module):
    def __init__(self, image_size, number_patches):
        super(ComplexDenoisingAutoencoderGrouped, self).__init__()
        self.image_size = image_size
        self.num_patches = number_patches
        self.patches_per_dim = int(number_patches**0.5)
        self.patch_size = image_size // self.patches_per_dim

        self.autoencoder = cPatchAutoencoderGrouped(number_patches)

    def extract_patches(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)  # [B, C, num_patches, patch_size, patch_size]
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [B, num_patches, C, patch_size, patch_size]
        patches = patches.view(B, self.num_patches * C, self.patch_size, self.patch_size)  # [B, num_patches * C, patch_size, patch_size]
        return patches

    def combine_patches(self, patches):
        B, C_times_num_patches, H_patch, W_patch = patches.shape
        C = C_times_num_patches // self.num_patches
        patches = patches.view(B, self.num_patches, C, H_patch, W_patch)  # [B, num_patches, C, patch_size, patch_size]
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, num_patches, patch_size, patch_size]
        combined = patches.view(B, C, self.patches_per_dim * H_patch, self.patches_per_dim * W_patch)  # [B, C, H, W]
        return combined

    def forward(self, x):
        patches = self.extract_patches(x) 
        denoised_patches = self.autoencoder(patches)
        combined = self.combine_patches(denoised_patches)
        return combined, denoised_patches


def test_complex_denoising_autoencoder_grouped():
    # Example parameters:
    image_size = 128
    num_patches = 4  # For a 2x2 patch grid (total 4 patches)
    B, C = 64, 1    # Batch of 4, 1 channel (complex)
    
    # Create a sample input: a complex tensor of shape [B, C, image_size, image_size]
    input_tensor = torch.randn(B, C, image_size, image_size, dtype=torch.cfloat)
    
    # Initialize the model
    model = ComplexDenoisingAutoencoderGrouped(image_size=image_size, number_patches=num_patches)
    
    # Forward pass
    output_tensor, patches = model(input_tensor)
    
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("Patches shape:", patches.shape)
if __name__ == "__main__":
    test_complex_denoising_autoencoder_grouped()