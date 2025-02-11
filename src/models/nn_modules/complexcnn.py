from complexNN.nn import cConv2d, cLinear, cElu, cMaxPool2d, cAvgPool2d, cDropout, cBatchNorm2d
import torch.nn as nn
from .complex_layers import CVEfficientChannelAtttention, cDenoisingAutoencoder

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
    def __init__(self, image_size, number_patches, model):
        super(ComplexConvNetDenoise, self).__init__()
        self.number_patches = number_patches
        self.denoiser = cDenoisingAutoencoder(image_size, number_patches)
        self.model = model
        
    def forward(self, x):
        combined_image, patches = self.denoiser(x)
        x = self.model(combined_image)
        return x, combined_image, patches