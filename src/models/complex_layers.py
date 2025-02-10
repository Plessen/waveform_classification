import torch.nn as nn
from torch.nn.functional import sigmoid
import torch
from complexNN.nn import cConv1d, cAvgPool2d
import numpy as np

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
    
# Example usage
if __name__ == "__main__":
    # Create a random complex tensor
    x_real = torch.randn(1, 8, 32, 32)
    x_imag = torch.randn(1, 8, 32, 32)
    x = torch.complex(x_real, x_imag)
    
    # Instantiate the CVEfficientChannelAtttention module
    attention_module = CVEfficientChannelAtttention(channels=8)
    
    # Apply the attention module
    output = attention_module(x)
    
    # Print the output
    print("Output shape:", output.shape)
    print("Output:", output)