from complexNN.nn import cConv2d, cLinear, cElu, cMaxPool2d, cAvgPool2d, cDropout, cBatchNorm2d
import torch.nn as nn

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