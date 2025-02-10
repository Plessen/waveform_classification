import torch.nn as nn

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