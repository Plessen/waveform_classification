import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ComplexMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # Split into real and imaginary components
        input_real = input.real
        input_imag = input.imag
        target_real = target.real
        target_imag = target.imag
        
        loss_real = F.mse_loss(input_real, target_real, reduction=self.reduction)
        loss_imag = F.mse_loss(input_imag, target_imag, reduction=self.reduction)
        return loss_real + loss_imag