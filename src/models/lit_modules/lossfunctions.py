import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ComplexMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # Split into real and imaginary components
        if torch.is_complex(input):
            input_real = input.real
            input_imag = input.imag
            target_real = target.real
            target_imag = target.imag
            
            loss_real = F.mse_loss(input_real, target_real, reduction=self.reduction)
            loss_imag = F.mse_loss(input_imag, target_imag, reduction=self.reduction)
            return loss_real + loss_imag

        weight = torch.where(target != 0, 10, torch.tensor(1.0, device=target.device))
        loss = F.mse_loss(input, target, reduction="none")
        weighted_loss = loss * weight
        return weighted_loss.mean()
        #return F.mse_loss(input, target, reduction=self.reduction)