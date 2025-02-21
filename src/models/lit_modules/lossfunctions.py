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


        return F.mse_loss(input, target, reduction=self.reduction) + F.l1_loss(input, target, reduction=self.reduction)

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # Get the centers corresponding to the labels
        centers_batch = self.centers[labels]
        
        # Compute the mean squared error between the features and their corresponding centers
        loss = F.mse_loss(x, centers_batch)
        
        return loss