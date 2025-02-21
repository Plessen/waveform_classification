import torch.nn as nn
from .real_layers import ECA, PatchAutoencoder
import torch
from vit_pytorch import ViT
import matplotlib.pyplot as plt
from vit_pytorch.cct import CCT
from vit_pytorch.cvt import CvT
from.realcbam import CBAM

class RealConvNet(nn.Module):
    
    def __init__(self, number_waveforms):
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
            nn.Linear(32*16*16, int(128 *  number_waveforms  /  8)), 
            nn.ELU(),
            nn.Linear(int(128 *  number_waveforms  /  8), number_waveforms)
        )
    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

class RealConvNetAttention(nn.Module):
    def __init__(self, number_waveforms):
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
            nn.Linear(32*16*16, int(128 *  number_waveforms  /  8)),
            nn.ELU(),
            nn.Linear(int(128 *  number_waveforms  /  8), number_waveforms)
        )
    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

class RealConvNetAttentionCenterLoss(nn.Module):
    def __init__(self, number_waveforms):
        super(RealConvNetAttentionCenterLoss, self).__init__()
        self.number_waveforms = number_waveforms
        self.features = nn.Sequential(
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
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(      
            nn.Dropout(0.5),
            nn.Linear(32*16*16, int(128 *  number_waveforms  /  8)),
            nn.ELU(),
            nn.Linear(int(128 *  number_waveforms  /  8), number_waveforms)
        )
    def forward(self, x):
        features = self.features(x)
        x = self.classifier(features)
        x = nn.functional.log_softmax(x, dim=1)
        return x, features
    
class RealConvNetCBAM(nn.Module):
    def __init__(self, number_waveforms):
        super(RealConvNetCBAM, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2, 8, (3, 3), padding="same"),
            nn.BatchNorm2d(8),
            nn.ELU(),
            CBAM(8, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, (3, 3), padding="same"),
            nn.BatchNorm2d(16),
            nn.ELU(),
            CBAM(16, 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (3, 3), padding="same"),
            nn.BatchNorm2d(32),
            nn.ELU(),
            CBAM(32, 2),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32*16*16, int(128 *  number_waveforms  /  8)),
            nn.ELU(),
            nn.Linear(int(128 *  number_waveforms  /  8), number_waveforms)
        )
    def forward(self, x):
        x = self.layers(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
  
class RealConvNetAttentionGrouped(nn.Module):
    def __init__(self):
        super(RealConvNetAttentionGrouped, self).__init__()
        
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
            nn.Linear(128, 2)
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
    def __init__(self, number_waveforms):
        super(RealViT, self).__init__()
        self.model = ViT(image_size=128, channels=2, patch_size=16, num_classes=number_waveforms, 
                         dim=128, depth=2, heads=4, mlp_dim=128, dropout=0.3)
    
    def forward(self, x):
        x = self.model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

class RealCvT(nn.Module):
    def __init__(self, number_waveforms):
        super(RealCvT, self).__init__()
        self.model = CvT(
            num_classes = number_waveforms,
            s1_emb_dim = 64,        # Stage 1: Embedding dimension
            s1_emb_kernel = 7,      # Convolution kernel for token embedding
            s1_emb_stride = 4,      # Stride for spatial downsampling
            s1_proj_kernel = 3,     # Kernel for attention projection
            s1_kv_proj_stride  = 2,       # Stride for key/value downsampling
            s1_heads = 1,           # Number of attention heads
            s1_depth = 1,           # Number of Transformer blocks
            s2_emb_dim = 128,       # Stage 2: Increased embedding dimension
            s2_emb_kernel = 3,
            s2_emb_stride = 2,
            s2_proj_kernel = 3,
            s2_kv_proj_stride  = 2,
            s2_heads = 2,
            s2_depth = 2,
            s3_emb_dim = 256,       # Stage 3: Final embedding dimension
            s3_emb_kernel = 3,
            s3_emb_stride = 2,
            s3_proj_kernel = 3,
            s3_kv_proj_stride  = 2,
            s3_heads = 4,
            s3_depth = 2,
            dropout=0.3,
            channels = 2            # Input channels (I/Q components)
        )
    
    def forward(self, x):
        x = self.model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
    
class RealCCT(nn.Module):
    def __init__(self, number_waveforms):
        super(RealCCT, self).__init__()
        self.model = CCT(
            img_size = (128, 128),
            num_classes = number_waveforms,
            n_input_channels=2,
            embedding_dim = 128,    
            n_conv_layers = 3,  
            kernel_size = 3,
            stride = 2,
            padding = 1,
            pooling_kernel_size = 2,
            pooling_stride = 2,
            num_layers = 4,        
            num_heads = 4,           
            mlp_ratio = 2,   
            positional_embedding = 'learnable',
            droupout_rate = 0.3
        )
    
    def forward(self, x):
        x = self.model(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x
   
class RealEnsembleClassifier(nn.Module):
    
    def __init__(self, model_classifier, model_group):
        super(RealEnsembleClassifier, self).__init__()
        
        self.model_classifier = model_classifier
        self.model_group = model_group
        
    def forward(self, x):
        outputs = self.model_classifier(x) #log_softmax probabilities
        predictions = outputs.argmax(dim = 1)
        
        group_outputs = self.model_group(x)
        group_predictions = group_outputs.argmax(dim = 1)
        
        num_classes = outputs.size(1)
        num_predictions = outputs.size(0)
        device = outputs.device
        
        grouped_labels = torch.tensor([4, 7], device=device)
        indices = torch.isin(predictions, grouped_labels)
        
        mask = torch.ones((num_predictions, num_classes), device=device)
        
        condition0 = torch.logical_and(group_predictions == 0, indices) #P4 group
        condition1 = torch.logical_and(group_predictions == 1, indices) #P1 group
        #mask[condition0, 4] = float('inf')
        #mask[condition1, 7] = float('inf')

        outputs[condition0, 4] = float('-inf')
        outputs[condition1, 7] = float('-inf')
        #masked_outputs = outputs * mask
        #return masked_outputs
        return outputs