import lightning as L
import torch.nn as nn
import lightning as L
import torch.optim as optim


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

class BaseLitModel(L.LightningModule):
    def __init__(self, model, lr = 0.001):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output = self(noisy_image)
        loss = self.criterion(output, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output = self.model(noisy_image)
        loss = self.criterion(output, label)
        
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output = self.model(noisy_image)
        loss = self.criterion(output, label)
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer
     
class BaseLitModelAutoencoder(L.LightningModule):
    def __init__(self, model, lr = 0.001, freeze = False):
        super().__init__()
        self.model = model
        if freeze:
            self.model = freeze_model(self.model)
            
        self.criterion_classifier = nn.NLLLoss()
        self.criterion_autoencoder = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output, noisy_image, noisy_patches = self(noisy_image)
        loss_classification = self.criterion_classifier(output, label)
        loss_patch_reconstruction = self.criterion_autoencoder(noisy_patches, self.model.denoiser.extract_patches(noisy_image))
        loss_image_denoising = self.criterion_autoencoder(noisy_image, clean_image)
        
        loss = loss_classification + loss_image_denoising + 0.1 * loss_patch_reconstruction
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output, noisy_image, noisy_patches = self.model(noisy_image)
        loss = self.criterion(output, label)
        loss_patch_reconstruction = self.criterion_autoencoder(noisy_patches, self.model.denoiser.extract_patches(noisy_image))
        loss_image_denoising = self.criterion_autoencoder(noisy_image, clean_image)
        
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_patch_reconstruction", loss_patch_reconstruction, on_epoch=True, prog_bar=True)
        self.log("val_image_denoising", loss_image_denoising, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output, _,  _ = self.model(noisy_image)
        loss = self.criterion(output, label)
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer