import lightning as L
import torch.nn as nn
import lightning as L
import torch.optim as optim
from .lossfunctions import ComplexMSELoss, CenterLoss
import torch
from pretty_confusion_matrix import pp_matrix_from_data

class BaseLitModel(L.LightningModule):
    def __init__(self, model, lr = 0.001, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.kwargs = kwargs
        self.preds = []
        self.labels = []
        
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
        
        self.preds.append(output.argmax(dim = 1).detach().cpu())
        self.labels.append(label.detach().cpu())
        
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        
        number_waveforms = self.kwargs.get("number_waveforms")
        signals_per_snr = self.kwargs.get("signals_per_snr")
        
        number_snr = labels.shape[0] // (number_waveforms * signals_per_snr)
        
        if number_waveforms == 8:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4']
        else:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
            
        for i in range(number_snr):
            path_to_save_img = self.logger.log_dir + "/confusion_matrix_{}.png".format(i)
            snr_labels = labels[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            snr_preds = preds[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            pp_matrix_from_data(snr_labels, snr_preds, cmap="viridis", columns=col, path_to_save_img=path_to_save_img)

    def configure_optimizers(self):
        return self.optimizer

class BaseLitModelCenterLoss(L.LightningModule):
    def __init__(self, model, lr = 0.001, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
        self.center_loss = CenterLoss(num_classes=model.number_waveforms, feat_dim=8192, use_gpu=True)
        self.automatic_optimization = False
        self.kwargs = kwargs
        self.preds = []
        self.labels = []
        self.lr = lr
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy_image, label = batch
        output, features = self(noisy_image)
        model_opt, center_opt = self.optimizers()
        
        loss = self.criterion(output, label)
        center_loss = self.center_loss(features, label)
        total_loss = loss + center_loss
        model_opt.zero_grad()
        center_opt.zero_grad()
        self.manual_backward(total_loss)
        model_opt.step()
        center_opt.step()
        
            
        self.log_dict({"train_loss": loss, "center_loss": center_loss}, on_epoch=True, prog_bar=True)
        #return loss
    
    def validation_step(self, batch, batch_idx):
        noisy_image, label = batch
        output, features = self(noisy_image)
        loss = self.criterion(output, label)
        
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        noisy_image, label = batch
        output, features = self(noisy_image)
        loss = self.criterion(output, label)
        acc = (output.argmax(dim = 1) == label).float().mean()
        
        self.preds.append(output.argmax(dim = 1).detach().cpu())
        self.labels.append(label.detach().cpu())
        
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        
        number_waveforms = self.kwargs.get("number_waveforms")
        signals_per_snr = self.kwargs.get("signals_per_snr")
        
        number_snr = labels.shape[0] // (number_waveforms * signals_per_snr)
        
        if number_waveforms == 8:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4']
        else:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
            
        for i in range(number_snr):
            path_to_save_img = self.logger.log_dir + "/confusion_matrix_{}.png".format(i)
            snr_labels = labels[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            snr_preds = preds[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            pp_matrix_from_data(snr_labels, snr_preds, cmap="viridis", columns=col, path_to_save_img=path_to_save_img)

    def configure_optimizers(self):
        optimizers = [optim.Adam(self.model.parameters(), lr=self.lr), optim.Adam(self.center_loss.parameters(), lr=self.lr)]
        return optimizers
   
class BaseLitModelCWD(L.LightningModule):
    def __init__(self, model, lr = 0.001, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.kwargs = kwargs
        self.preds = []
        self.labels = []
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy_image, label = batch
        output = self(noisy_image)
        loss = self.criterion(output, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        noisy_image, label = batch
        output = self.model(noisy_image)
        loss = self.criterion(output, label)
        
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        noisy_image, label = batch
        output = self.model(noisy_image)
        loss = self.criterion(output, label)
        acc = (output.argmax(dim = 1) == label).float().mean()
        
        self.preds.append(output.argmax(dim = 1).detach().cpu())
        self.labels.append(label.detach().cpu())
        
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        
        number_waveforms = self.kwargs.get("number_waveforms")
        signals_per_snr = self.kwargs.get("signals_per_snr")
        
        number_snr = labels.shape[0] // (number_waveforms * signals_per_snr)
        
        if number_waveforms == 8:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4']
        else:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
            
        for i in range(number_snr):
            path_to_save_img = self.logger.log_dir + "/confusion_matrix_{}.png".format(i)
            snr_labels = labels[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            snr_preds = preds[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            pp_matrix_from_data(snr_labels, snr_preds, cmap="viridis", columns=col, path_to_save_img=path_to_save_img)

    def configure_optimizers(self):
        return self.optimizer
    
class BaseLitModelUsingAutoencoder(L.LightningModule):
    def __init__(self, model, lr = 0.001, **kwargs):
        super().__init__()
        self.model = model            
        self.criterion_classifier = nn.NLLLoss()
        self.criterion_autoencoder = ComplexMSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output, denoised_images, denoised_patches = self(noisy_image)
        loss_classification = self.criterion_classifier(output, label)
        loss_patch_reconstruction = self.criterion_autoencoder(denoised_patches, self.model.denoiser.extract_patches(clean_image))
        #loss_image_denoising = self.criterion_autoencoder(denoised_images, clean_image)
        loss = loss_classification + loss_patch_reconstruction
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_patch_reconstruction", loss_patch_reconstruction, on_epoch=True, prog_bar=True)
        #self.log("train_image_denoising", loss_image_denoising, on_epoch=True, prog_bar=True)
        self.log("classification_loss", loss_classification, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output, denoised_images, denoised_patches = self(noisy_image)
        loss = self.criterion_classifier(output, label)
        loss_patch_reconstruction = self.criterion_autoencoder(denoised_patches, self.model.denoiser.extract_patches(clean_image))
        loss_image_denoising = self.criterion_autoencoder(noisy_image, clean_image)
        
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_patch_reconstruction", loss_patch_reconstruction, on_epoch=True, prog_bar=True)
        self.log("val_image_denoising", loss_image_denoising, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output, denoised_images, denoised_patches = self(noisy_image)
        loss = self.criterion_classifier(output, label)
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer

class BaseLitModelAutoencoder(L.LightningModule):
    def __init__(self, model, lr = 0.001, **kwargs):
        super().__init__()
        self.model = model            
        self.criterion = ComplexMSELoss(reduction="mean")
        self.automatic_optimization = False
        self.lr = lr
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        denoised_image, denoised_patches = self(noisy_image)
        #loss = self.criterion(denoised_patches, self.model.extract_patches(clean_image))
        
        clean_patches = self.model.extract_patches(clean_image)
        num_patches = self.model.num_patches
        optimizers = self.optimizers()
        total_patch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for i, optimizer in enumerate(optimizers):
            patch_group = denoised_patches[i::num_patches, ...]
            target_group = clean_patches[i::num_patches, ...]
            patch_loss = self.criterion(patch_group, target_group)
            total_patch_loss = total_patch_loss + patch_loss
        
        for optimizer in optimizers:
            optimizer.zero_grad()
            
        self.manual_backward(total_patch_loss)   
             
        for optimizer in optimizers:
            optimizer.step()
            
        loss_full = self.criterion(clean_image, denoised_image)
        self.log("train_loss", loss_full, on_epoch=True, prog_bar=True)        
        #return loss_full
    
    def validation_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        denoised_image, denoised_patches = self(noisy_image)
        #loss = self.criterion(denoised_patches, self.model.extract_patches(clean_image))
        loss = self.criterion(clean_image, denoised_image)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        denoised_image, denoised_patches = self(noisy_image)
        #loss = self.criterion(denoised_patches, self.model.extract_patches(clean_image))
        loss = self.criterion(clean_image, denoised_image)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizers = [optim.Adam(ae.parameters(), lr=self.lr) for ae in self.model.autoencoders]
        return optimizers

class BaseLitModelGrouped(L.LightningModule):
    def __init__(self, model, lr = 0.001, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output = self(noisy_image)
        
        group_labels = torch.where(torch.logical_or(label == 0, label == 7), torch.tensor(0, device=label.device), torch.tensor(1, device=label.device))
        loss = self.criterion(output, group_labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output = self.model(noisy_image)
        
        group_labels = torch.where(torch.logical_or(label == 0, label == 7), torch.tensor(0, device=label.device), torch.tensor(1, device=label.device))
        loss = self.criterion(output, group_labels)
        
        acc = (output.argmax(dim = 1) == group_labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        clean_image, noisy_image, label = batch
        output = self.model(noisy_image)
        
        group_labels = torch.where(torch.logical_or(label == 0, label == 7), torch.tensor(0, device=label.device), torch.tensor(1, device=label.device))
        loss = self.criterion(output, group_labels)
    
        acc = (output.argmax(dim = 1) == group_labels).float().mean()
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer

class BaseLitModelCWDVSST(L.LightningModule):
    def __init__(self, model, lr = 0.001, **kwargs):
        super().__init__()
        self.model = model
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.kwargs = kwargs
        self.preds = []
        self.labels = []
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        *images, label = batch
        output = self(images)
        loss = self.criterion(output, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        *images, label = batch
        output = self.model(images)
        loss = self.criterion(output, label)
        
        acc = (output.argmax(dim = 1) == label).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        *images, label = batch
        output = self.model(images)
        loss = self.criterion(output, label)
        acc = (output.argmax(dim = 1) == label).float().mean()
        
        self.preds.append(output.argmax(dim = 1).detach().cpu())
        self.labels.append(label.detach().cpu())
        
        self.log("test_acc", acc, on_step=True, prog_bar=True)
        self.log("test_loss", loss, on_step=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds).numpy()
        labels = torch.cat(self.labels).numpy()
        
        number_waveforms = self.kwargs.get("number_waveforms")
        signals_per_snr = self.kwargs.get("signals_per_snr")
        
        number_snr = labels.shape[0] // (number_waveforms * signals_per_snr)
        
        if number_waveforms == 8:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4']
        else:
            col = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
            
        for i in range(number_snr):
            path_to_save_img = self.logger.log_dir + "/confusion_matrix_{}.png".format(i)
            snr_labels = labels[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            snr_preds = preds[i * number_waveforms * signals_per_snr: (i + 1) * number_waveforms * signals_per_snr]
            pp_matrix_from_data(snr_labels, snr_preds, cmap="viridis", columns=col, path_to_save_img=path_to_save_img)

    def configure_optimizers(self):
        return self.optimizer