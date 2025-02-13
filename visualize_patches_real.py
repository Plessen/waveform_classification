import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.models.nn_modules.realcnn import RealDenoisingAutoencoder
from src.models.lit_modules.baselit import BaseLitModelAutoencoder
from src.models.lit_modules.lossfunctions import ComplexMSELoss

from src.data.datasets import SignalDatasetReal  # [SignalDatasetReal](src/data/datasets.py)
from src.models.nn_modules.realcnn import RealDenoisingAutoencoder  # [RealDenoisingAutoencoder](src/models/nn_modules/realcnn.py)
from torch.utils.data import DataLoader

# Parameters
image_size = 128         # adjust as needed
num_patches = 4          # e.g. 2x2 grid
num_images = 10          # number of images to process
data_path = r'C:\Users\batistd1\Documents\MATLAB\Thesis\FSSTn-master\FSSTn-master\DataGeneration\data\input_test_nearest_SST.h5'
checkpoint_path = r'C:\Users\batistd1\Documents\MATLAB\Thesis\DL\waveform_classification\logs\real_autoencoder_mode_sst_nearest\version_1\checkpoints\real_autoencoder_mode_sst_nearest-epoch=03-val_loss=0.02-val_acc=0.00.ckpt'
# Set device to CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SignalDatasetReal(data_path)
dataloader = DataLoader(dataset, batch_size=num_images, shuffle=False, num_workers=0)
# Set device and load the first batch from the DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = next(iter(dataloader))
clean_images, noisy_images, labels = batch  # SignalDatasetReal returns (clean, noisy, label)
clean_images = clean_images.to(device)
noisy_images = noisy_images.to(device)
img = clean_images[0].unsqueeze(0).to(device)

# Load the autoencoder from checkpoint.
autoencoder_model = RealDenoisingAutoencoder(image_size=image_size, number_patches=num_patches)
lit_model = BaseLitModelAutoencoder.load_from_checkpoint(checkpoint_path, model=autoencoder_model)
lit_model.model.to(device)
lit_model.model.eval()

# For backwards compatibility with your patch functions, alias the model.
autoencoder = lit_model.model


# Run the autoencoder on the noisy images.
with torch.no_grad():
    # Assumes the forward pass returns a tuple: (denoised_image, denoised_patches)
    reconstructed, _ = autoencoder(noisy_images)

# Compute MSE Loss between reconstructed and clean images.
loss = ComplexMSELoss()(reconstructed, clean_images)
print(f"MSE Loss between reconstructed and original images: {loss.item()}")

# Visualization: For each image, show the noisy input, clean original, and reconstructed output.
def visualize_results(noisy, clean, recon, num):
    # If images are complex, use the magnitude; here images are real.
    noisy_np = noisy.abs().squeeze().cpu().numpy()
    clean_np = clean.abs().squeeze().cpu().numpy()
    recon_np = recon.abs().squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(3, num, figsize=(num*2, 6))
    for i in range(num):
        axes[0, i].imshow(noisy_np[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(clean_np[i])
        axes[1, i].axis("off")
        axes[2, i].imshow(recon_np[i])
        axes[2, i].axis("off")
    
    axes[0, 0].set_title("Noisy")
    axes[1, 0].set_title("Clean")
    axes[2, 0].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

noisy_images = torch.complex(noisy_images[:, 0, :, :], noisy_images[:, 1, :, :])
clean_images = torch.complex(clean_images[:, 0, :, :], clean_images[:, 1, :, :])
reconstructed = torch.complex(reconstructed[:, 0, :, :], reconstructed[:, 1, :, :])
visualize_results(noisy_images, clean_images, reconstructed, num_images)