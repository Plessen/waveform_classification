import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.models.nn_modules.complexcnn import ComplexDenoisingAutoencoder
from src.models.lit_modules.baselit import BaseLitModelAutoencoder
from src.models.lit_modules.lossfunctions import ComplexMSELoss

# Parameters
image_size = 128         # adjust as needed
num_patches = 4          # e.g. 2x2 grid
num_images = 10          # number of images to process
data_path = r'C:\Users\batistd1\Documents\MATLAB\Thesis\FSSTn-master\FSSTn-master\DataGeneration\data\input_test_nearest_SST.h5'
checkpoint_path = r'C:\Users\batistd1\Documents\MATLAB\Thesis\DL\waveform_classification\logs\complex_autoencoder_mode_sst_nearest\version_1\checkpoints\complex_autoencoder_mode_sst_nearest-epoch=00-val_loss=0.00-val_acc=0.00.ckpt'
# Set device to CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load test images from HDF5 file.
with h5py.File(data_path, 'r') as f:
    # Load first num_images from clean and noisy datasets.
    # Ensure the images will have shape (num_images, 1, H, W) for the autoencoder.
    clean_images_np = f['/clean_images/images_real'][:num_images]
    noisy_images_np = f['/noisy_images/images_real'][:num_images]

# Convert numpy arrays to torch tensors and add a channel dimension if needed.
# Assuming the numpy arrays are shape (N, H, W), unsqueeze channel axis.
clean_images = torch.tensor(clean_images_np, dtype=torch.complex64).unsqueeze(1)
noisy_images = torch.tensor(noisy_images_np, dtype=torch.complex64).unsqueeze(1)

clean_images = clean_images.to(device)
noisy_images = noisy_images.to(device)

# Load the autoencoder from checkpoint.
autoencoder_model = ComplexDenoisingAutoencoder(image_size=image_size, number_patches=num_patches)
lit_model = BaseLitModelAutoencoder.load_from_checkpoint(checkpoint_path, model=autoencoder_model)
lit_model.model.to(device)
lit_model.model.eval()  # set to evaluation mode

# For backwards compatibility with your patch functions, alias the model.
autoencoder = lit_model.model

# Test patch extraction and recombination for all clean images.
for i in range(clean_images.shape[0]):
    image = clean_images[i:i+1]  # (1,1,H,W)
    patches = autoencoder.extract_patches(image)
    recombined = autoencoder.combine_patches(patches)
    assert torch.allclose(image, recombined, atol=1e-6), \
        f"Image {i} recombination failed; max diff: {(image - recombined).abs().max().item()}"
print("Success: All clean images are correctly reconstructed from patches.")

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
        axes[0, i].imshow(noisy_np[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(clean_np[i], cmap="gray")
        axes[1, i].axis("off")
        axes[2, i].imshow(recon_np[i], cmap="gray")
        axes[2, i].axis("off")
    
    axes[0, 0].set_title("Noisy")
    axes[1, 0].set_title("Clean")
    axes[2, 0].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

visualize_results(noisy_images, clean_images, reconstructed, num_images)