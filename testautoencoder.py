# Python
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.models.lit_modules.baselit import BaseLitModelAutoencoder  # [BaseLitModelAutoencoder](src/models/lit_modules/baselit.py)
from src.models.nn_modules.complexcnn import ComplexDenoisingAutoencoder  # [ComplexDenoisingAutoencoder](src/models/nn_modules/complexcnn.py)
from src.data.datasets import SignalDatasetComplex  # [SignalDatasetComplex](src/data/datasets.py)

# Set your checkpoint and test data paths
checkpoint_path = r'C:\Users\batistd1\Documents\MATLAB\Thesis\DL\waveform_classification\logs\complex_autoencoder_mode_sst_nearest\version_0\checkpoints\complex_autoencoder_mode_sst_nearest-epoch=23-val_loss=0.00-val_acc=0.00.ckpt'
test_data_path = r'C:\Users\batistd1\Documents\MATLAB\Thesis\FSSTn-master\FSSTn-master\DataGeneration\data\input_test_nearest_SST.h5'  # adjust as needed

# Hyperparameters
image_size = 128         # adjust depending on your model
num_patches = 4          # assumes a 2x2 grid, total 4 autoencoders
batch_size = 10

# Set device and move model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load model
autoencoder = ComplexDenoisingAutoencoder(image_size=image_size, number_patches=num_patches)
lit_model = BaseLitModelAutoencoder.load_from_checkpoint(checkpoint_path, model=autoencoder)
lit_model.model.eval()
lit_model.model.to(device)

# Create the test dataset and dataloader
test_dataset = SignalDatasetComplex(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Get the first batch (10 images)
clean_imgs, noisy_imgs, labels = next(iter(test_loader))
noisy_imgs = noisy_imgs.to(device)

# Run the autoencoder on the noisy test images
with torch.no_grad():
    # Assumes your autoencoder returns a tuple: (denoised_image, denoised_patches)
    denoised_imgs, _ = lit_model.model(noisy_imgs)

def visualize_results(inputs, outputs, num_images=2):
    # For complex tensors, use their magnitude for visualization.
    if torch.is_complex(inputs):
        inputs = inputs.abs()
    if torch.is_complex(outputs):
        outputs = outputs.abs()
    
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    for i in range(num_images):
        axes[0, i].imshow(inputs[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_title("Input")
    axes[1, 0].set_title("Output")
    plt.tight_layout()
    plt.show()

visualize_results(noisy_imgs, denoised_imgs)