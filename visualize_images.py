# Python
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Update this to your actual test file path
data_path = r'C:\Users\batistd1\Documents\MATLAB\Thesis\FSSTn-master\FSSTn-master\DataGeneration\data\input_test_nearest_SST.h5'

# Dataset keys (adjust if needed)
clean_key = '/clean_images/images_real'
noisy_key = '/noisy_images/images_real'

# Number of images to display
num_images = 10

with h5py.File(data_path, 'r') as f:
    clean_images = f[clean_key][:num_images]
    noisy_images = f[noisy_key][:num_images]

def show_images(clean, noisy, num):
    fig, axes = plt.subplots(2, num, figsize=(num * 2, 4))
    for i in range(num):
        axes[0, i].imshow(np.squeeze(clean[i]))
        axes[0, i].axis('off')
        axes[1, i].imshow(np.squeeze(noisy[i]))
        axes[1, i].axis('off')
    axes[0, 0].set_title("Clean")
    axes[1, 0].set_title("Noisy")
    plt.tight_layout()
    plt.show()

show_images(clean_images, noisy_images, num_images)