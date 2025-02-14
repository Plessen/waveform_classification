import torch
import matplotlib.pyplot as plt
from src.models.nn_modules.realcnn import RealDenoisingAutoencoder

def test_reassembly(image_size=128, num_patches=4):
    # Create a synthetic single-channel image
    # shape: (1, 1, image_size, image_size)
    img = torch.randn(32, 1, image_size, image_size, dtype=torch.float32)

    # Instantiate the autoencoder (ensure line 106 in realcnn.py is commented out)
    model = RealDenoisingAutoencoder(image_size=image_size, number_patches=num_patches)
    model.eval()

    # Extract patches
    patches = model.extract_patches(img)

    
    combined, denoised_patches = model(img)
    # Recombine patches
    #reassembled = model.combine_patches(patches)
    reassembled = combined
    # Check if images match
    max_diff = (img - reassembled).abs().max().item()
    if torch.allclose(img, reassembled, atol=1e-6):
        print("Success: The reassembled image matches the original. Max diff:", max_diff)
    else:
        print("Reassembly mismatch! Max diff:", max_diff)

if __name__ == "__main__":
    test_reassembly()