import lightning as L
from src.utils import parse_args
from src.models.factory import model_factory
import torch

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main(args):
    #L.seed_everything(42, workers=True)
    data_paths = {'train': args.train_data_path, 'test': args.test_data_path}
    batch_sizes = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size}
    model, data_module, lit_module = model_factory(args.architecture, data_paths, batch_sizes, args.num_workers, args.val_split, 
                                       args.learning_rate, image_size=128,
                                       number_patches=16, checkpoint_path_list=args.checkpoint_path_list, 
                                       pretrained_model_name_list=args.pretrained_model_name_list, 
                                       freeze=args.freeze, number_waveforms=args.num_waveforms, 
                                       signals_per_snr=args.signals_per_snr)

    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()
    
    for idx, layer in enumerate(model.model.layers):
        print(f"Index: {idx} -> {layer}")
        
    features = {}
    def hook_fn(m, i, o):
        features["feat"] = o.detach()
    
    layer_to_hook = model.model.layers[-5]
    layer_to_hook.register_forward_hook(hook_fn)
    
    features_list = []
    labels_list = []
    with torch.no_grad():
        for clean_image, noisy_image, label in test_dataloader:
            clean_image = clean_image.cuda()
            noisy_image = noisy_image.cuda()
            label = label.cuda()
            output = model(clean_image)
            features_list.append(features["feat"].cpu())
            labels_list.append(label.cpu())
        
    all_feats = torch.cat(features_list, dim=0).numpy()
    all_labels = torch.cat(labels_list, dim=0).numpy()
    
    print(all_feats.shape)
    print(all_labels.shape)
    
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(all_feats)
    
    # Plot test features in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        pca_features[:, 0],
        pca_features[:, 1],
        pca_features[:, 2],
        c=all_labels,
        cmap='tab10',
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.title("PCA Feature Visualization (Test Set)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()
    
if __name__ == "__main__":
    args = parse_args()
    main(args)