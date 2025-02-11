from .nn_modules.realcnn import RealConvNet
from .nn_modules.complexcnn import ComplexConvNet, ComplexConvNetAttention, ComplexConvNetDenoise
from .lit_modules import BaseLitModel, BaseLitModelAutoencoder
from ..data.datamodules import SignalDataModule
from ..data.datasets import SignalDatasetComplex, SignalDatasetReal

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def model_factory(model_name, data_paths, batch_sizes, num_workers, val_split, lr, image_size=128, number_patches=4, checkpoint_path=None, pretrained_model_name=None, freeze = False):
    model_config = {
        "realcnn": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_class": RealConvNet,
            "model_args": {}
        },
        "complexcnn": {
            "dataset_class": SignalDatasetComplex,
            "lit_model_class": BaseLitModel,
            "model_class": ComplexConvNet,
            "model_args": {}
        },
        "complexcnn-attention": {
            "dataset_class": SignalDatasetComplex,
            "lit_model_class": BaseLitModel,
            "model_class": ComplexConvNetAttention,
            "model_args": {}
        },
        "complexcnn-autoencoder": {
            "dataset_class": SignalDatasetComplex,
            "lit_model_class": BaseLitModelAutoencoder,
            "model_class": ComplexConvNetDenoise,
            "model_args": {"image_size": image_size, "number_patches": number_patches}
        }
    }

    # Load pretrained model if specified
    pretrained_model = None
    if pretrained_model_name:
        if checkpoint_path is None:
            raise ValueError("Checkpoint path required for pretrained model")
        if pretrained_model_name not in model_config:
            raise ValueError(f"Invalid pretrained model: {pretrained_model_name}")
        
        pretrained_cfg = model_config[pretrained_model_name]
        pretrained_model = pretrained_cfg["lit_model_class"].load_from_checkpoint(checkpoint_path,model=pretrained_cfg["model_class"](**pretrained_cfg["model_args"]),lr=lr).model
        if freeze:
            pretrained_model = freeze_model(pretrained_model)
            
    # Handle autoencoder special case
    if model_name == "complexcnn-autoencoder":
        if not pretrained_model:
            model_config["complexcnn-autoencoder"]["model_args"]["model"] = ComplexConvNet()
        else: 
            model_config["complexcnn-autoencoder"]["model_args"]["model"] = pretrained_model

    # Validate model name
    if model_name not in model_config:
        raise ValueError(f"Unsupported model: {model_name}")
    cfg = model_config[model_name]

    # Instantiate model
    model_instance = cfg["model_class"](**cfg["model_args"])
    if checkpoint_path and not pretrained_model_name:
        model = cfg["lit_model_class"].load_from_checkpoint(checkpoint_path, model=model_instance, lr=lr)
    else:
        model = cfg["lit_model_class"](model_instance, lr)

    # Create datamodule
    data_module = SignalDataModule(cfg["dataset_class"], data_paths, batch_sizes, num_workers, val_split)
    
    return model, data_module