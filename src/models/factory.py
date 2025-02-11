from .nn_modules.realcnn import RealConvNet
from .nn_modules.complexcnn import ComplexConvNet, ComplexConvNetAttention, ComplexConvNetDenoise
from .lit_modules import BaseLitModel, BaseLitModelAutoencoder
from ..data.datamodules import SignalDataModule
from ..data.datasets import SignalDatasetComplex, SignalDatasetReal

def model_factory(model_name, data_paths, batch_sizes, num_workers, val_split, lr, image_size = 128, number_patches = 4, checkpoint_path = None, pretrained_model_name = None):
    model_config = {
        "realcnn": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_instance": RealConvNet,
            "model_args": {}
        },
        "complexcnn": {
            "dataset_class": SignalDatasetComplex,
            "lit_model_class": BaseLitModel,
            "model_instance": ComplexConvNet,
            "model_args": {}
        },
        "complexcnn-attention": {
            "dataset_class": SignalDatasetComplex,
            "lit_model_class": BaseLitModel,
            "model_instance": ComplexConvNetAttention,
            "model_args": {}
        },
        "complexcnn-autoencoder": {
            "dataset_class": SignalDatasetComplex,
            "lit_model_class": BaseLitModelAutoencoder,
            "model_instance": ComplexConvNetDenoise,
            "model_args": {"image_size": image_size, "num_patches": number_patches, "model": ComplexConvNet()}
        }
    }

    checkpoint_model = None
    if pretrained_model_name is not None:
        if checkpoint_path is None:
            raise ValueError("Checkpoint path must be provided when using pretrained model")
        
        if pretrained_model_name not in model_config:
            raise ValueError(f"Unsupported Pretrained model: {pretrained_model_name}")
        
        config = model_config[pretrained_model_name]
        checkpoint_model = config["lit_model_class"].load_from_checkpoint(checkpoint_path, model=config["model_instance"](config["model_args"]), lr=lr)
                
    # Validate model name
    if model_name not in model_config:
        raise ValueError(f"Unsupported model: {model_name}")

    config = model_config[model_name]
    model = config["lit_model_class"](config["model_instance"](config["model_args"]), lr)
    if checkpoint_model is not None:
        model.model = checkpoint_model.model

    # Always create data module based on model type
    data_module = SignalDataModule(config["dataset_class"],data_paths,batch_sizes,num_workers,val_split)

    return model, data_module