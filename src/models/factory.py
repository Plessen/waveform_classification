from .nn_modules.realcnn import RealConvNet, RealCWDVSST, RealConvNetAttentionCWD, RealConvNetAttention, RealConvNetDenoise, RealDenoisingAutoencoder, RealViT, RealConvNetAttentionGrouped, RealEnsembleClassifier, RealCCT, RealCvT,RealConvNetCBAM, RealConvNetAttentionCenterLoss
from .nn_modules.complexcnn import ComplexConvNet, ComplexConvNetAttention, ComplexConvNetDenoise, ComplexDenoisingAutoencoder, ComplexDenoisingAutoencoderGrouped
from .lit_modules import BaseLitModel,BaseLitModelCWDVSST, BaseLitModelCWD, BaseLitModelAutoencoder, BaseLitModelUsingAutoencoder, BaseLitModelGrouped, BaseLitModelCenterLoss
from ..data.datamodules import SignalDataModule
from ..data.datasets import SignalDatasetComplex, SignalDatasetReal, SignalDatasetCWD, SignalDatasetCombined, SignalDatasetWSST

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_pretrained_model(model_config, model_name_list, chekpoint_path_list, lr, number_waveforms, signals_per_snr, freeze):
    loaded_models = []
    for model_name, ckpt_path in zip(model_name_list, chekpoint_path_list):
        if model_name not in model_config:
            raise ValueError(f"Unsupported model: {model_name}")
        cfg = model_config[model_name]

        # Instantiate model
        model_instance = cfg["model_class"](**cfg["model_args"])
        model = cfg["lit_model_class"].load_from_checkpoint(ckpt_path, model=model_instance, lr=lr, 
                                                            number_waveforms=number_waveforms, signals_per_snr = signals_per_snr)
        base_model = model.model

        if freeze:
            base_model = freeze_model(base_model)
        loaded_models.append(base_model)
    
    return loaded_models

def model_factory(model_name, data_paths, batch_sizes, num_workers, val_split, lr, image_size=128, number_patches=16, checkpoint_path_list=[], pretrained_model_name_list=[], freeze = False, number_waveforms = 8, signals_per_snr = 1000):

    model_config = {
        "realcnn": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_class": RealConvNet,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-attention": {
            "dataset_class": SignalDatasetWSST,
            "lit_model_class": BaseLitModelCWD,
            "model_class": RealConvNetAttention,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-attention-slower": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModelCWD,
            "model_class": RealConvNetAttention,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-attention-cwd": {
            "dataset_class": SignalDatasetCWD,
            "lit_model_class": BaseLitModelCWD,
            "model_class": RealConvNetAttentionCWD,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-attention-wsst": {
            "dataset_class": SignalDatasetWSST,
            "lit_model_class": BaseLitModelCWD,
            "model_class": RealConvNetAttention,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-attention-cwd-vsst": {
            "dataset_class": SignalDatasetCombined,
            "lit_model_class": BaseLitModelCWDVSST,
            "model_class": RealCWDVSST,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-attention-centerloss": {
            "dataset_class": SignalDatasetWSST,
            "lit_model_class": BaseLitModelCenterLoss,
            "model_class": RealConvNetAttentionCenterLoss,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-cbam": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_class": RealConvNetCBAM,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "realcnn-autoencoder": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModelUsingAutoencoder,
            "model_class": RealConvNetDenoise,
            "model_args": {"image_size": image_size, "number_patches": number_patches}
        },
        "real-autoencoder": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModelAutoencoder,
            "model_class": RealDenoisingAutoencoder,
            "model_args": {"image_size": image_size, "number_patches": number_patches}
        },
        "real-vit": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_class": RealViT,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "real-cvt": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_class": RealCvT,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "real-cct": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_class": RealCCT,
            "model_args": {"number_waveforms": number_waveforms}
        },
        "real-grouped": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModelGrouped,
            "model_class": RealConvNetAttentionGrouped,
            "model_args": {}
        },
        "real-grouped-classifier": {
            "dataset_class": SignalDatasetReal,
            "lit_model_class": BaseLitModel,
            "model_class": RealEnsembleClassifier,
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
            "lit_model_class": BaseLitModelUsingAutoencoder,
            "model_class": ComplexConvNetDenoise,
            "model_args": {}
        },
        "complex-autoencoder": {
            "dataset_class": SignalDatasetComplex,
            "lit_model_class": BaseLitModelAutoencoder,
            "model_class": ComplexDenoisingAutoencoder,
            "model_args": {"image_size": image_size, "number_patches": number_patches}
        }
    }

        # Validate model name
    
    if model_name not in model_config:
        raise ValueError(f"Unsupported model: {model_name}")
    # Load pretrained model if specified
    pretrained_models = load_pretrained_model(model_config, pretrained_model_name_list, checkpoint_path_list, lr, number_waveforms, signals_per_snr, freeze) 
    # Populate model arguments
    
    cfg = model_config[model_name]
    if model_name == "complexcnn-autoencoder":
        cfg["model_args"]["model"] = ComplexConvNetAttention()
        cfg["model_args"]["autoencoder"] = ComplexDenoisingAutoencoder(image_size, number_patches) if len(pretrained_models) == 0 else pretrained_models[0]

    if model_name == "realcnn-autoencoder":
        cfg["model_args"]["model"] = RealConvNetAttention()
        cfg["model_args"]["autoencoder"] = RealDenoisingAutoencoder(image_size, number_patches) if len(pretrained_models) == 0 else pretrained_models[0]

    if model_name=="real-grouped-classifier":
        if len(pretrained_models) != 2:
            raise ValueError("Two pretrained models are required for real-grouped-classifier")
        
        cfg["model_args"]["model_classifier"] = pretrained_models[0]
        cfg["model_args"]["model_group"] = pretrained_models[1]
    
    if model_name == "realcnn-attention-cwd-vsst":
        if len(pretrained_models) != 2:
            raise ValueError("Two pretrained models are required for realcnn-attention-cwd-vsst")
        
        cfg["model_args"]["model_vsst"] = pretrained_models[0]
        cfg["model_args"]["model_cwd"] = pretrained_models[1]
                 
    # Instantiate model
    model_instance = cfg["model_class"](**cfg["model_args"])
    if len(checkpoint_path_list) > 0 and len(pretrained_models) == 0:
        print("Loading model from checkpoint")
        model = cfg["lit_model_class"].load_from_checkpoint(checkpoint_path_list[0], model=model_instance, lr=lr, number_waveforms=number_waveforms, signals_per_snr = signals_per_snr)
    else:
        model = cfg["lit_model_class"](model_instance, lr, number_waveforms=number_waveforms, signals_per_snr = signals_per_snr)

    desired_labels = None
    if model_name == "real-grouped":
        desired_labels = [4, 7]
    # Create datamodule
    data_module = SignalDataModule(cfg["dataset_class"], data_paths, batch_sizes, num_workers, val_split, desired_labels=desired_labels)
    lit_model_class = cfg["lit_model_class"]
    
    return model, data_module, lit_model_class