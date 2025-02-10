from .nn_modules.realcnn import RealConvNet
from .nn_modules.complexcnn import ComplexConvNet, ComplexConvNetAttention
from .lit_modules import BaseLitModel
from ..data.datamodules import SignalDataModule
from ..data.datasets import SignalDatasetComplex, SignalDatasetReal

def model_factory(model_name, data_paths, batch_sizes, num_workers, val_split):
    if model_name == "realcnn":
        dataset_class = SignalDatasetReal
        model = BaseLitModel(RealConvNet())
    elif model_name == "complexcnn":
        dataset_class = SignalDatasetComplex
        model = BaseLitModel(ComplexConvNet())
    elif model_name == "complexcnn-attention":
        dataset_class = SignalDatasetComplex
        model = BaseLitModel(ComplexConvNetAttention())
    else:
        raise ValueError("Model name not recognized.")
    
    data_module = SignalDataModule(dataset_class, data_paths, batch_sizes, num_workers, val_split)
    
    return model, data_module