from nn_modules.realcnn import RealConvNet
from nn_modules.complexcnn import ComplexConvNetV3
from lit_modules import BaseLitModel
from ..data.datamodules import SignalDataModule
from ..data.datasets import SignalDatasetComplex, SignalDatasetReal

def model_factory(model_name, data_paths, batch_sizes, num_workers, val_split):
    if model_name == "realcnn":
        dataset_class = SignalDatasetReal
        model = BaseLitModel(RealConvNet())
    elif model_name == "complexcnn":
        dataset_class = SignalDatasetComplex
        model = BaseLitModel(ComplexConvNetV3())
    else:
        raise ValueError("Model name not recognized.")
    
    data_module = SignalDataModule(dataset_class, data_paths, batch_sizes, num_workers, val_split)
    
    return model, data_module