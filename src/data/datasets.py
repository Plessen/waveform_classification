import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class SignalDatasetComplex(Dataset):    
           
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None

        with h5py.File(self.file_path, 'r') as file:
            self.total_size = file['/clean_images/images_real'].shape[0]
            self.class_indices = np.argmax(np.array(file['/labels']), axis=1)
            
    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        #with h5py.File(self.file_path, 'r') as file:
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
            
        clean_real = torch.tensor(self.file['/clean_images/images_real'][idx], dtype=torch.float32)
        clean_imag = torch.tensor(self.file['/clean_images/images_imag'][idx], dtype=torch.float32)
        clean_image = torch.complex(clean_real, clean_imag).unsqueeze(0)
        
        noisy_real = torch.tensor(self.file['/noisy_images/images_real'][idx], dtype=torch.float32)
        noisy_imag = torch.tensor(self.file['/noisy_images/images_imag'][idx], dtype=torch.float32)
        noisy_image = torch.complex(noisy_real, noisy_imag).unsqueeze(0)
        
        label = torch.argmax(torch.tensor(self.file['/labels'][idx], dtype=torch.float32), dim=0)
            
        return clean_image, noisy_image, label

    def  __del__(self):
        if self.file:
            self.file.close()
    
class SignalDatasetReal(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None  
        
        with h5py.File(self.file_path, 'r') as file:
            self.total_size = file['/clean_images/images_real'].shape[0]
            self.class_indices = np.argmax(np.array(file['/labels']), axis=1)
            
    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):        
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
            
        clean_real = torch.tensor(self.file['/clean_images/images_real'][idx], dtype=torch.float32).unsqueeze(0)
        clean_imag = torch.tensor(self.file['/clean_images/images_imag'][idx], dtype=torch.float32).unsqueeze(0)
        clean_image = torch.cat((clean_real, clean_imag), dim=0)
        
        noisy_real = torch.tensor(self.file['/noisy_images/images_real'][idx], dtype=torch.float32).unsqueeze(0)
        noisy_imag = torch.tensor(self.file['/noisy_images/images_imag'][idx], dtype=torch.float32).unsqueeze(0)
        noisy_image = torch.cat((noisy_real, noisy_imag), dim=0)
        
        label = torch.argmax(torch.tensor(self.file['/labels'][idx], dtype=torch.float32), dim=0)
            
        return clean_image, noisy_image, label

    def __del__(self):
        if self.file:
            self.file.close()