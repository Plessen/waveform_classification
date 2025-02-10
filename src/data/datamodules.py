from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import lightning as L

class SignalDataModule(L.LightningDataModule):
    def __init__(self, dataset_class, data_paths, batch_sizes, num_workers, val_split = 0.2, random_state = 42):
        super().__init__()
        self.dataset = dataset_class
        self.train_data_path = data_paths['train']
        self.test_data_path = data_paths['test']
        self.train_batch_size = batch_sizes['train']
        self.val_batch_size = batch_sizes['val']
        self.test_batch_size = batch_sizes['test']
        self.num_workers = num_workers
        self.val_split = val_split
        self.random_state = random_state
    
    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            train_dataset = self.dataset(self.train_data_path)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=self.random_state)
            indices = np.arange(len(train_dataset))
            train_idx, val_idx = next(sss.split(indices, train_dataset.class_indices))
            
            self.train_subset = Subset(train_dataset, train_idx)
            self.val_subset = Subset(train_dataset, val_idx)
            
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset(self.test_data_path)
    
    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)