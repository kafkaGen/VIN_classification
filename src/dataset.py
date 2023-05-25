import sys

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T

sys.path.append('../')
from settings.config import Config


class VINDataset(torch.utils.data.Dataset):
    """Dataset class for reading and preprocessing the data.

    Args:
        subset (str): Subset of data (train or test).
        transforms (callable, optional): Optional transformations to be applied to the data.

    Attributes:
        df (pandas.DataFrame): Dataframe of the dataset.
        mapping (pandas.DataFrame): Dataframe of ASCII mapping.

    """
    
    def __init__(self, subset, transforms=None):
        self.subset = subset
        self.transforms = transforms
        
        if subset == 'train':
            self.df = pd.read_csv(Config.train_data_path, header=None)
        else:
            self.df = pd.read_csv(Config.test_data_path, header=None)
        self.mapping = pd.read_csv(Config.mapping_path, header=None, delimiter=' ', 
                                   index_col=0, names=['ASCII'])
        self.mapping.index.name = 'label'
        self.mapping['char'] = self.mapping['ASCII'].apply(chr)
        # leave only digits and upper case letter labels except I, Q and O
        self.mapping = self.mapping[self.mapping['char'].str.contains(r'^[0-9A-HJ-NP-PR-Z]+$')]
        self.mapping['old_index'] = self.mapping.index
        self.mapping.index = range(self.mapping.shape[0])
        self.df = self.df[self.df[0].isin(self.mapping['old_index'])]
        
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.df.shape[0]
        
    def __getitem__(self, idx):
        """Returns the data and label of the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple of the image and its corresponding label.

        """
        img = self.df.iloc[idx, 1:].to_numpy().reshape(Config.resize_to)[np.newaxis,: ,:]
        img = torch.from_numpy(img).to(torch.float32)
        label = self.df.iloc[idx, 0]
        label = np.argwhere(self.mapping['old_index'] == label)[0][0]
        label = torch.tensor(label)
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.Normalize(Config.mean, Config.std)(img)
        
        return img, label