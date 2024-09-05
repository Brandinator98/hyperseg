#!/usr/bin/env python

import sys

from hsdataset import HSDataModule, HSDataset
from torchvision import transforms
from hyperseg.datasets.transforms import PermuteData
#from hyperseg.datasets.transforms import ToTensor, PermuteData
from torch.utils.data import Dataset
import os
import torch
from tqdm import tqdm
from PIL import Image
import glob
import pandas as pd

from pathlib import Path
import hyperseg

from typing import List, Any, Optional

class EuroSatRGB(HSDataModule):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

        self.transform = transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            #PermuteData(new_order=[2,0,1]),
        ])

        self.n_classes = 10
        self.undef_idx = None
        self.filepath = self.basepath

        dataset = EuroSatRGBDataset(self.filepath, mode="Val", transform=self.transform, debug=self.debug)
        img,_ = dataset[0]
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]


    def setup(self, stage: Optional[str] = None):
        self.dataset_train = EuroSatRGBDataset(  self.filepath, 
                                    transform=self.transform,
                                    debug=self.debug, mode="Train")
        self.dataset_test = EuroSatRGBDataset(   self.filepath,
                                    transform=self.transform,
                                    debug=self.debug, mode="Test")
        self.dataset_val = EuroSatRGBDataset(    self.filepath,
                                    transform=self.transform,
                                    debug=self.debug, mode="Val")
        
        # calculate data statistics for normalization
        if self.normalize:
            self.enable_normalization()
    
N_DEBUG_SAMPLES = 200

class EuroSatRGBDataset(Dataset):
    def __init__(self, folder_path, debug=False, mode='Train', transform=None):
        self.folder_path = folder_path
        self.mode = mode
        self.transform = transform
        self.debug = debug
        if self.mode == 'Train':
            self.data_frame = pd.read_csv(os.path.join(self.folder_path, 'train.csv'))
        if self.mode == 'Test':
            self.data_frame = pd.read_csv(os.path.join(self.folder_path, 'test.csv'))
        if self.mode == 'Val':
            self.data_frame = pd.read_csv(os.path.join(self.folder_path, 'validation.csv'))
        if self.debug == True:
            # only use N_DEBUG_SAMPLES samples overall
            self.data_frame = self.data_frame[:N_DEBUG_SAMPLES]
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Get the image filename and the label
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data_frame.iloc[idx, 2])  # Label is the integer class

        if self.transform:
            image = self.transform(image)

        return image, label

# name: 'loveda'
# log_name: 'loveda'
# basepath: '/mnt/data/datasets/LoveDA'
# label_set: 'semantic'
# batch_size: 4
# label_def: 'loveda_labeldef.txt'

if __name__ == "__main__":
    datamodule = EuroSatRGB(
                basepath='/mnt/data/datasets/EuroSAT',
                batch_size=4,
                label_def=Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs").joinpath('eurosat_labeldef.txt'),
                debug=True, 
                num_workers=4
            )
    datamodule.setup()
    print(datamodule.dataset_train.__len__())
    print(datamodule.dataset_train.__getitem__(0).shape) # torch.Size([3, 1024, 1024])