#!/usr/bin/env python

from .hsdataset import HSDataModule, HSDataset
from torchvision import transforms
#from hyperseg.datasets.transforms import ToTensor, PermuteData
from torch.utils.data import Dataset
import os
import torch
from tqdm import tqdm
from PIL import Image
import glob

from pathlib import Path
import hyperseg

from typing import List, Any, Optional

class LoveDA(HSDataModule):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            #PermuteData(new_order=[2,0,1]),
        ])

        self.n_classes = 7
        self.undef_idx=0
        self.filepath = self.basepath

        dataset = LoveDADataset(self.filepath, mode="Val", transform=self.transform, debug=self.debug)
        img,_ = dataset[0]
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]


    def setup(self, stage: Optional[str] = None):
        self.dataset_train = LoveDADataset(  self.filepath, 
                                    transform=self.transform,
                                    debug=self.debug, mode="Train")
        self.dataset_test = LoveDADataset(   self.filepath,
                                    transform=self.transform,
                                    debug=self.debug, mode="Test")
        self.dataset_val = LoveDADataset(    self.filepath,
                                    transform=self.transform,
                                    debug=self.debug, mode="Val")
        
        # calculate data statistics for normalization
        if self.normalize:
            self.enable_normalization()
    
N_DEBUG_SAMPLES = 200

class LoveDADataset(Dataset):
    def __init__(self, folder_path, debug=False, mode='Train', transform=None):
        self.folder_path = folder_path
        self.mode = mode
        self.transform = transform
        self.debug = debug
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, mode, '*', 'images_png', '*.png')))
        self.label_paths = sorted(glob.glob(os.path.join(folder_path, mode, '*', 'masks_png', '*.png')))
        if self.debug == True:
            # only use N_DEBUG_SAMPLES samples overall
            self.image_paths = self.image_paths[:N_DEBUG_SAMPLES]
            self.label_paths = self.label_paths[:N_DEBUG_SAMPLES]
        self.max_label_value = self.get_max_value_over_images()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(img_path)
        label = Image.open(label_path).convert("L")
        if self.transform:
            image = self.transform(image)
            label = self.transform(label) 

        #valid/performant way of normalizing? I guess not.. 
        label = label / self.max_label_value
        label = label * 6.0
        label = label.to(torch.int64)
        return image, label


    def get_max_value_over_images(self):
        overall_max = -float('inf')  # Initialize with a very low value
        for label_path in tqdm(self.label_paths, desc="Calculate normalization value"):
            label = Image.open(label_path).convert("L")
            #print(label, self.transform)
            #exit()
            if self.transform:
                label = self.transform(label)
            max_val = torch.max(label)
            if max_val > overall_max:
                overall_max = max_val
        return overall_max
    


# name: 'loveda'
# log_name: 'loveda'
# basepath: '/mnt/data/datasets/LoveDA'
# label_set: 'semantic'
# batch_size: 4
# label_def: 'loveda_labeldef.txt'

if __name__ == "__main__":
    datamodule = LoveDA(
                basepath='/mnt/data/datasets/LoveDA',
                batch_size=4,
                label_def=Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs").joinpath('loveda_labeldef.txt'),
                debug=True, 
                num_workers=4
            )
    datamodule.setup()
    print(datamodule.dataset_train.__len__())
    print(datamodule.dataset_train.__getitem__(0)[0].shape) # torch.Size([3, 1024, 1024])