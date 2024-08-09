#!/usr/bin/env python

from .hsdataset import HSDataModule, HSDataset
from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage
from hyperseg.datasets.utils import apply_pca

from typing import List, Any, Optional

class LoveDA(HSDataModule):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

        self.transform = transforms.Compose([
            ToTensor(),
            PermuteData(new_order=[2,0,1]),
        ])

        self.n_classes = 7
        self.undef_idx=0
        self.filepath_train = self.basepath.joinpath('hyko2_train.h5')
        self.filepath_test = self.basepath.joinpath('hyko2_test.h5')
        self.filepath_val = self.basepath.joinpath('hyko2_val.h5')

        dataset = HSDataset(self.filepath_val, transform=self.transform)
        img,_ = dataset[0]
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]


    def setup(self, stage: Optional[str] = None):
        self.dataset_train = HSDataset(  self.filepath_train, 
                                    transform=self.transform,
                                    debug=self.debug)
        self.dataset_test = HSDataset(   self.filepath_test,
                                    transform=self.transform,
                                    debug=self.debug)
        self.dataset_val = HSDataset(    self.filepath_val,
                                    transform=self.transform,
                                    debug=self.debug)
        
        # calculate data statistics for normalization
        if self.normalize:
            self.enable_normalization()
    