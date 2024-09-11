#!/usr/bin/env python

import sys
if '/home/abrand/hyperseg/src/hyperspectral-semantic-segmentation' in sys.path:
    sys.path.remove('/home/abrand/hyperseg/src/hyperspectral-semantic-segmentation')
sys.path.append('/home/abrand/hyperseg')
from pathlib import Path
#from osgeo import gdal
import tifffile
import numpy as np
from torchvision import transforms
from typing import List, Any, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from hyperseg.datasets.analysis.tools import StatCalculator
from hyperseg.datasets.transforms import ToTensor, PermuteData, Normalize, ReplaceLabels, SpectralAverage, InsertEmptyChannelDim

from hyperseg.datasets.utils import apply_pca
from hsdataset import HSDataModule, HSDataset
import matplotlib.pyplot as plt
import numpy as np
import hyperseg

def get_label_path(path):
        return Path(str(path).replace('image', 'label'))

class WHUOHS(HSDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.save_hyperparameters()

        self.n_classes = 25
        self.undef_idx = 0
        self.filepath_train = self.basepath.joinpath('whuohs_train.h5')
        self.filepath_test = self.basepath.joinpath('whuohs_test.h5')
        self.filepath_val = self.basepath.joinpath('whuohs_val.h5')

        self.transform = transforms.Compose([
                            ToTensor(),
                            PermuteData(new_order=[2,0,1]),
                        ])

        if self.spectral_average:
            self.transform = transforms.Compose([
                                self.transform,
                                SpectralAverage()
                             ])
        if self.prep_3dconv:
            self.transform = transforms.Compose([
                self.transform,
                InsertEmptyChannelDim(1)
            ])

        if self.pca is not None:
            self.enable_pca()

        # read dimensions from image
        dataset = HSDataset(filepath=self.filepath_val, transform=self.transform)
        img, _ = dataset[0]
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

    def enable_pca(self):
        # train
        outpath_train = self.pca_out_dir.joinpath(f'whuohs_train_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_train, outpath_train, 
                    debug=self.debug, half_precision=False)
        self.filepath_train = outpath_train

        # test
        outpath_test = self.pca_out_dir.joinpath(f'whuohs_test_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_test, outpath_test, 
                    debug=self.debug, half_precision=False)
        self.filepath_test = outpath_test

        # val 
        outpath_val = self.pca_out_dir.joinpath(f'whuohs_val_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_val, outpath_val, 
                    debug=self.debug, half_precision=False)
        self.filepath_val = outpath_val

class WHUOHSDataset(Dataset):
    def __init__(self, 
                basepath,
                transform,
                mode='train',
                debug=False):
        self.basepath = Path(basepath).expanduser()
        self._transform = transform
        self.mode = mode
        self.debug = debug

        if mode not in ('train','test','val','full'):
            raise RuntimeError("Invalid mode! It must be `train`,`test`, `val` or `full`.")
        
        if mode in ('train', 'test', 'val'):
            imagedir = self.basepath.joinpath(self.mode,'image')
            self._samplelist = [ p for p in imagedir.iterdir() 
                                if(p.suffix == '.tif')]
            if self.debug:
                if mode == 'train':
                    self._samplelist = self._samplelist[:90]
                elif mode =='val':
                    self._samplelist = self._samplelist[:10]
                elif mode =='test':
                    self._samplelist = self._samplelist[:10]
            
        else:
            self._samplelist = []
            for mode in ('train', 'test', 'val'):
                imagedir = self.basepath.joinpath(mode, 'image')
                self._samplelist.extend([ p for p in imagedir.iterdir()
                                    if(p.suffix == '.tif')])
                if self.debug:
                    self._samplelist = self.samplelist[:100]
            

    def enable_normalization(self, means, stds):
        self._transform = transforms.Compose([
            self._transform,
            Normalize(means=means, stds=stds)
        ])
        self.mean = means
        self.std = stds

    def __getitem__(self, i):
        image_path = self._samplelist[i]
        label_path = get_label_path(self._samplelist[i])

        '''
        image = gdal.Open(image_path, gdal.GA_ReadOnly)
        label = gdal.Open(label_path, gdal.GA_ReadOnly)
        image = image.ReadAsArray().astype(np.float32) / 10000.0
        label = label.ReadAsArray().astype(np.longlong)

        Above you see how the data is loaded in the authors repo:
        https://github.com/zjjerica/WHU-OHS-Pytorch/tree/main
        However, setting up GDAL with python is a pain in the ass and has dependencies that
        cannot be installed with pip. Therefore we use tifffile-package instead.
        '''
        image = tifffile.imread(image_path).astype(np.float32) / 10000.0
        label = tifffile.imread(label_path).astype(np.longlong)

        sample = (image, label)

        if self._transform:
            sample = self._transform(sample)
        return sample

    def samplelist(self):
        return self._samplelist

    def __len__(self):
        return len(self._samplelist)

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# Step 1: Load the label-color definitions from a file
def load_label_colors(file_path):
    label_colors = {}
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            label = int(row[0])
            r, g, b = int(row[2]), int(row[3]), int(row[4])
            label_colors[label] = (r, g, b)
    return label_colors

# Step 2: Map the labels to colors
def create_color_label_image(label_image, label_colors):
    # Create an empty array for the RGB image with the same height and width as the label image
    color_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.uint8)

    # Map each label in the label image to its corresponding RGB color
    for label, color in label_colors.items():
        color_image[label_image == label] = color  # Replace each label with its RGB value
    
    return color_image

if __name__ == '__main__':
    datamodule = WHUOHS(
                basepath='/mnt/data/datasets/whuohs',
                batch_size=4,
                label_def=Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs").joinpath('whuohs_labeldef.txt'),
                debug=True, 
                num_workers=4
            )
    datamodule.setup()
    print(len(datamodule.val_dataloader()))
    image = datamodule.dataset_val.__getitem__(0)[0]
    label = datamodule.dataset_val.__getitem__(0)[1]
    print(image.size)
    rgb_image = image[[19, 8, 0], :, :]
    
    # Transponiere das Bild in [Höhe, Breite, Kanäle]
    rgb_image = np.transpose(rgb_image, (1, 2, 0))

    # Normalisiere oder skaliere das Bild, wenn nötig (optional)
    # Zeige das Bild an
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    plt.axis('off')  # Optional: Verstecke die Achsen
    plt.savefig('/mnt/data/images/rgb_image.png')

    rgb_image_norm = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
    plt.imshow(rgb_image_norm)
    plt.title('RGB Image Normalized')
    plt.axis('off')  # Optional: Verstecke die Achsen
    plt.savefig('/mnt/data/images/rgb_image_norm.png')

    label_color_file = Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs").joinpath('whuohs_labeldef.txt')  # Path to your label-color file
    label_colors = load_label_colors(label_color_file)
    color_label_image = create_color_label_image(label, label_colors)

    plt.imshow(color_label_image)
    plt.title('Label Image')
    plt.axis('off')  # Optional: Verstecke die Achsen
    plt.savefig('/mnt/data/images/label_image.png')
