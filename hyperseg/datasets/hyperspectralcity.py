#!/usr/bin/env python

from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage, InsertEmptyChannelDim
from .hsdataset import HSDataModule, HSDataset

class HyperspectralCityV2(HSDataModule):
    def __init__(self, half_precision=False, n_pc=None, **kwargs):
        super().__init__(**kwargs)
        self.half_precision = half_precision
        self.n_pc = n_pc

        self.transform = transforms.Compose([
            ToTensor(half_precision=self.half_precision),
            PermuteData(new_order=[2,0,1]),
            ReplaceLabels({255:19})
        ])

        if self.spectral_average:
            self.transform = transforms.Compose([
                self.transform,
                SpectralAverage()
            ])

        self.n_classes = 20
        self.undef_idx=19

        dataset = HSDataset(self.filepath, transform=self.transform)
        img,_ = dataset[0]
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]

