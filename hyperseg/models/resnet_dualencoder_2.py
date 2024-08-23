import torch
import numpy as np
import copy
from .semsegmodule import SemanticSegmentationModule
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models import ResNet50_Weights

from pathlib import Path
import hyperseg

class DualEncoderResNet(SemanticSegmentationModule):
    def __init__(self,
            **kwargs):
        super(DualEncoderResNet, self).__init__(**kwargs)
        #load model 
        #  https://pytorch.org/hub/pytorch_vision_fcn_resnet101/ 

        # model weights -- https://pytorch.org/vision/stable/models.html#semantic-segmentation (available weights)
        # weights =FCN_ResNet50_Weights.DEFAULT
        # weights =FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1

        # backbone weights -- are pretrained weights for the standard resnet50, that is used as a backbone for the model. All ResNet50_Weights are possible https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights (available weights)
        # weights_backbone=None - no pretrained weights 
        # weights_backbone="DEFAULT" - pretrained weights - ImageNet v1

        self.save_hyperparameters()

        original_model = fcn_resnet50(weights=None,weights_backbone=None)
        self.encoder = original_model.backbone
        self.classifier = original_model.classifier
        self.hsi_encoder = copy.deepcopy(self.encoder)
        self.hsi_encoder.conv1 = torch.nn.Conv2d(
            in_channels=self.n_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        ) 
        self.classifier[4] = torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=16, padding=4, stride=8, output_padding=0)
        self.conv1x1 = torch.nn.Conv2d(in_channels=512, out_channels=self.n_classes, kernel_size=1)
        
    def forward(self, image):
        #print(f"Image shape: {image.shape}, {image.shape[2]}") # print in-between shapes
        rgb_image = image
        if self.n_channels > 20:
            rgb_image = image[:, [15, 4, 0], :, :]  # Extract pseudo-RGB image like Ding et al. - for the 32 bands of WHUOHS, ranging from 420-980nm, which would bei R-> 16, G-> 5 and B-> 1 (minus index), see paper
        y = self.hsi_encoder(image)
        x = self.encoder(rgb_image)
        #print(f"X shape: {np.shape(x['out'])}, Y shape: {np.shape(y['out'])}") # print in-between shapes
        # Insert feature addition
        z = self.add_features(x['out'], y['out'])
        z = self.classifier(z)
        z = self.conv1x1(z)
        return z

    def add_features(self, x, y):
        return torch.add(x,y)
    
   
#if __name__ == "__main__":
    # load pre-trained weights into model 
    # model = DualEncoderResNet(n_classes=25, n_channels=220, learning_rate=0.001, label_def=Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs").joinpath('whuohs_labeldef.txt'), loss_name="cross_entropy", optimizer_name="AdamW", momentum=0, weight_decay=0, ignore_index=0)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # weights = torch.load("./models/LoveDA_9.pth")
    # pretrained_dict = {k: v for k, v in weights.items() if k.startswith("encoder")}
    # model.load_state_dict(pretrained_dict, strict=False)