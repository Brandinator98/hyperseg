import torch
import numpy as np
import copy
from semsegmodule import SemanticSegmentationModule


class DualEncoderResNet(SemanticSegmentationModule):
    def __init__(self, out_channels=25):
        super(DualEncoderResNet, self).__init__()
        #load model 
        #  https://pytorch.org/hub/pytorch_vision_fcn_resnet101/ 

        # model weights -- https://pytorch.org/vision/stable/models.html#semantic-segmentation (available weights)
        # weights=None - no pretrained weights
        # weights="DEFAULT" - pretrained weights - coco with voc labels v1

        # backbone weights -- are pretrained weights for the standard resnet50, that is used as a backbone for the model. All ResNet50_Weights are possible https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights (available weights)
        # weights_backbone=None - no pretrained weights 
        # weights="DEFAULT" - pretrained weights - ImageNet v1
        original_model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', weights='DEFAULT')
        self.encoder = torch.nn.Sequential(*list(original_model.children())[:-2])
        self.classifier = torch.nn.Sequential(original_model.classifier)
        self.aux_classifier = torch.nn.Sequential(original_model.aux_classifier)
        #self.upsample = torch.nn.Upsample(size=600)
        self.upsample = torch.nn.ConvTranspose2d(in_channels=21, out_channels=21, kernel_size=16, padding=4, stride=8, output_padding=0)
        self.conv1x1 = torch.nn.Conv2d(in_channels=21, out_channels=out_channels, kernel_size=1) #indian pines has 16 classes
        self.hsi_initialized = False
        
    def forward(self, image):
        #print(f"Image shape: {image.shape}, {image.shape[2]}") # print in-between shapes
        if (self.hsi_initialized):
            rgb_image = image[:, [19, 8, 0], :, :]  # Extract pseudo-RGB image like Ding et al. - bands 63, 19, 1
            y = self.hsi_encoder(image)
            x = self.encoder(rgb_image)
            #print(f"X shape: {np.shape(x['out'])}, Y shape: {np.shape(y['out'])}") # print in-between shapes
            # Insert feature addition
            z = self.add_features(x['out'], y['out'])
        else: 
            x = self.encoder(image)
            z = x['out']
        z = self.classifier(z)
        z = self.upsample(z)
        z = self.conv1x1(z)
        return z

    def add_features(self, x, y):
        return torch.add(x,y)
    
    def define_hsi_encoder(self, hsi_image):
        try:
            self.hsi_encoder = copy.deepcopy(self.encoder)
            self.hsi_encoder[0].conv1 = torch.nn.Conv2d(
                in_channels=hsi_image.shape[0], out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
            ) 
            self.hsi_initialized = True
            self.upsample = torch.nn.Upsample(size=hsi_image.shape[2])
            return
        except Exception as e:
            print(e)
            return



#if __name__ == "__main__":
    # load pre-trained weights into model 
    # model = DualEncoderResNet()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # weights = torch.load("./models/LoveDA_9.pth")
    # pretrained_dict = {k: v for k, v in weights.items() if k.startswith("encoder")}
    # model.load_state_dict(pretrained_dict, strict=False)