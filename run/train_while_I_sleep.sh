#python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=300 dataset=hsidrivepseudorgb model=hs3-baseline
#python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=300 dataset=hsidrivepca1 model=hs3-baseline dataset.drop_last=True
#python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hrpseudorgb model=hs3-baseline model.optimizer_eps=1e-04 dataset.drop_last=True
#python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hrpca1 model=deeplabv3plus-mobilenet model.optimizer_eps=1e-04 dataset.drop_last=True
#python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hrpseudorgb model=deeplabv3plus-mobilenet model.optimizer_eps=1e-04 dataset.drop_last=True
##python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hr model=deeplabv3plus-mobilenet model.optimizer_eps=1e-04 dataset.drop_last=True
python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hrpseudorgb model=deeplabv3plus-mobilenet model.optimizer_eps=1e-04 dataset.drop_last=True model.pretrained_backbone=True model.pretrained_weights=/home/hyperseg/git/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hrpseudorgb model=deeplabv3plus-mobilenet model.optimizer_eps=1e-04 dataset.drop_last=True model.pretrained_backbone=True 
python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hrrgb model=deeplabv3plus-mobilenet model.optimizer_eps=1e-04 dataset.drop_last=True
python train.py logging.project_name=hs3bench training.early_stopping=True training.max_epochs=100 dataset=hcv2_hrrgb model=deeplabv3plus-mobilenet model.optimizer_eps=1e-04 dataset.drop_last=True model.pretrained_backbone=True
