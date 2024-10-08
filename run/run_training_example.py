from hyperseg.datasets.groundbased.prep import download_dataset
from hyperseg.datasets.groundbased import HyKo2
from hyperseg.datasets.callbacks import ExportSplitCallback
from hyperseg.models.imagebased import UNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torch
import torchinfo

if __name__ == '__main__':
    # --------------------------------------
    #NOTE this can be set to medium | high, if tensor cores available for possible performance gain, default: highest
    torch.set_float32_matmul_precision('highest')

    # Seed
    manual_seed=42
    seed_everything(manual_seed, workers=True)

    # Dataset Parameters
    n_classes = 11 # 11 - 1 because class 0 is undefined
    n_channels = 15
    ignore_index = 10
    precalc_histograms = "False"
    dataset_label_def = "/home/hyperseg/data/hyko2_semantic_labels.txt"
    dataset_filepath = download_dataset('~/data','HyKo2-VIS_Semantic')

    # Train/Val Parameters
    resume_path = None
    log_dir = "/mnt/data/RBMT_results/hyko2VisSem"
    train_proportion = 0.5
    val_proportion = 0.2
    batch_size = 16

    num_workers = 8
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32

    # input shape for printing model summary; use None if not required
    input_shape = None
    input_shape = (batch_size, n_channels, 512, 272) # only relevant for printed model summary
    # --------------------------------------

    data_module = HyKo2(
            filepath=dataset_filepath, 
            num_workers=num_workers,
            batch_size=batch_size,
            label_set='semantic',
            train_prop=train_proportion,
            val_prop=val_proportion,
            n_classes=n_classes,
            manual_seed=manual_seed,
            precalc_histograms=precalc_histograms)

    model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def=dataset_label_def, 
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='AdamW',
            momentum=0.0,
            weight_decay=0.0,
            ignore_index=ignore_index,
            mdmc_average='samplewise',
            bilinear=True,
            batch_norm=False,
            class_weighting=None)

    # print model summary
    torchinfo.summary(model, input_size=input_shape)

    checkpoint_callback = ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-UNet-{epoch:02d}-{val_iou_epoch:.2f}",
            save_top_k=3,
            mode='max'
            )
    export_split_callback = ExportSplitCallback()

    trainer = Trainer(
            default_root_dir=log_dir,
            callbacks=[checkpoint_callback, export_split_callback],
            accelerator='gpu',
            devices=[0], 
            max_epochs=500,
            precision=precision,
            enable_model_summary=False # enable for default model parameter printing at start
            )
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path)
