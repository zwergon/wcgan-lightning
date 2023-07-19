import os
import torch

import lightning as L

from gan.dcgan import DCGAN
from gan.wgan import WGAN
from gan.data_module import MNISTDataModule

from lightning.pytorch import loggers 


PATH_DATASETS = "./data"
BATCH_SIZE = 256
NUM_WORKERS = 4
GAN_TYPE="wgan" # "dcgan"

if __name__ == "__main__":
    dm = MNISTDataModule(batch_size=BATCH_SIZE, data_dir=PATH_DATASETS, num_workers=NUM_WORKERS)
    if GAN_TYPE == "wgan":
        model = WGAN(*dm.dims)
    elif GAN_TYPE == "dcgan":
        model = DCGAN(*dm.dims)
    else:
        raise Exception(f"wrong GAN {GAN_TYPE}")

    tensorboard = loggers.TensorBoardLogger(save_dir="./runs")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=5,
        logger=tensorboard
    )
    trainer.fit(model, dm)
    
