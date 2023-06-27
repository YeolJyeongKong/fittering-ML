import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader

from lightning_modules import CNNForwardModule, AutoEncoderModule
from datamodule import DataModule
from utils.callbacks import ImagePredictionLogger


def train_CNNForwardModule():
    dm = DataModule(batch_size=8, train_data_range=(0, 10000), test_data_range=(0, 1000))
    dm.prepare_data()
    dm.setup()

    module = CNNForwardModule()
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

    trainer = pl.Trainer(max_epochs=50, 
                        gpus=1,
                        logger=wandb_logger,
                        callbacks=[EarlyStopping(monitor='val_loss'), 
                                    ModelCheckpoint()])

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)
    wandb.finish()


def train_AutoEncoderModule():
    dm = DataModule(batch_size=8, train_data_range=(0, 10000), test_data_range=(0, 1000))
    dm.prepare_data()
    dm.setup()

    module = AutoEncoderModule()
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

    val_samples = next(iter(dm.val_dataloader()))

    trainer = pl.Trainer(max_epochs=50, 
                        gpus=1,
                        logger=wandb_logger,
                        callbacks=[EarlyStopping(monitor='val_loss'), 
                                    ModelCheckpoint(),
                                    ImagePredictionLogger(val_samples=val_samples)])

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)
    wandb.finish()


if __name__ == "__main__":
    train_AutoEncoderModule()
    # train_CNNForwardModule()
