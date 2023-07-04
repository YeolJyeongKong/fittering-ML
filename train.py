import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import warnings
warnings.filterwarnings(action='ignore')

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader

from lightning_modules import CNNForwardModule, AutoEncoderModule
from datamodule import DataModule
from utils.callbacks import ImagePredictionLogger, BetaPredictionLogger, MeasurementsLogger


def train_CNNForwardModule():
    batch_size = 16
    epochs = 50

    dm = DataModule(batch_size=batch_size, train_data_range=(0, 10000), test_data_range=(0, 1000))
    dm.prepare_data()
    dm.setup()

    module = CNNForwardModule(device=torch.device("cuda"), learning_rate=2e-4)
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')
    val_samples = next(iter(dm.val_dataloader()))

    trainer = pl.Trainer(max_epochs=epochs, 
                        gpus=1,
                        logger=wandb_logger,
                        callbacks=[EarlyStopping(monitor='val_loss'), 
                                    ModelCheckpoint(),
                                    # BetaPredictionLogger(val_samples, 
                                    #                      batch_size=batch_size, 
                                    #                      device=module.device), 
                                    MeasurementsLogger(val_samples, 
                                                       batch_size=batch_size, 
                                                       device=module.device)])

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)
    wandb.finish()


def train_AutoEncoderModule():
    batch_size = 8
    epochs = 50

    dm = DataModule(batch_size=batch_size, train_data_range=(0, 10000), test_data_range=(0, 1000))
    dm.prepare_data()
    dm.setup()

    module = AutoEncoderModule()
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

    val_samples = next(iter(dm.val_dataloader()))

    trainer = pl.Trainer(max_epochs=epochs, 
                        gpus=1,
                        logger=wandb_logger,
                        callbacks=[EarlyStopping(monitor='val_loss'), 
                                    ModelCheckpoint(),
                                    ImagePredictionLogger(val_samples=val_samples)])

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)
    wandb.finish()


if __name__ == "__main__":
    # train_AutoEncoderModule()
    train_CNNForwardModule()
