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
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from modeling.lightning_modules import CNNForwardModule
from Data.datamodule import DataModule


dm = DataModule(batch_size=8)
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