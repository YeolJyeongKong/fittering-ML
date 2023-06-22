import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
from torchmetrics import MeanAbsoluteError

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchsummary import summary
from efficientnet_pytorch import EfficientNet as efficientnet
from torchvision import models

from modeling.models import EfficientNet

class CNNForwardModule(pl.LightningModule):
    def __init__(self, learning_rate=2e-4):
        super().__init__()
        self.learning_rate = learning_rate

        self.save_hyperparameters()
        
        self.model = EfficientNet()

        self.mae = torchmetrics.MeanAbsoluteError()
    
    def forward(self, image, height):
        return self.model(image, height)
    
    def training_step(self, batch, batch_idx):
        front, side, height, betas =\
              batch['front_image'], batch['side_image'], batch['height'], batch['betas']
        image = torch.cat((front, side), dim=1)
        logits = self(image, height)
        loss = F.mse_loss(logits, betas)

        mae = self.mae(logits, betas)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        front, side, height, betas =\
              batch['front_image'], batch['side_image'], batch['height'], batch['betas']
        image = torch.cat((front, side), dim=1)
        logits = self(image, height)
        loss = F.mse_loss(logits, betas)

        mae = self.mae(logits, betas)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        front, side, height, betas =\
              batch['front_image'], batch['side_image'], batch['height'], batch['betas']
        image = torch.cat((front, side), dim=1)
        logits = self(image, height)
        loss = F.mse_loss(logits, betas)

        mae = self.mae(logits, betas)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mae', mae, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

if __name__ == "__main__":
    target = torch.tensor([[1,2], [3,4]])
    preds = torch.tensor([[1, 3], [3, 5]])
    mean_absolute_error = MeanAbsoluteError()
    print(mean_absolute_error(preds, target))