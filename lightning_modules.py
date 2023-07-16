import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torchmetrics
from torchmetrics import MeanAbsoluteError, Accuracy

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.models import EfficientNet, AutoEncoder, EfficientNetv2, SimpleNet
from utils.metrics import AccuracyBinaryImage, MeasureMAE

class CNNForwardModule(pl.LightningModule):
    def __init__(self, learning_rate=2e-4):
        super().__init__()
        self.learning_rate = learning_rate

        self.save_hyperparameters()
        
        self.model = EfficientNetv2()

        self.mae = torchmetrics.MeanAbsoluteError()
    
    def forward(self, front, side, height):
        return self.model(front, side, height)
    
    def training_step(self, batch, batch_idx):
        front, side, height, meas =\
              batch['front'], batch['side'], batch['height'], batch['meas']
        logits = self(front, side, height)
        loss = F.mse_loss(logits, meas)

        mae = self.mae(logits, meas)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        front, side, height, meas =\
              batch['front'], batch['side'], batch['height'], batch['meas']
        logits = self(front, side, height)
        loss = F.mse_loss(logits, meas)

        mae = self.mae(logits, meas)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        front, side, height, meas =\
              batch['front'], batch['side'], batch['height'], batch['meas']
        logits = self(front, side, height)
        loss = F.mse_loss(logits, meas)

        mae = self.mae(logits, meas)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_mae', mae, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

class AutoEncoderModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.autoencoder = AutoEncoder()

        self.acc = Accuracy(task='binary')

    def forward(self, x):
        return self.autoencoder.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['side']
        y = self.autoencoder(x)
        loss = F.binary_cross_entropy(y, x)

        logits = (y > 0.5).float()
        acc = self.acc(logits, x)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['side']
        y = self.autoencoder(x)
        loss = F.binary_cross_entropy(y, x)

        logits = (y > 0.5).float()
        acc = self.acc(logits, x)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['side']
        y = self.autoencoder(x)
        loss = F.binary_cross_entropy(y, x)

        logits = (y > 0.5).float()
        acc = self.acc(logits, x)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    target = torch.zeros((32, 2, 512, 512))
    preds = torch.ones((32, 2, 512, 512))
    # acc = torchmetrics.Accuracy()
    # print(acc(preds, target))