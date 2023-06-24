import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
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

from modeling.models import EfficientNet, AutoEncoder
from utils.metrics import AccuracyBinaryImage

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
    

class AutoEncoderModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.autoencoder = AutoEncoder()

        self.avg_acc_train = AccuracyBinaryImage()
        self.avg_acc_val = AccuracyBinaryImage()
        self.avg_acc_test = AccuracyBinaryImage()

    def forward(self, x):
        return self.autoencoder.encoder(x)
    
    def training_step(self, batch, batch_idx):
        x = torch.cat((batch['front_image'], batch['side_image']), dim=1)
        y = self.autoencoder(x)
        loss = F.binary_cross_entropy(y, x)

        avg_acc = self.avg_acc_train.metric(y, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_avg_acc', avg_acc, on_step=True, on_epoch=True, logger=True)

        return loss
    
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.avg_acc_train.compute_score()
    
    def validation_step(self, batch, batch_idx):
        x = torch.cat((batch['front_image'], batch['side_image']), dim=1)
        y = self.autoencoder(x)
        loss = F.binary_cross_entropy(y, x)

        avg_acc = self.avg_acc_val.metric(y, x)
        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_avg_acc', avg_acc, prog_bar=True)

        return loss
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        val_avg_acc = self.avg_acc_val.compute_score()
        self.log('val_avg_acc', val_avg_acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        x = torch.cat((batch['front_image'], batch['side_image']), dim=1)
        y = self.autoencoder(x)
        loss = F.binary_cross_entropy(y, x)

        avg_acc = self.avg_acc(y, x)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_avg_acc', avg_acc, prog_bar=True)

        return loss
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.avg_acc_test.compute_score()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    target = torch.zeros((32, 2, 512, 512))
    preds = torch.ones((32, 2, 512, 512))
    # acc = torchmetrics.Accuracy()
    # print(acc(preds, target))