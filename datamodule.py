import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

import wandb

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, ToPILImage, Resize
from torchvision.transforms import functional as F

from data.dataset import BinaryImageMeasDataset, AihubDataset
from data.preprocessing import *

import config


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset_mode="synthetic"):
        super().__init__()
        self.batch_size = batch_size

        assert dataset_mode in ("synthetic", "aihub")
        self.dataset_mode = dataset_mode

        self.transform = transforms.Compose(
            [
                ToTensor(),
                BinTensor(threshold=0.5),
                Lambda(crop_true),
                Resize((512, 512), interpolation=F.InterpolationMode.NEAREST),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.dataset_mode == "synthetic":
                full_dataset = BinaryImageMeasDataset(
                    data_dir=config.GEN_TRAIN_DIR, transform=self.transform
                )
            elif self.dataset_mode == "aihub":
                full_dataset = AihubDataset(
                    data_dir=os.path.join(config.AIHUB_DATA_DIR, "train"),
                    transform=self.transform,
                )
            full_length = len(full_dataset)
            train_len = int(full_length * 0.9)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, full_length - train_len]
            )

        if stage == "test" or stage is None:
            if self.dataset_mode == "synthetic":
                self.test_dataset = BinaryImageMeasDataset(
                    data_dir=config.GEN_TEST_DIR, transform=self.transform
                )
            elif self.dataset_mode == "aihub":
                self.test_dataset = AihubDataset(
                    data_dir=os.path.join(config.AIHUB_DATA_DIR, "test"),
                    transform=self.transform,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
