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

from src.data.dataset import BinaryImageMeasDataset, AihubDataset
from src.data.preprocess import *

from extras import paths


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        transform,
        train_dir="/home/shin/VScodeProjects/fittering-ML/data/aihub/train",
        test_dir="/home/shin/VScodeProjects/fittering-ML/data/aihub/test",
        dataset_mode="aihub",
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        train_ratio=0.9,
        shuffle=True,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir

        assert dataset_mode in ("synthetic", "aihub")
        self.dataset_mode = dataset_mode

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ratio = train_ratio
        self.shuffle = shuffle

        self.transform = transform

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.dataset_mode == "synthetic":
                full_dataset = BinaryImageMeasDataset(
                    data_dir=self.train_dir, transform=self.transform
                )
            elif self.dataset_mode == "aihub":
                full_dataset = AihubDataset(
                    data_dir=self.train_dir,
                    transform=self.transform,
                )
            full_length = len(full_dataset)
            train_len = int(full_length * self.train_ratio)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_len, full_length - train_len]
            )

        if stage == "test" or stage is None:
            if self.dataset_mode == "synthetic":
                self.test_dataset = BinaryImageMeasDataset(
                    data_dir=self.test_dir, transform=self.transform
                )
            elif self.dataset_mode == "aihub":
                self.test_dataset = AihubDataset(
                    data_dir=self.test_dir,
                    transform=self.transform,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
