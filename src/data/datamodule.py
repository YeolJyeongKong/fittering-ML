import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torchmetrics

import wandb
from sklearn.model_selection import train_test_split
import pandas as pd

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, ToPILImage, Resize
from torchvision.transforms import functional as F

from src.data.dataset import BinaryImageMeasDataset, AihubDataset, FashionDataset
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


class FashionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        transform,
        data_dir=paths.FASHION_DATA_DIR,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.test_size = 0.1

        self.root_img_dir = os.path.join(data_dir, "Img")
        anno_dir = os.path.join(data_dir, "Anno_coarse")
        self.category_attr_path = os.path.join(anno_dir, "list_category_cloth.txt")
        self.category_path = os.path.join(anno_dir, "list_category_img.txt")
        self.bbox_path = os.path.join(anno_dir, "list_bbox.txt")

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        category_attr = pd.read_csv(self.category_attr_path, sep="\s+", skiprows=1)
        self.idx2category = category_attr["category_name"].values.tolist()
        self.category2idx = {
            category: idx for idx, category in enumerate(self.idx2category)
        }

        category = pd.read_csv(self.category_path, sep="\s+", skiprows=1)

        bbox = pd.read_csv(self.bbox_path, sep="\s+", skiprows=1)

        df = pd.merge(category, bbox, on="image_name")
        df["category_label"] = df["category_label"].apply(lambda x: x - 1)

        train_val_df, test_df = train_test_split(
            df, test_size=self.test_size, stratify=df["category_label"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.test_size,
            stratify=train_val_df["category_label"],
            random_state=42,
        )

        self.train_dataset = FashionDataset(
            df=train_df, root_img_dir=self.root_img_dir, transform=self.transform
        )
        self.val_dataset = FashionDataset(
            df=val_df, root_img_dir=self.root_img_dir, transform=self.transform
        )
        self.test_dataset = FashionDataset(
            df=test_df, root_img_dir=self.root_img_dir, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
