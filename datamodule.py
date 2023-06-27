import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from data.dataset import BinaryImageBetaDataset
from data.augmentation import AugmentBetasCam
from data.preprocessing import *


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data_range=None, test_data_range=None,
                 train_data_dir="/home/shin/VScodeProjects/fittering-ML/data/source/amass_up3d_3dpw_train.npz",
                 test_data_dir="/home/shin/VScodeProjects/fittering-ML/data/source/up3d_3dpw_val.npz"):
        super().__init__()
        self.batch_size = batch_size
        
        self.train_data_range = train_data_range
        self.test_data_range = test_data_range

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)), 
            transforms.ToTensor()
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = BinaryImageBetaDataset(ord_data_path=self.train_data_dir, data_range=self.train_data_range,
                                     augment=AugmentBetasCam(device=torch.device('cuda'), t_z_range=[0, 0], t_xy_std=0), 
                                     transform=self.transform)
            full_length = len(full_dataset)
            train_len = int(full_length * 0.8)
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_len, full_length-train_len])
        
        if stage == 'test' or stage is None:
            self.test_dataset = BinaryImageBetaDataset(ord_data_path=self.test_data_dir, data_range=self.test_data_range, 
                                     augment=AugmentBetasCam(device=torch.device('cuda'), t_z_range=[0, 0], t_xy_std=0, 
                                                             K_std=0, betas_std_vect=0), 
                                     transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)