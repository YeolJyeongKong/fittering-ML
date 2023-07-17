import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
import warnings

warnings.filterwarnings(action="ignore")

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader

import os
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge

from lightning_modules import CNNForwardModule, AutoEncoderModule
from datamodule import DataModule
from utils.callbacks import (
    ImagePredictionLogger,
    BetaPredictionLogger,
    MeasurementsLogger,
    RealDataPredictLogger,
)

import config


def train_CNNForwardModule():
    batch_size = 8
    epochs = 50

    dm = DataModule(batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    module = CNNForwardModule(learning_rate=2e-4)
    wandb_logger = WandbLogger(project="wandb-lightning", job_type="train")
    val_samples = next(iter(dm.val_dataloader()))

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_mae"),
            ModelCheckpoint(),
            MeasurementsLogger(
                val_samples, batch_size=batch_size, device=module.device
            ),
            RealDataPredictLogger(real_user_dir=config.REAL_USER_DIR),
        ],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)
    wandb.finish()


def train_AutoEncoderModule():
    batch_size = 16
    epochs = 50

    dm = DataModule(batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    module = AutoEncoderModule()
    wandb_logger = WandbLogger(project="wandb-lightning", job_type="train")

    val_samples = next(iter(dm.val_dataloader()))

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(),
            ImagePredictionLogger(val_samples=val_samples),
        ],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)
    wandb.finish()


def train_LinearRegression():
    train_dataset = np.load(os.path.join(config.GEN_ENCODE_DIR, "train.npz"))
    test_dataset = np.load(os.path.join(config.GEN_ENCODE_DIR, "test.npz"))
    # svr = SVR()
    svr = LinearRegression()
    # svr = KernelRidge(alpha=0.1, kernel="poly", degree=2)
    # svr.fit(test_dataset["x"][:10], test_dataset["y"][:10])
    # pred = svr.predict(test_dataset["x"])
    svr = svr.fit(train_dataset["x"], train_dataset["y"])
    train_mae = mean_absolute_error(train_dataset["y"], svr.predict(train_dataset["x"]))
    test_mae = mean_absolute_error(test_dataset["y"], svr.predict(test_dataset["x"]))

    print(f"train score: {train_mae}, test score: {test_mae}")
    pickle.dump(svr, open(os.path.join(config.MODEL_WEIGHTS_DIR, "svr.pickle"), "wb"))


if __name__ == "__main__":
    # train_AutoEncoderModule()
    # train_CNNForwardModule()
    train_LinearRegression()
