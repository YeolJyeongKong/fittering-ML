import os
import numpy as np
import pickle
import warnings

warnings.filterwarnings(action="ignore")
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

import wandb

from src.models.lightning_modules import CNNForwardModule, AutoEncoderModule
from src.data.datamodule import DataModule
from utils.callbacks import (
    ImagePredictionLogger,
    BetaPredictionLogger,
    MeasurementsLogger,
    RealDataPredictLogger,
)

from extras import paths


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
    epochs = 30

    dm = DataModule(batch_size=batch_size, dataset_mode="aihub")
    dm.prepare_data()
    dm.setup()

    module = AutoEncoderModule(model_mode="front")
    wandb_logger = WandbLogger(project="wandb-lightning", job_type="train")

    val_samples = next(iter(dm.val_dataloader()))

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(),
            ImagePredictionLogger(val_samples=val_samples, model_mode="front"),
        ],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)
    wandb.finish()


def train_LinearRegression():
    train_dataset = np.load(os.path.join(paths.AIHUB_ENCODED_DIR, "train.npz"))
    test_dataset = np.load(os.path.join(paths.AIHUB_ENCODED_DIR, "test.npz"))
    train_x, train_y = train_dataset["x"], train_dataset["y"]
    test_x, test_y = test_dataset["x"], test_dataset["y"]
    train_x = train_x[np.isnan(train_y).sum(axis=1) == 0]
    train_y = train_y[np.isnan(train_y).sum(axis=1) == 0]

    print(f"train data x shape = {train_dataset['x'].shape}")
    print(f"train data y shape = {train_dataset['y'].shape}")

    reg = KernelRidge(alpha=0.2, kernel="poly", degree=3)
    reg.fit(train_x, train_y)
    train_mae = mean_absolute_error(train_y, reg.predict(train_x))
    test_mae = mean_absolute_error(test_y, reg.predict(test_x))

    print(f"train score: {train_mae}, test score: {test_mae}")
    pickle.dump(reg, open(os.path.join(paths.MODEL_WEIGHTS_DIR, "reg.pickle"), "wb"))


if __name__ == "__main__":
    # train_CNNForwardModule()
    # train_AutoEncoderModule()
    train_LinearRegression()
