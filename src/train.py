import os
import numpy as np
import pickle
import warnings
from tqdm import tqdm
import yaml

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
import hydra
from omegaconf import DictConfig, OmegaConf
import bentoml
import logging

log = logging.getLogger(__name__)

from src.models.lightning_modules import CNNForwardModule, AutoEncoderModule
from src.data.datamodule import DataModule
from src.inference import encoder_inference
from src import utils

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


def train_AutoEncoderModule(cfg: DictConfig):
    dm = hydra.utils.instantiate(cfg.data.autoencoder)
    dm.prepare_data()
    dm.setup()

    module = hydra.utils.instantiate(cfg.model.autoencoder)

    wandb_logger = hydra.utils.instantiate(cfg.logger.wandb)
    utils.print_wandb_run(cfg)

    val_samples = next(iter(dm.val_dataloader()))

    trainer = hydra.utils.instantiate(
        cfg.trainer.autoencoder,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(dirpath=cfg.paths.model_save_dir, filename="autoencoder"),
            utils.ImagePredictionLogger(val_samples=val_samples),
        ],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)

    bentoml_autoencoder = bentoml.pytorch_lightning.save_model("autoencoder", module)

    return module, dm, str(bentoml_autoencoder.tag)


def train_Regression(train, test, cfg: DictConfig):
    train_x, train_y = train
    test_x, test_y = test

    print(f"train data x shape = {train_x.shape}")
    print(f"train data y shape = {train_y.shape}")

    train_x = train_x[np.isnan(train_y).sum(axis=1) == 0]
    train_y = train_y[np.isnan(train_y).sum(axis=1) == 0]

    print(f"train data x shape = {train_x.shape}")
    print(f"train data y shape = {train_y.shape}")

    reg = hydra.utils.instantiate(cfg.model.regression)
    reg.fit(train_x, train_y)
    train_mae = mean_absolute_error(train_y, reg.predict(train_x))
    test_mae = mean_absolute_error(test_y, reg.predict(test_x))

    pickle.dump(
        reg, open(os.path.join(cfg.paths.model_save_dir, "regression.pickle"), "wb")
    )

    wandb.log({"regression_train_mae": train_mae, "regression_test_mae": test_mae})

    wandb.finish()
    bentoml_regression = bentoml.sklearn.save_model("regression", reg)

    return str(bentoml_regression.tag)


def save_segment(cfg: DictConfig, saved=True):
    if saved:
        return "segment:udnfvsbqigxhfzys"

    segment = hydra.utils.instantiate(cfg.model.segment)
    segment.load_state_dict(
        torch.load(paths.SEGMODEL_PATH, map_location=torch.device("cpu"))
    )
    bentoml_segment = bentoml.pytorch.save_model("segment", segment)
    return str(bentoml_segment.tag)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    segment_tag = save_segment(cfg, saved=True)

    module, datamodule, autoencoder_tag = train_AutoEncoderModule(cfg)
    (train_x, train_y), (test_x, test_y) = encoder_inference.encode(
        module, datamodule, device=torch.device("cuda")
    )

    regression_tag = train_Regression((train_x, train_y), (test_x, test_y), cfg)

    bentofile = OmegaConf.load(paths.BENTOFILE_DEFAULT_PATH)
    bentofile.models = [
        segment_tag,
        autoencoder_tag,
        regression_tag,
    ]

    bentofile_path = os.path.join(cfg.paths.output_dir, "bentofile.yaml")
    OmegaConf.save(config=bentofile, f=bentofile_path)


if __name__ == "__main__":
    # train_CNNForwardModule()
    # train_AutoEncoderModule()
    # train_LinearRegression()
    main()
