import numpy as np
from sklearn.metrics import mean_absolute_error

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import bentoml
import wandb
import hydra
from omegaconf import DictConfig
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src import utils
from extras import paths


def save_segment(cfg: DictConfig, saved=True):
    if saved:
        return "segment:latest"

    segment = hydra.utils.instantiate(cfg.model.segment)
    segment.load_state_dict(
        torch.load(paths.SEGMODEL_PATH, map_location=torch.device("cpu"))
    )
    bentoml_segment = bentoml.pytorch.save_model(
        "segment",
        segment.to("cpu"),
        custom_objects={"preprocess": hydra.utils.instantiate(cfg.preprocess.segment)},
    )
    return str(bentoml_segment.tag)


def train_AutoEncoderModule(cfg: DictConfig, wandb_logger):
    dm = hydra.utils.instantiate(cfg.data.autoencoder)
    dm.prepare_data()
    dm.setup()

    module = hydra.utils.instantiate(cfg.model.autoencoder)

    utils.print_wandb_run(cfg)

    val_samples = next(iter(dm.val_dataloader()))

    trainer = hydra.utils.instantiate(
        cfg.trainer.autoencoder,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(
                dirpath="./model_weights",
                filename="human_size.ckpt",
                monitor="val_loss",
            ),
            utils.ImagePredictionLogger(val_samples=val_samples),
        ],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)

    bentoml_autoencoder = bentoml.pytorch_lightning.save_model(
        "autoencoder",
        module,
        custom_objects={"preprocess": dm.transform},
    )

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

    wandb.log({"regression_train_mae": train_mae, "regression_test_mae": test_mae})

    bentoml_regression = bentoml.sklearn.save_model("regression", reg)

    return str(bentoml_regression.tag)
