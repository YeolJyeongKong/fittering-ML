import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchmetrics

import bentoml
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.lightning_modules import CNNForwardModule, AutoEncoderModule
from src.data.datamodule import DataModule
from src.inference import encoder_inference
from src import utils


def train_ProductModule(cfg: DictConfig, wandb_logger):
    dm = hydra.utils.instantiate(cfg.data.product_encode)
    dm.prepare_data()
    dm.setup()

    module = hydra.utils.instantiate(cfg.model.product_encode)

    utils.print_wandb_run(cfg)

    val_samples = next(iter(dm.callback_dataloader()))

    trainer = hydra.utils.instantiate(
        cfg.trainer.product_encode,
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="val_loss"),
            ModelCheckpoint(
                dirpath="./model_weights/product_encode.ckpt", monitor="val_loss"
            ),
            utils.ProductLogger(val_samples=val_samples, idx2category=dm.idx2category),
        ],
    )

    trainer.fit(module, datamodule=dm)
    trainer.test(datamodule=dm)

    bentoml_product = bentoml.pytorch.save_model("product_encode", module.model.encoder)

    return str(bentoml_product.tag)
