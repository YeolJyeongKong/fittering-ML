import os
from sklearn.metrics import mean_absolute_error

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import bentoml
import wandb
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pyrootutils

root_dir = pyrootutils.setup_root(os.curdir, indicator=".project-root", pythonpath=True)
from src import utils
from extras import paths


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    segment = hydra.utils.instantiate(cfg.model.segment)
    segment.load_state_dict(
        torch.load(paths.SEGMODEL_PATH, map_location=torch.device("cpu"))
    )
    bentoml_segment = bentoml.pytorch.save_model(
        "segment",
        segment.to("cpu"),
        custom_objects={"preprocess": hydra.utils.instantiate(cfg.preprocess.segment)},
        signatures={"__call__": {"batchable": True}},
    )


if __name__ == "__main__":
    main()
