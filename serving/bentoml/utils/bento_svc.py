import os
import sys
import pandas as pd
import bentoml
import hydra
from omegaconf import OmegaConf


def human_size_svc(root_dir):
    output_dir = os.environ["OUTPUT_DIR"]
    cfg = OmegaConf.load(os.path.join(root_dir, output_dir, ".hydra/config.yaml"))

    segment_preprocess = hydra.utils.instantiate(cfg.preprocess.segment)
    segment_runner = bentoml.pytorch.get("segment:latest").to_runner()

    autoencoder_preprocess = hydra.utils.instantiate(cfg.preprocess.autoencoder)
    autoencoder_runner = bentoml.pytorch_lightning.get("autoencoder:latest").to_runner()
    del sys.modules["prometheus_client"]

    regression_runner = bentoml.sklearn.get("regression:latest").to_runner()

    svc = bentoml.Service(
        "human_size_predict",
        runners=[segment_runner, autoencoder_runner, regression_runner],
    )
    return (
        svc,
        segment_runner,
        autoencoder_runner,
        regression_runner,
        segment_preprocess,
        autoencoder_preprocess,
    )


def product_recommendation_svc(root_dir):
    output_dir = os.environ["OUTPUT_DIR"]
    cfg = OmegaConf.load(os.path.join(root_dir, output_dir, ".hydra/config.yaml"))

    product_encode_preprocess = hydra.utils.instantiate(cfg.preprocess.product_encode)
    product_encode_runner = bentoml.pytorch.get("product_encode:latest").to_runner()
    # del sys.modules["prometheus_client"]

    svc = bentoml.Service(
        "product_recommendation",
        runners=[product_encode_runner],
    )
    return (svc, product_encode_runner, product_encode_preprocess)