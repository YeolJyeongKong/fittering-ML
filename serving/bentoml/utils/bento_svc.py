import os
import sys
import bentoml
from omegaconf import OmegaConf
import pyrootutils

ROOT_DIR = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras import paths


def human_size_svc(root_dir):
    segment_model = bentoml.pytorch.get("segment:latest")
    segment_preprocess = segment_model.custom_objects["preprocess"]
    segment_runner = segment_model.to_runner()

    autoencoder_model = bentoml.pytorch_lightning.get("autoencoder:latest")
    autoencoder_preprocess = autoencoder_model.custom_objects["preprocess"]
    autoencoder_runner = autoencoder_model.to_runner()
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
    bentofile_yml = OmegaConf.load(paths.BENTOFILE_BEST_SWEEP_PATH)
    bento_model = bentoml.pytorch.get(bentofile_yml.models[0])
    product_encode_preprocess = bento_model.custom_objects["preprocess"]
    product_encode_runner = bento_model.to_runner()
    # del sys.modules["prometheus_client"]

    svc = bentoml.Service(
        "fashion-cbf",
        runners=[product_encode_runner],
    )
    return (svc, product_encode_runner, product_encode_preprocess)
