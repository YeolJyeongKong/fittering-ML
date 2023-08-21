import os
import sys
import pandas as pd
import bentoml
import hydra
import boto3
import pymysql
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
    del sys.modules["prometheus_client"]

    svc = bentoml.Service(
        "product_recommendation",
        runners=[product_encode_runner],
    )
    return (svc, product_encode_runner, product_encode_preprocess)


def s3(s3_access_key_path):
    try:
        s3_access_key = pd.read_csv(s3_access_key_path)
        s3 = boto3.client(
            "s3",
            aws_access_key_id=s3_access_key["Access key ID"].values[0],
            aws_secret_access_key=s3_access_key["Secret access key"].values[0],
            region_name="ap-northeast-2",
        )
    except:
        s3 = boto3.client("s3")

    return s3


def rds(host, user, password, db, port):
    rds = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset="utf8",
        port=port,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    return rds
