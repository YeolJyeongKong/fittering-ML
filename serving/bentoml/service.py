import os
import json
import sys
from typing import Dict, Any
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import base64
from io import BytesIO

import torch
import torchvision.transforms.functional as F
import bentoml
from bentoml.io import JSON
import boto3
import pyrootutils
from omegaconf import OmegaConf
import hydra
from pydantic import BaseModel

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from extras import paths
from src.utils import preprocess
from src.data.datamodule import DataModule


# output_dir = "outputs/2023-08-01/10-55-57"
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

try:
    s3_access_key = pd.read_csv(paths.S3_ACCESS_KEY_PATH)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=s3_access_key["Access key ID"].values[0],
        aws_secret_access_key=s3_access_key["Secret access key"].values[0],
        region_name="ap-northeast-2",
    )
except:
    s3 = boto3.client("s3")

BUCKET_NAME = "fittering-measurements-images"


class ImageS3Path(BaseModel):
    front: str = "0/front.jpg"
    side: str = "0/side.jpg"


@svc.api(
    input=JSON(pydantic_model=ImageS3Path), output=JSON(pydantic_model=ImageS3Path)
)
def masking(input: ImageS3Path) -> ImageS3Path:
    input = input.dict()
    front_path = input["front"]
    side_path = input["side"]

    front_masked_path = str(Path(front_path).parent / "front_masked.jpg")
    side_masked_path = str(Path(side_path).parent / "side_masked.jpg")
    front = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=front_path)["Body"])
    front_size = front.size
    side = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=side_path)["Body"])
    side_size = side.size

    front = segment_preprocess(front).unsqueeze(0)
    side = segment_preprocess(side).unsqueeze(0)
    masked = segment_runner.run(torch.cat([front, side], dim=0))

    front_str = preprocess.to_bytearray(masked[0], front_size)

    side_str = preprocess.to_bytearray(masked[1], side_size)

    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=front_masked_path,
        Body=front_str,
        ContentType="image/jpg",
    )
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=side_masked_path,
        Body=side_str,
        ContentType="image/jpg",
    )

    return {"front": front_masked_path, "side": side_masked_path}


class User(BaseModel):
    front: str = "0/front_masked.jpg"
    side: str = "0/side_masked.jpg"
    height: float = 177
    weight: float = 65
    sex: str = "M"


class UserSize(BaseModel):
    height: float
    chest_circumference: float
    waist_circumference: float
    hip_circumference: float
    thigh_left_circumference: float
    arm_left_length: float
    inside_leg_height: float
    shoulder_breadth: float


@svc.api(input=JSON(pydantic_model=User), output=JSON(pydantic_model=UserSize))
def human_size(input: User) -> UserSize:
    input = input.dict()
    front_path = input["front"]
    side_path = input["side"]

    front = Image.open(
        s3.get_object(Bucket=BUCKET_NAME, Key=front_path)["Body"]
    ).convert("L")
    side = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=side_path)["Body"]).convert(
        "L"
    )

    front = autoencoder_preprocess(front).unsqueeze(dim=0)
    side = autoencoder_preprocess(side).unsqueeze(dim=0)

    encoded = autoencoder_runner.run(front, side)

    height = torch.tensor(input["height"]).reshape((1, 1))
    weight = torch.tensor(input["weight"]).reshape((1, 1))
    sex = torch.tensor(float(input["sex"] == "M")).reshape((1, 1))
    z = torch.cat(
        [encoded[0].cpu(), encoded[1].cpu(), height, weight, sex], dim=1
    ).numpy()

    pred = regression_runner.run(z)

    return {
        "height": pred[0][0],
        "chest_circumference": pred[0][1],
        "waist_circumference": pred[0][2],
        "hip_circumference": pred[0][3],
        "thigh_left_circumference": pred[0][4],
        "arm_left_length": pred[0][5],
        "inside_leg_height": pred[0][6],
        "shoulder_breadth": pred[0][7],
    }
