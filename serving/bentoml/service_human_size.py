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
import pymysql
from omegaconf import OmegaConf
import hydra
from pydantic import BaseModel
import pyrootutils

from serving.bentoml.utils import feature, s3, bento_svc

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from extras import paths, constant
from src.utils import preprocess


(
    svc,
    segment_runner,
    autoencoder_runner,
    regression_runner,
    segment_preprocess,
    autoencoder_preprocess,
) = bento_svc.human_size_svc(root_dir)


@svc.on_startup
async def connect_s3(context: bentoml.Context):
    s3_obj = s3.connect(paths.S3_ACCESS_KEY_PATH)
    context.state["s3_obj"] = s3_obj


@svc.api(
    input=JSON(pydantic_model=feature.ImageS3Path),
    output=JSON(pydantic_model=feature.ImageS3Path),
)
def masking_user(
    input: feature.ImageS3Path, context: bentoml.Context
) -> feature.ImageS3Path:
    input = input.dict()
    front_path = input["front"]
    side_path = input["side"]

    front_masked_path = str(Path(front_path).parent / "front_masked.jpg")
    side_masked_path = str(Path(side_path).parent / "side_masked.jpg")
    front = Image.open(
        context.state["s3_obj"].get_object(
            Bucket=constant.BUCKET_NAME_HUMAN, Key=front_path
        )["Body"]
    )
    front_size = front.size
    side = Image.open(
        context.state["s3_obj"].get_object(
            Bucket=constant.BUCKET_NAME_HUMAN, Key=side_path
        )["Body"]
    )
    side_size = side.size

    front = segment_preprocess(front).unsqueeze(0)
    side = segment_preprocess(side).unsqueeze(0)
    masked = segment_runner.run(torch.cat([front, side], dim=0))

    front_str = preprocess.to_bytearray(masked[0], front_size)

    side_str = preprocess.to_bytearray(masked[1], side_size)

    context.state["s3_obj"].put_object(
        Bucket=constant.BUCKET_NAME_HUMAN,
        Key=front_masked_path,
        Body=front_str,
        ContentType="image/jpg",
    )
    context.state["s3_obj"].put_object(
        Bucket=constant.BUCKET_NAME_HUMAN,
        Key=side_masked_path,
        Body=side_str,
        ContentType="image/jpg",
    )

    return {"front": front_masked_path, "side": side_masked_path}


@svc.api(
    input=JSON(pydantic_model=feature.User),
    output=JSON(pydantic_model=feature.UserSize),
)
def human_size(input: feature.User, context: bentoml.Context) -> feature.UserSize:
    input = input.dict()
    front_path = input["front"]
    side_path = input["side"]

    front = Image.open(
        context.state["s3_obj"].get_object(
            Bucket=constant.BUCKET_NAME_HUMAN, Key=front_path
        )["Body"]
    ).convert("L")
    side = Image.open(
        context.state["s3_obj"].get_object(
            Bucket=constant.BUCKET_NAME_HUMAN, Key=side_path
        )["Body"]
    ).convert("L")

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
        "chest": pred[0][1],
        "waist": pred[0][2],
        "hip": pred[0][3],
        "thigh": pred[0][4],
        "arm": pred[0][5],
        "leg": pred[0][6],
        "shoulder": pred[0][7],
    }
