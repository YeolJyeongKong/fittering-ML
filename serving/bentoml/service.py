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

import torchvision.transforms.functional as F
import bentoml
from bentoml.io import JSON
import boto3
import pyrootutils
from pydantic import BaseModel

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from extras import paths

segmodel_runner = bentoml.pytorch.get("segmodel:latest").to_runner()
autoencoder_runner = bentoml.pytorch_lightning.get("autoencoder:latest").to_runner()
del sys.modules["prometheus_client"]
regression_runner = bentoml.sklearn.get("regression:latest").to_runner()
svc = bentoml.Service(
    "human_size_predict",
    runners=[segmodel_runner, autoencoder_runner, regression_runner],
)

# s3_access_key = pd.read_csv(paths.S3_ACCESS_KEY_PATH)
s3 = boto3.client(
    "s3",
    # aws_access_key_id=s3_access_key["Access key ID"].values[0],
    # aws_secret_access_key=s3_access_key["Secret access key"].values[0],
    # region_name="ap-northeast-2",
)
BUCKET_NAME = "fittering-measurements-images"


class ImageS3Path(BaseModel):
    front: str
    side: str


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

    front = F.pil_to_tensor(front)
    front = F.resize(front, size=(512, 512))
    front = front.unsqueeze(dim=0).float() / 255.0
    front_masked = segmodel_runner.run(front)[0]
    front_masked = F.resize(front_masked, size=front_size[::-1])
    front_masked = F.to_pil_image(front_masked).convert("RGB")
    buffered = BytesIO()
    front_masked.save(buffered, format="JPEG")
    front_str = buffered.getvalue()

    side = F.pil_to_tensor(side)
    side = F.resize(side, size=(512, 512))
    side = side.unsqueeze(dim=0).float() / 255.0
    side_masked = segmodel_runner.run(side)[0]
    side_masked = F.resize(side_masked, size=side_size[::-1])
    side_masked = F.to_pil_image(side_masked).convert("RGB")
    buffered = BytesIO()
    side_masked.save(buffered, format="JPEG")
    side_str = buffered.getvalue()

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
