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

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from extras import paths, constant
from src.utils import preprocess
from src.data.datamodule import DataModule
from serving.bentoml import feature, load, rds_info


svc, product_encode_runner, product_encode_preprocess = load.product_recommendation_svc(
    root_dir
)

s3 = load.s3(paths.S3_ACCESS_KEY_PATH)

rds_conn = load.rds(
    host=rds_info.host,
    user=rds_info.user,
    password=rds_info.password,
    db=rds_info.db,
    port=rds_info.port,
)


@svc.api(
    input=JSON(pydantic_model=feature.Product_Input),
    output=JSON(pydantic_model=feature.Product_Output),
)
def fashion_cbf(input: feature.Product_Input) -> feature.Product_Output:
    top_k = 3
    recommendation_n = 2

    product_dict = input.dict()
    product_ids = product_dict["product_ids"]
    product_gender = product_dict["gender"]

    if not product_ids:
        cursor = rds_conn.cursor()
        query = f"""
            SELECT * FROM product
        """
        cursor.execute(query)
        products = cursor.fetchall()
        products_df = pd.DataFrame(products)

        product_top_view = products_df[
            (products_df["gender"] == "A") | (products_df["gender"] == product_gender)
        ].sort_values(by="view", ascending=False)

        recommendation_products = (
            product_top_view[:top_k]["product_id"].sample(n=recommendation_n).to_list()
        )

    return {"product_id": recommendation_products}
