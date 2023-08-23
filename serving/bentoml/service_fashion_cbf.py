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
import asyncio
import pymysql
from omegaconf import OmegaConf
import hydra
from pydantic import BaseModel
import pyrootutils

from serving.bentoml.utils import feature, load, s3_image, vector_db

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from extras import paths, constant
from src.data.datamodule import DataModule
from serving.bentoml import rds_info


svc, product_encode_runner, product_encode_preprocess = load.product_recommendation_svc(
    root_dir
)

# s3 = load.aios3(paths.S3_ACCESS_KEY_PATH)

rds_conn = load.rds(
    host=rds_info.host,
    user=rds_info.user,
    password=rds_info.password,
    db=rds_info.db,
    port=rds_info.port,
)


@svc.api(input=JSON(), output=JSON())
def product_encode(input: Dict[str, Any]) -> Dict[str, Any]:
    cursor = rds_conn.cursor()
    query = f"""
        SELECT P.PRODUCT_ID AS PRODUCT_ID, I.URL AS URL, P.GENDER AS GENDER
        FROM PRODUCT P
        INNER JOIN (SELECT URL, PRODUCT_ID FROM IMAGEPATH WHERE THUMBNAIL = 1) I
        ON P.PRODUCT_ID = I.PRODUCT_ID;
    """
    cursor.execute(query)
    products = cursor.fetchall()
    products_df = pd.DataFrame(products)

    imgs_tensor = asyncio.run(
        s3_image.load_img(products_df["URL"].to_list(), product_encode_preprocess)
    )
    imgs_encoded = product_encode_runner.run(imgs_tensor)
    vector_db.save_vector(
        embedded=imgs_encoded.numpy(),
        product_id=products_df["PRODUCT_ID"].to_list(),
        gender=products_df["GENDER"].to_list(),
    )

    return {"status": 200}


@svc.api(
    input=JSON(pydantic_model=feature.Product_Input),
    output=JSON(pydantic_model=feature.Product_Output),
)
def fashion_cbf(input: feature.Product_Input) -> feature.Product_Output:
    print("-----------------------------------------------------")
    top_k = 3
    recommendation_n = 2

    product_dict = input.dict()
    product_ids = product_dict["product_ids"]
    product_gender = product_dict["gender"]

    if product_ids:
        recommendation_products = vector_db.search_vector(
            product_ids, product_gender, top_k, recommendation_n
        )

    else:
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
