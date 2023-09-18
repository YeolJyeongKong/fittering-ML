import requests
from typing import Dict, Any
import pandas as pd
import torchvision.transforms.functional as F
from bentoml.io import JSON
import asyncio
import torch
import bentoml
import pyrootutils

from serving.bentoml.utils import feature, s3, rds, vector_db, bento_svc, utils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from serving.bentoml import rds_info
from extras.constant import *


(
    svc,
    product_encode_runner,
    product_encode_preprocess,
) = bento_svc.product_recommendation_svc(root_dir)


@svc.on_startup
async def connect_db(context: bentoml.Context):
    if utils.local_check():
        vector_db.connect(host=MILVUS_PUBLIC_HOST, port=MILVUS_PORT)
    else:
        vector_db.connect(host=MILVUS_PRIVATE_HOST, port=MILVUS_PORT)

    rds_conn = rds.connect(
        host=rds_info.host,
        user=rds_info.user,
        password=rds_info.password,
        db=rds_info.db,
        port=rds_info.port,
    )
    context.state["rds_conn"] = rds_conn


@svc.on_shutdown
async def disconnect_db(context: bentoml.Context):
    vector_db.disconnect()
    context.state["rds_conn"].close()


@svc.api(input=JSON(pydantic_model=feature.NewProductId), output=JSON())
def product_encode(
    input: feature.NewProductId, context: bentoml.Context
) -> Dict[str, Any]:
    input = input.dict()
    product_ids = input["product_ids"]

    cursor = context.state["rds_conn"].cursor()
    products_df = rds.load_ProductImageGender(cursor, product_ids)

    imgs_tensor, nframes = asyncio.run(
        s3.load_img(products_df["URL"].to_list(), product_encode_preprocess)
    )

    imgs_encoded = []
    for i in range(imgs_tensor.shape[0] // 16 + 1):
        imgs_encoded.append(
            product_encode_runner.run(imgs_tensor[i * 16 : (i + 1) * 16])
        )
    imgs_encoded = torch.cat(imgs_encoded, dim=0)
    imgs_encoded = utils.mean_nframe_encoded(imgs_encoded, nframes)

    if vector_db.exist_collection(MILVUS_COLLECTION_NAME):
        vector_db.add_vector(
            collection_name=MILVUS_COLLECTION_NAME,
            embedded=imgs_encoded.numpy(),
            product_id=products_df["PRODUCT_ID"].to_list(),
            gender=products_df["GENDER"].to_list(),
        )
    else:
        vector_db.save_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            embedded=imgs_encoded.numpy(),
            product_id=products_df["PRODUCT_ID"].to_list(),
            gender=products_df["GENDER"].to_list(),
        )

    return {"status": 200}


@svc.api(
    input=JSON(pydantic_model=feature.Product_Input),
    output=JSON(pydantic_model=feature.Product_Output),
)
def fashion_cbf(
    input: feature.Product_Input, context: bentoml.Context
) -> feature.Product_Output:
    top_k = 12
    recommendation_n = 12

    product_dict = input.dict()
    product_ids = product_dict["product_ids"]
    product_gender = product_dict["gender"]

    if product_ids:
        recommendation_products = vector_db.search_vector(
            MILVUS_COLLECTION_NAME,
            product_ids,
            product_gender,
            top_k,
            recommendation_n,
        )

    else:
        cursor = context["rds_conn"].cursor()
        products_df = rds.load_Product(cursor)

        product_top_view = products_df[
            (products_df["gender"] == "A") | (products_df["gender"] == product_gender)
        ].sort_values(by="view", ascending=False)

        recommendation_products = (
            product_top_view[:top_k]["product_id"].sample(n=recommendation_n).to_list()
        )

    return {"product_ids": recommendation_products}
