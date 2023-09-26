import math
import logging
from typing import Dict, Any
from tqdm import tqdm
from bentoml.io import JSON
import asyncio
import torch
import bentoml
import pyrootutils

from serving.bentoml.utils import feature, rds, vector_db, bento_svc, utils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from serving.bentoml import rds_info
from extras.constant import *


bentoml_logger = logging.getLogger()
bentoml_logger.setLevel(logging.WARNING)

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

    imgs_encoded_lst = []
    for i in tqdm(
        range(math.ceil(products_df.shape[0] / INFERENCE_BATCH_SIZE)), desc="Inference"
    ):
        imgs_tensor, nframes = asyncio.run(
            utils.load_img(
                products_df["image"].to_list()[
                    i * INFERENCE_BATCH_SIZE : (i + 1) * INFERENCE_BATCH_SIZE
                ],
                product_encode_preprocess,
            )
        )

        imgs_encoded = product_encode_runner.run(imgs_tensor)
        imgs_encoded = utils.mean_nframe_encoded(imgs_encoded, nframes)
        imgs_encoded_lst.append(imgs_encoded)

    imgs_encoded_tensor = torch.cat(imgs_encoded_lst, dim=0)

    if vector_db.exist_collection(MILVUS_COLLECTION_NAME):
        vector_db.add_vector(
            collection_name=MILVUS_COLLECTION_NAME,
            embedded=imgs_encoded_tensor.numpy(),
            product_id=products_df["product_id"].to_list(),
            gender=products_df["gender"].to_list(),
        )
    else:
        vector_db.save_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            embedded=imgs_encoded_tensor.numpy(),
            product_id=products_df["product_id"].to_list(),
            gender=products_df["gender"].to_list(),
        )

    return {"status": 200}


@svc.api(
    input=JSON(pydantic_model=feature.Product_Input),
    output=JSON(pydantic_model=feature.Product_Output),
)
def fashion_cbf(
    input: feature.Product_Input, context: bentoml.Context
) -> feature.Product_Output:
    product_dict = input.dict()
    product_ids = product_dict["product_ids"]
    product_gender = product_dict["gender"]

    if product_ids:
        recommendation_products = vector_db.search_vector(
            MILVUS_COLLECTION_NAME,
            product_ids,
            product_gender,
            TOP_K,
            RECOMMENDATION_N,
        )

    else:
        cursor = context.state["rds_conn"].cursor()
        products_df = rds.load_Product(cursor)

        product_top_view = products_df[
            (products_df["gender"] == "A") | (products_df["gender"] == product_gender)
        ].sort_values(by="view", ascending=False)

        recommendation_products = (
            product_top_view[:TOP_K]["product_id"].sample(n=RECOMMENDATION_N).to_list()
        )

    return {"product_ids": recommendation_products}
