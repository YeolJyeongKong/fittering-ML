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
) = bento_svc.product_recommendation_svc()


@svc.api(input=JSON(pydantic_model=feature.NewProductId), output=JSON())
def product_encode(
    input: feature.NewProductId, context: bentoml.Context
) -> Dict[str, Any]:
    vector_db.connect()
    rds_conn = rds.connect()
    cursor = rds_conn.cursor()

    input = input.dict()
    product_ids = input["product_ids"]

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

    vector_db.disconnect()
    rds_conn.close()

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
        vector_db.connect()

        recommendation_products = vector_db.search_vector(
            MILVUS_COLLECTION_NAME,
            product_ids,
            product_gender,
            TOP_K,
            RECOMMENDATION_N,
        )
        vector_db.disconnect()
    else:
        rds_conn = rds.connect()
        cursor = rds_conn.cursor()
        products_df = rds.load_Product(cursor)

        product_top_view = products_df[
            (products_df["gender"] == "A") | (products_df["gender"] == product_gender)
        ].sort_values(by="view", ascending=False)

        recommendation_products = (
            product_top_view[:TOP_K]["product_id"].sample(n=RECOMMENDATION_N).to_list()
        )
        rds_conn.close()

    return {"product_ids": recommendation_products}
