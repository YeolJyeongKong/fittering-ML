import requests
from typing import Dict, Any
import pandas as pd
import torchvision.transforms.functional as F
from bentoml.io import JSON
import asyncio
import pyrootutils

from serving.bentoml.utils import feature, s3, rds, vector_db, bento_svc, utils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from serving.bentoml import rds_info


(
    svc,
    product_encode_runner,
    product_encode_preprocess,
) = bento_svc.product_recommendation_svc(root_dir)


@svc.api(input=JSON(pydantic_model=feature.NewProductId), output=JSON())
def product_encode(input: feature.NewProductId) -> Dict[str, Any]:
    rds_conn = rds.connect(
        host=rds_info.host,
        user=rds_info.user,
        password=rds_info.password,
        db=rds_info.db,
        port=rds_info.port,
    )
    if utils.local_check():
        vector_db.connect(host="13.125.214.45", port="19530")
    else:
        vector_db.connect(host="172.31.3.122", port="19530")

    input = input.dict()
    product_ids = input["product_ids"]

    cursor = rds_conn.cursor()
    products_df = rds.load_ProductImageGender(cursor, product_ids)

    imgs_tensor = asyncio.run(
        s3.load_img(products_df["URL"].to_list(), product_encode_preprocess)
    )
    imgs_encoded = product_encode_runner.run(imgs_tensor)

    if vector_db.exist_collection("image_embedding_collection"):
        vector_db.add_vector(
            collection_name="image_embedding_collection",
            embedded=imgs_encoded.numpy(),
            product_id=products_df["PRODUCT_ID"].to_list(),
            gender=products_df["GENDER"].to_list(),
        )
    else:
        vector_db.save_collection(
            collection_name="image_embedding_collection",
            embedded=imgs_encoded.numpy(),
            product_id=products_df["PRODUCT_ID"].to_list(),
            gender=products_df["GENDER"].to_list(),
        )

    vector_db.disconnect()
    rds_conn.close()
    return {"status": 200}


@svc.api(
    input=JSON(pydantic_model=feature.Product_Input),
    output=JSON(pydantic_model=feature.Product_Output),
)
def fashion_cbf(input: feature.Product_Input) -> feature.Product_Output:
    top_k = 10
    recommendation_n = 5

    product_dict = input.dict()
    product_ids = product_dict["product_ids"]
    product_gender = product_dict["gender"]

    rds_conn = rds.connect(
        host=rds_info.host,
        user=rds_info.user,
        password=rds_info.password,
        db=rds_info.db,
        port=rds_info.port,
    )
    if utils.local_check():
        vector_db.connect(host="13.125.214.45", port="19530")
    else:
        vector_db.connect(host="172.31.3.122", port="19530")

    if product_ids:
        recommendation_products = vector_db.search_vector(
            "image_embedding_collection",
            product_ids,
            product_gender,
            top_k,
            recommendation_n,
        )

    else:
        cursor = rds_conn.cursor()
        products_df = rds.load_Product(cursor)

        product_top_view = products_df[
            (products_df["gender"] == "A") | (products_df["gender"] == product_gender)
        ].sort_values(by="view", ascending=False)

        recommendation_products = (
            product_top_view[:top_k]["product_id"].sample(n=recommendation_n).to_list()
        )

    vector_db.disconnect()
    rds_conn.close()
    return {"product_ids": recommendation_products}
