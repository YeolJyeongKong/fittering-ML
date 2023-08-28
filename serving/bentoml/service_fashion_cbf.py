from typing import Dict, Any
import pandas as pd
import torchvision.transforms.functional as F
from bentoml.io import JSON
import asyncio
import pyrootutils

from serving.bentoml.utils import feature, s3, rds, vector_db, bento_svc

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from serving.bentoml import rds_info


svc, product_encode_runner, product_encode_preprocess = bento_svc.product_recommendation_svc(
    root_dir
)

rds_conn = rds.connect(
    host=rds_info.host,
    user=rds_info.user,
    password=rds_info.password,
    db=rds_info.db,
    port=rds_info.port,
)


@svc.api(input=JSON(), output=JSON())
def product_encode(input: Dict[str, Any]) -> Dict[str, Any]:
    cursor = rds_conn.cursor()
    products_df = rds.load_ProductImageGender(cursor)

    imgs_tensor = asyncio.run(
        s3.load_img(products_df["URL"].to_list(), product_encode_preprocess)
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
        products_df = rds.load_Product(cursor)

        product_top_view = products_df[
            (products_df["gender"] == "A") | (products_df["gender"] == product_gender)
        ].sort_values(by="view", ascending=False)

        recommendation_products = (
            product_top_view[:top_k]["product_id"].sample(n=recommendation_n).to_list()
        )

    return {"product_ids": recommendation_products}