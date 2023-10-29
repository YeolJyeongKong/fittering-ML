import logging
import bentoml
from bentoml.io import JSON
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from serving.bentoml.utils import feature, rds, sklearn_model


svc = bentoml.Service(
    "fashion-ubf",
)
bentoml_logger = logging.getLogger("bentoml")
bentoml_logger.setLevel(logging.DEBUG)


@svc.api(
    input=JSON(pydantic_model=feature.UserId),
    output=JSON(pydantic_model=feature.Product_Output),
)
def fashion_ubf(
    input: feature.UserId, context: bentoml.Context
) -> feature.Product_Output:
    rds_conn = rds.connect()
    cursor = rds_conn.cursor()

    userid_dict = input.dict()
    user_id = userid_dict["user_id"]
    users_df = rds.load_UserMeas(cursor)

    if user_id not in users_df["user_id"].to_list():
        context.response.status_code = 404
        error_msg = f"user_id:{user_id} is not found in Database"
        bentoml_logger.error(error_msg)
        return {"msg": error_msg}

    recommendation_products = sklearn_model.knn_predict(
        user_id, users_df, n_neighbors=10, n_recommendations=5
    )
    rds_conn.close()
    bentoml_logger.debug(f"response body: {recommendation_products}")

    return {"product_ids": recommendation_products}
