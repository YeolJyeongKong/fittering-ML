import bentoml
from bentoml.io import JSON
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from serving.bentoml.utils import feature, rds, sklearn_model


svc = bentoml.Service(
    "fashion-ubf",
)


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

    recommendation_products = sklearn_model.knn_predict(
        user_id, users_df, n_neighbors=10, n_recommendations=5
    )
    rds_conn.close()

    return {"product_ids": recommendation_products}
