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
    rds_conn = rds.connect_()
    cursor = rds_conn.cursor()

    userid_dict = input.dict()
    user_id = userid_dict["user_id"]
    users_df = rds.load_UserMeas(cursor)

    user_info = users_df[users_df["USER_ID"] == user_id]
    other_users = users_df[
        (users_df["USER_ID"] != user_id)
        & (users_df["GENDER"] == user_info["GENDER"].to_list()[0])
    ]

    recommendation_products = sklearn_model.knn_predict(
        user_info, other_users, n_neighbors=10, n_recommendations=5
    )
    rds_conn.close()

    return {"product_ids": recommendation_products}
