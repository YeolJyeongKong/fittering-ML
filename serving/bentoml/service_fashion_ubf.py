import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import bentoml
from bentoml.io import JSON
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from serving.bentoml.utils import feature, rds
from serving.bentoml import rds_info

# fashion_ubf_runner = bentoml.sklearn.get("ubf:latest").to_runner()

svc = bentoml.Service(
    "product_recommendation",
)

rds_conn = rds.connect(
    host=rds_info.host,
    user=rds_info.user,
    password=rds_info.password,
    db=rds_info.db,
    port=rds_info.port,
)


@svc.api(
    input=JSON(pydantic_model=feature.UserId),
    output=JSON(pydantic_model=feature.Product_Output),
)
def fashion_ubf(input: feature.UserId) -> feature.Product_Output:
    knn = KNeighborsClassifier(n_neighbors=5)

    userid_dict = input.dict()
    user_id = userid_dict["user_id"]

    cursor = rds_conn.cursor()
    users_df = rds.load_UserMeas(cursor)

    user_info = users_df[users_df["USER_ID"] == user_id]
    other_users = users_df[
        (users_df["USER_ID"] != user_id)
        & (users_df["GENDER"] == user_info["GENDER"].to_list()[0])
    ]

    knn.fit(
        other_users.drop(columns=["USER_ID", "GENDER", "PRODUCT_ID"]),
        other_users["PRODUCT_ID"],
    )
    probalities = knn.predict_proba(
        user_info.drop(columns=["USER_ID", "GENDER", "PRODUCT_ID"])
        .iloc[0]
        .values.reshape(1, -1)
    )[0]
    recommendation_products = np.random.choice(
        knn.classes_, 5, p=probalities, replace=False
    ).tolist()

    return {"product_ids": recommendation_products}
