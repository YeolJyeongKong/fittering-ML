import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras.constant import *


def knn_predict(user_id, users_df, n_neighbors=5, n_recommendations=5):
    if (
        users_df[users_df["user_id"] == user_id][["height", "weight"]]
        .isna()
        .any()
        .any()
    ):
        return [], True
    users_df[MEAS_COLUMN] = users_df[MEAS_COLUMN].fillna(users_df[MEAS_COLUMN].mean())

    user_info = users_df[users_df["user_id"] == user_id]
    other_users = users_df[
        (users_df["user_id"] != user_id)
        & (users_df["gender"] == user_info["gender"].to_list()[0])
        & (users_df["product_id"].notna())
    ]
    if len(other_users) == 0:
        return [], False

    x = other_users.drop(columns=["user_id", "gender", "product_id"])
    y = other_users["product_id"]
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(other_users)))

    knn.fit(x, y)

    probalities = knn.predict_proba(
        user_info.drop(columns=["user_id", "gender", "product_id"])
        .iloc[0]
        .values.reshape(1, -1)
    )[0]
    recommendation_products = np.random.choice(
        knn.classes_,
        min(n_recommendations, len(knn.classes_)),
        p=probalities,
        replace=False,
    ).tolist()

    return recommendation_products, False
