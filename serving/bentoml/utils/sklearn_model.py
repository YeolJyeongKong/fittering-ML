import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def knn_predict(user_info, other_users, n_neighbors=5, n_recommendations=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

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
        knn.classes_, n_recommendations, p=probalities, replace=False
    ).tolist()

    return recommendation_products
