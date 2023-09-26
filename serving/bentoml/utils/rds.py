import pandas as pd
import pymysql


def connect(host, user, password, db, port):
    rds = pymysql.connect(
        host=host,
        user=user,
        password=password,
        db=db,
        charset="utf8",
        port=port,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    return rds


def load_ProductImageGender(cursor, product_ids):
    if product_ids == []:
        query = f"""
            SELECT product_id, image, gender
            FROM product
        """
    else:
        product_ids = tuple(product_ids + [0])

        query = f"""
            SELECT product_id, image, gender
            FROM product
            WHERE product_id IN {product_ids};
        """
    cursor.execute(query)
    products = cursor.fetchall()
    products_df = pd.DataFrame(products)
    return products_df


def load_Product(cursor):
    query = f"""
        SELECT * FROM product
    """
    cursor.execute(query)
    products = cursor.fetchall()
    products_df = pd.DataFrame(products)
    return products_df


def load_UserMeas(cursor):
    query = f"""
        SELECT U.USER_ID AS USER_ID, U.GENDER AS GENDER, M.HEIGHT AS HEIGHT, M.WEIGHT AS WEIGHT, M.ARM AS ARM, M.LEG AS LEG, M.SHOULDER AS SHOULDER, M.WAIST AS WAIST, M.CHEST AS CHEST, M.THIGH AS THIGH, M.HIP AS HIP, U.PRODUCT_ID AS PRODUCT_ID
        FROM MEASUREMENT M
        INNER JOIN (
                SELECT U.USER_ID AS USER_ID, U.MEASUREMENT_ID AS MEASUREMENT_ID, U.GENDER AS GENDER, R.PRODUCT_ID AS PRODUCT_ID
                FROM USER U
                INNER JOIN (SELECT * FROM RECENT) R
                ON U.USER_ID = R.USER_ID
            ) U
        ON M.MEASUREMENT_ID = U.MEASUREMENT_ID
    """
    cursor.execute(query)
    users = cursor.fetchall()
    users_df = pd.DataFrame(users)
    return users_df
