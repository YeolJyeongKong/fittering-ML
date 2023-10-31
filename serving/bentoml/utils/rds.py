import pandas as pd
import pymysql
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from serving.bentoml import rds_info, rds_info_


def connect():
    rds = pymysql.connect(
        host=rds_info.host,
        user=rds_info.user,
        password=rds_info.password,
        db=rds_info.db,
        charset="utf8",
        port=rds_info.port,
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    return rds


def connect_():
    rds = pymysql.connect(
        host=rds_info_.host,
        user=rds_info_.user,
        password=rds_info_.password,
        db=rds_info_.db,
        charset="utf8",
        port=rds_info_.port,
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
        SELECT U.user_id AS user_id, U.gender AS gender, M.height AS height, M.weight AS weight, M.arm AS arm, M.leg AS leg, M.shoulder AS shoulder, M.waist AS waist, M.chest AS chest, M.thigh AS thigh, M.hip AS hip, U.product_id AS product_id
        FROM measurement M
        INNER JOIN (
                SELECT U.user_id AS user_id, U.measurement_id AS measurement_id, U.gender AS gender, R.product_id AS product_id
                FROM user U
                LEFT JOIN (SELECT user_id, product_id 
                            FROM recent
                            where timestamp in (SELECT MAX(timestamp) as timestamp FROM recent GROUP BY user_id)
                            ) R
                ON U.user_id = R.user_id
            ) U
        ON M.measurement_id = U.measurement_id
    """
    cursor.execute(query)
    users = cursor.fetchall()
    users_df = pd.DataFrame(users)
    return users_df
