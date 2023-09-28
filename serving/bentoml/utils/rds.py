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
