from io import BytesIO
import pandas as pd
from PIL import Image
import asyncio
import aiobotocore
from aiobotocore.session import get_session
import torch
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras import paths, constant


# def load_img(cursor, s3, preprocess):
#     query = f"""
#         SELECT P.PRODUCT_ID AS PRODUCT_ID, I.URL AS URL, P.GENDER AS GENDER
#         FROM PRODUCT P
#         INNER JOIN (SELECT URL, PRODUCT_ID FROM IMAGEPATH WHERE THUMBNAIL = 1) I
#         ON P.PRODUCT_ID = I.PRODUCT_ID;
#     """
#     cursor.execute(query)
#     products = cursor.fetchall()
#     products_df = pd.DataFrame(products)

#     img_lst = []
# for i in range(len(products_df)):
#     print(products_df.iloc[i]["URL"])
#     img = Image.open(
#         s3.get_object(
#             Bucket=constant.BUCKET_NAME_PRODUCT, Key=products_df.iloc[i]["URL"]
#         )["Body"]
#     )
#     img_lst += [preprocess(img)]
# print(img_lst[:2])


async def load_img_(url, client, preprocess):
    response = await client.get_object(Bucket=constant.BUCKET_NAME_PRODUCT, Key=url)
    obj = await response["Body"].read()
    try:
        return preprocess(Image.open(BytesIO(obj)))
    except:
        print(obj)
        raise Exception("Image open error")


async def load_img(urls, preprocess):
    session = get_session()
    s3_access_key = pd.read_csv(paths.S3_ACCESS_KEY_PATH)

    async with session.create_client(
        "s3",
        aws_access_key_id=s3_access_key["Access key ID"].values[0],
        aws_secret_access_key=s3_access_key["Secret access key"].values[0],
        region_name="ap-northeast-2",
    ) as client:
        tasks = [load_img_(url, client, preprocess) for url in urls]
        img_lst = await asyncio.gather(*tasks)
        return torch.stack(img_lst, dim=0)
