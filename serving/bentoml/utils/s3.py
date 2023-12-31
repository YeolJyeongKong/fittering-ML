from io import BytesIO
import pandas as pd
import PIL
from PIL import Image
import asyncio
import boto3
from aiobotocore.session import get_session
import torch
import pyrootutils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from extras import paths, constant


def prefix_exits(bucket, path, s3_obj):
    res = s3_obj.list_objects_v2(Bucket=bucket, Prefix=path, MaxKeys=1)
    return "Contents" in res


def connect(s3_access_key_path):
    try:
        s3_access_key = pd.read_csv(s3_access_key_path)
        s3 = boto3.client(
            "s3",
            aws_access_key_id=s3_access_key["Access key ID"].values[0],
            aws_secret_access_key=s3_access_key["Secret access key"].values[0],
            region_name="ap-northeast-2",
        )
    except:
        s3 = boto3.client("s3")

    return s3


async def load_img_(url, client, preprocess):
    response = await client.get_object(
        Bucket=constant.S3_BUCKET_NAME_PRODUCT,
        Key=constant.S3_BUCKET_PATH_PRODUCT + url,
    )
    obj = await response["Body"].read()
    try:
        image_obj = Image.open(BytesIO(obj))
        if isinstance(image_obj, PIL.GifImagePlugin.GifImageFile):
            imgs_lst = []
            for i in range(image_obj.n_frames):
                image_obj.seek(i)
                imgs_lst.append(preprocess(image_obj.convert("RGB")))
            return torch.stack(imgs_lst, dim=0), image_obj.n_frames

        return preprocess(Image.open(BytesIO(obj))).unsqueeze(0), 1
    except:
        raise Exception("Image open error")


async def load_img(urls, preprocess):
    session = get_session()
    try:
        s3_access_key = pd.read_csv(paths.S3_ACCESS_KEY_PATH)

        async with session.create_client(
            "s3",
            aws_access_key_id=s3_access_key["Access key ID"].values[0],
            aws_secret_access_key=s3_access_key["Secret access key"].values[0],
            region_name="ap-northeast-2",
        ) as client:
            tasks = [load_img_(url, client, preprocess) for url in urls]
            img_nframe_lst = await asyncio.gather(*tasks)
            imgs, nframes = [], []
            for img, nframe in img_nframe_lst:
                imgs.append(img)
                nframes.append(nframe)
            return torch.cat(imgs, dim=0), nframes
    except:
        async with session.create_client("s3") as client:
            tasks = [load_img_(url, client, preprocess) for url in urls]
            img_nframe_lst = await asyncio.gather(*tasks)
            imgs, nframes = [], []
            for img, nframe in img_nframe_lst:
                imgs.append(img)
                nframes.append(nframe)
            return torch.cat(imgs, dim=0), nframes
