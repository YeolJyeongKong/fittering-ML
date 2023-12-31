from PIL import Image, ImageOps
import torch
import bentoml
from bentoml.io import JSON
from botocore.exceptions import ClientError
import logging
import pyrootutils

from serving.bentoml.utils import feature, s3, bento_svc, utils

root_dir = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from extras import paths, constant


(
    svc,
    segment_runner,
    autoencoder_runner,
    regression_runner,
    segment_preprocess,
    autoencoder_preprocess,
) = bento_svc.human_size_svc()
bentoml_logger = logging.getLogger("bentoml")


@svc.api(
    input=JSON(pydantic_model=feature.ImageS3Path),
    output=JSON(pydantic_model=feature.ImageS3Path),
)
def masking_user(
    input: feature.ImageS3Path, context: bentoml.Context
) -> feature.ImageS3Path:
    s3_obj = s3.connect(paths.S3_ACCESS_KEY_PATH)

    input = input.dict()
    image_fname = input["image_fname"]

    image = Image.open(
        s3_obj.get_object(
            Bucket=constant.BUCKET_NAME_HUMAN,
            Key=constant.S3_BUCKET_PATH_BODY + image_fname,
        )["Body"]
    )
    image = ImageOps.exif_transpose(image)
    image_size = image.size

    image = segment_preprocess(image).unsqueeze(0)
    masked = segment_runner.run(image)

    image_str = utils.to_bytearray(masked[0], image_size)

    s3_obj.put_object(
        Bucket=constant.BUCKET_NAME_HUMAN,
        Key=constant.S3_BUCKET_PATH_SILHOUETTE + image_fname,
        Body=image_str,
        ContentType="image/jpg",
    )

    return {"image_fname": image_fname}


def prefix_exits(bucket, path, s3_obj):
    res = s3_obj.list_objects_v2(Bucket=bucket, Prefix=path, MaxKeys=1)
    return "Contents" in res


@svc.api(
    input=JSON(pydantic_model=feature.User),
    output=JSON(pydantic_model=feature.UserSize),
)
def human_size(input: feature.User, context: bentoml.Context) -> feature.UserSize:
    s3_obj = s3.connect(paths.S3_ACCESS_KEY_PATH)
    input = input.dict()
    front_fname = input["front"]
    side_fname = input["side"]

    try:
        front = Image.open(
            s3_obj.get_object(
                Bucket=constant.BUCKET_NAME_HUMAN,
                Key=constant.S3_BUCKET_PATH_SILHOUETTE + front_fname,
            )["Body"]
        ).convert("L")
        side = Image.open(
            s3_obj.get_object(
                Bucket=constant.BUCKET_NAME_HUMAN,
                Key=constant.S3_BUCKET_PATH_SILHOUETTE + side_fname,
            )["Body"]
        ).convert("L")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            context.response.status_code = 404
            error_msg = f"{e.response['Error']['Key']} is not found in s3"
            bentoml_logger.error(error_msg)
            return {"msg": error_msg}

    front = autoencoder_preprocess(front).unsqueeze(dim=0)
    side = autoencoder_preprocess(side).unsqueeze(dim=0)

    encoded = autoencoder_runner.run(front, side)

    height = torch.tensor(input["height"]).reshape((1, 1))
    weight = torch.tensor(input["weight"]).reshape((1, 1))
    sex = torch.tensor(float(input["sex"] == "M")).reshape((1, 1))
    z = torch.cat(
        [encoded[0].cpu(), encoded[1].cpu(), height, weight, sex], dim=1
    ).numpy()

    pred = regression_runner.run(z)

    return {
        "height": round(height[0][0].tolist(), 2),
        "weight": round(weight[0][0].tolist(), 2),
        "chest": round(pred[0][1], 2),
        "waist": round(pred[0][2], 2),
        "hip": round(pred[0][3], 2),
        "thigh": round(pred[0][4], 2),
        "arm": round(pred[0][5], 2),
        "leg": round(pred[0][6], 2),
        "shoulder": round(pred[0][7], 2),
    }
