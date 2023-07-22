import sys
from PIL import Image
from eval import predict, predict_time, Inference
import boto3
import json
import time

BUCKET_NAME = "fittering-measurements-images"
FRONT_FILE_NAME = "front.jpg"
SIDE_FILE_NAME = "side.jpg"
s3 = boto3.client("s3")

inference = Inference(cnnmodel_path="./model_weights/epoch=19-step=160000.ckpt")

# def handler(event, context):
# 	t1 = time.time()
# 	front_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=FRONT_FILE_NAME)['Body'])
# 	side_image = Image.open(s3.get_object(Bucket=BUCKET_NAME, Key=SIDE_FILE_NAME)['Body'])
# 	height = event['body-json']['height']
# 	t2 = time.time()
# 	ta, tb, tc, td = inference.predict(front_image, side_image, height)
# 	t3 = time.time()
# 	return {
# 		'statusCode': 200,
# 		'body': f"pred time: {t3-t2}, data load time: {t2 - t1}, total time: {t3-t1}, load model: {tb-ta}, segment: {tc-ta}, cnn: {td-tc}"
# 	}


def handler(event, context):
    front_image = Image.open(
        s3.get_object(Bucket=BUCKET_NAME, Key=FRONT_FILE_NAME)["Body"]
    )
    side_image = Image.open(
        s3.get_object(Bucket=BUCKET_NAME, Key=SIDE_FILE_NAME)["Body"]
    )
    height = event["body-json"]["height"]
    meas = inference.predict(front_image, side_image, height)
    return {"statusCode": 200, "body": meas}


if __name__ == "__main__":
    print(handler(None, None))
